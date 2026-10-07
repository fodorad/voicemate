import asyncio
import http.server
import os
import tempfile
import threading
import unittest
from pathlib import Path

from tests.helpers import make_tool_context
from voicemate.agent.tools import EGRESS_TOOLS, build_tools
from voicemate.agent.tools.base import ToolError
from voicemate.events import NotesChangedEvent, ToolEvent

PAGE = b"""<html><head><title>LoRA explained</title></head><body>
<nav>Home | About</nav>
<article><h1>LoRA explained</h1>
<p>Low-rank adaptation freezes the pretrained weights and trains small low-rank matrices.
This makes fine-tuning large language models cheap and memory efficient.</p>
<p>It is widely used for adapting models such as Llama and Qwen to new tasks.</p></article>
<footer>Copyright</footer></body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self):  # noqa: N802
        type(self).requests += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


class ToolTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tool_context(self.tmp.name)
        self.tools = {t.name: t for t in build_tools(self.ctx)}
        self.events = []
        self.run_config = {"configurable": {"emit": self.events.append}}

    async def asyncTearDown(self):
        await self.ctx.http.aclose()
        self.ctx.notes.close()
        self.tmp.cleanup()

    async def call(self, name, **args):
        return await self.tools[name].ainvoke(args, config=self.run_config)

    def tool_events(self):
        return [e for e in self.events if isinstance(e, ToolEvent)]


class TestRegistry(ToolTestCase):
    async def test_all_tools_have_descriptions_and_no_config_in_schema(self):
        expected = {
            "get_time", "calculator", "get_weather", "web_search", "fetch_page",
            "arxiv_search", "download_paper", "remember", "recall_memory", "add_note",
            "list_notes", "complete_note", "list_files", "find_files", "read_file", "write_file",
        }  # fmt: skip
        self.assertEqual(set(self.tools), expected)
        self.assertTrue(EGRESS_TOOLS <= expected)
        for tool in self.tools.values():
            self.assertTrue(tool.description)
            self.assertNotIn("config", tool.tool_call_schema.model_json_schema()["properties"])


class TestLocalTools(ToolTestCase):
    async def test_time_and_calculator(self):
        self.assertIn("2026-09-28 21:30", await self.call("get_time"))
        self.assertEqual(await self.call("calculator", expression="23*42"), "23*42 = 966")
        self.assertEqual([e.source for e in self.tool_events()], ["local", "local"])

    async def test_calculator_error_is_raised_as_tool_error(self):
        with self.assertRaises(ToolError):
            await self.call("calculator", expression="__import__('os')")


class TestWebSearch(ToolTestCase):
    async def test_second_similar_search_is_served_from_memory(self):
        first = await self.call("web_search", query="mixture of experts survey")
        self.assertIn("MoE survey", first)
        self.assertIn("Untrusted", first)
        second = await self.call("web_search", query="mixture of experts survey")
        self.assertIn("[from memory, retrieved", second)
        self.assertEqual(self.ctx.search.queries, ["mixture of experts survey"])
        self.assertEqual([e.source for e in self.tool_events()], ["web", "memory"])

    async def test_results_become_recallable_documents(self):
        await self.call("web_search", query="mixture of experts survey")
        recalled = await self.call("recall_memory", query="mixture of experts routes tokens")
        self.assertIn("Research memory", recalled)
        self.assertIn("https://example.org/moe", recalled)


class TestFetchPage(ToolTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        _Handler.requests = 0
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/lora"
        self.ctx.offered_urls.add(self.url)  # as if a search had listed it

    async def asyncTearDown(self):
        await asyncio.to_thread(self.server.shutdown)  # blocks ~0.5 s; keep the loop free
        self.server.server_close()
        await super().asyncTearDown()

    async def test_page_is_extracted_then_cached(self):
        text = await self.call("fetch_page", url=self.url)
        self.assertIn("Low-rank adaptation", text)
        self.assertNotIn("Copyright", text)
        again = await self.call("fetch_page", url=self.url)
        self.assertIn("[from memory", again)
        self.assertEqual(_Handler.requests, 1)

    async def test_non_http_url_is_rejected(self):
        with self.assertRaises(ToolError):
            await self.call("fetch_page", url="file:///etc/passwd")


class TestMemoryTools(ToolTestCase):
    async def test_remember_and_recall(self):
        self.assertEqual(
            await self.call("remember", fact="Adam prefers short answers"), "Remembered."
        )
        self.assertIn(
            "Adam prefers short answers", await self.call("recall_memory", query="short answers")
        )
        self.assertIn("Nothing relevant", await self.call("recall_memory", query="zzz qqq"))


class TestNotesTools(ToolTestCase):
    async def test_add_list_complete_emit_changes(self):
        out = await self.call(
            "add_note", text="call the doctor", kind="reminder", due="2026-10-01T09:00"
        )
        self.assertIn("reminder #1", out)
        self.assertIn("call the doctor", await self.call("list_notes", kind="reminder"))
        self.assertIn("call the doctor", await self.call("list_notes", query="doctor"))
        self.assertIn("done", await self.call("complete_note", note_id=1))
        self.assertEqual(await self.call("list_notes"), "No matching notes.")
        self.assertEqual(sum(isinstance(e, NotesChangedEvent) for e in self.events), 2)

    async def test_invalid_input_becomes_tool_error(self):
        with self.assertRaises(ToolError):
            await self.call("add_note", text="x", kind="shopping")
        with self.assertRaises(ToolError):
            await self.call("complete_note", note_id=42)


class TestFileTools(ToolTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.docs = Path(self.tmp.name) / "Documents"
        (self.docs / "cv").mkdir()
        (self.docs / "cv" / "resume.md").write_text("# Adam\nML engineer")
        (self.docs / ".secret").write_text("hidden")

    async def test_list_find_read(self):
        listing = await self.call("list_files", path=str(self.docs))
        self.assertIn("cv/", listing)
        self.assertNotIn(".secret", listing)
        self.assertIn("resume.md", await self.call("find_files", pattern="resume"))
        self.assertIn("resume.md", await self.call("find_files", pattern="*.md"))
        self.assertIn(
            "ML engineer", await self.call("read_file", path=str(self.docs / "cv" / "resume.md"))
        )
        self.assertIn("No files", await self.call("find_files", pattern="nothing-like-this"))

    async def test_relative_paths_are_inside_the_first_root(self):
        self.assertIn("ML engineer", await self.call("read_file", path="cv/resume.md"))

    async def test_write_creates_new_files_and_folders(self):
        target = self.docs / "notes" / "summary.md"
        self.assertIn("Wrote", await self.call("write_file", path=str(target), content="v1"))
        self.assertEqual(target.read_text(), "v1")
        # Replacing an existing file needs the user's answer (tests/agent/test_graph.py).

    async def test_folder_and_oversized_writes_are_rejected(self):
        with self.assertRaises(ToolError):
            await self.call("write_file", path=str(self.docs / "cv"), content="x")
        with self.assertRaises(ToolError):
            await self.call("write_file", path=str(self.docs / "big.txt"), content="x" * 1_000_001)

    async def test_sandbox_escapes_are_denied(self):
        outside = Path(self.tmp.name) / "outside.txt"
        outside.write_text("private")
        (self.docs / "link").symlink_to(outside)
        for path in (
            str(outside),
            str(self.docs / ".." / "outside.txt"),
            str(self.docs / "link"),
            "../outside.txt",
            str(self.docs / ".secret"),
        ):
            with self.subTest(path=path), self.assertRaises(ToolError):
                await self.call("read_file", path=path)
        with self.assertRaises(ToolError):
            await self.call("write_file", path=str(outside), content="x")
        self.assertEqual(outside.read_text(), "private")

    async def test_binary_and_missing_files(self):
        (self.docs / "blob.bin").write_bytes(b"\x00\x01\x02")
        with self.assertRaises(ToolError):
            await self.call("read_file", path=str(self.docs / "blob.bin"))
        with self.assertRaises(ToolError):
            await self.call("read_file", path=str(self.docs / "missing.txt"))
        with self.assertRaises(ToolError):
            await self.call("list_files", path=str(self.docs / "cv" / "resume.md"))

    async def test_pdf_is_read_as_text(self):
        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with (self.docs / "blank.pdf").open("wb") as handle:
            writer.write(handle)
        self.assertIn(
            "PDF, 1 pages", await self.call("read_file", path=str(self.docs / "blank.pdf"))
        )


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION"), "network")
class TestNetworkTools(ToolTestCase):  # pragma: no cover - integration only
    async def test_weather_home_location(self):
        out = await self.call("get_weather", days=2)
        self.assertIn("Gyöngyös", out)
        self.assertEqual(self.tool_events()[-1].source, "web")

    async def test_arxiv_search_and_download(self):
        out = await self.call(
            "arxiv_search", query="attention is all you need transformer", max_results=2
        )
        self.assertIn("[", out)
        out = await self.call("download_paper", arxiv_id="1706.03762")
        self.assertIn("voicemate-papers", out)
        self.assertTrue(any(self.ctx.config.files.papers_path.glob("1706.03762*.pdf")))


if __name__ == "__main__":
    unittest.main()
