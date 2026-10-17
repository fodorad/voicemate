import asyncio
import http.server
import tempfile
import threading
import unittest
from pathlib import Path

import httpx

from tests.helpers import HashingEmbedder, StaticArxivBackend, StaticSearchBackend
from voicemate.agent.tools.arxiv_tools import Paper
from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.research import Research
from voicemate.config import MemoryConfig
from voicemate.memory.store import MemoryStore


class FailingBackend:
    name = "broken"

    async def search(self, query: str, max_results: int):
        raise ConnectionError("rate limited")


PAGE = b"<html><body><article><p>Low-rank adaptation freezes the weights.</p></article></body>"
PDF = b"%PDF-not-really"


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


class ResearchTestCase(unittest.IsolatedAsyncioTestCase):
    def build(self, web=None, arxiv=None, max_results=5) -> Research:
        return Research(
            self.memory,
            web or self.web,
            arxiv or self.arxiv,
            max_results,
            self.http,
            allowed_private_hosts=frozenset({"127.0.0.1"}),
        )

    async def asyncSetUp(self):
        self.http = httpx.AsyncClient()
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(
            Path(self.tmp.name),
            HashingEmbedder(),
            MemoryConfig(search_cache_similarity=0.9, recall_min_similarity=0.2),
        )
        self.web = StaticSearchBackend()
        self.arxiv = StaticArxivBackend()
        self.research = self.build()

    async def asyncTearDown(self):
        await self.http.aclose()
        self.tmp.cleanup()


class TestWebSearch(ResearchTestCase):
    async def test_a_new_query_goes_to_the_backend(self):
        found = await self.research.web("mixture of experts survey")
        self.assertFalse(found.from_memory)
        self.assertEqual(found.backend, "static")
        self.assertEqual([r["url"] for r in found.results][:1], ["https://example.org/moe"])
        self.assertEqual(self.web.queries, ["mixture of experts survey"])

    async def test_the_same_query_is_then_answered_from_memory(self):
        await self.research.web("mixture of experts survey")
        again = await self.research.web("mixture of experts survey")
        self.assertTrue(again.from_memory)
        self.assertEqual(again.query, "mixture of experts survey")
        self.assertGreater(again.retrieved_at, 0)
        self.assertEqual(self.web.queries, ["mixture of experts survey"])  # no second request

    async def test_results_become_recallable_documents(self):
        await self.research.web("mixture of experts survey")
        hits = self.memory.search_documents("mixture of experts routes tokens")
        self.assertIn("https://example.org/moe", [h.meta.get("url") for h in hits])

    async def test_max_results_limits_the_backend(self):
        research = self.build(max_results=1)
        self.assertEqual(len((await research.web("anything at all")).results), 1)

    async def test_backend_failures_become_tool_errors(self):
        research = self.build(web=FailingBackend())
        with self.assertRaisesRegex(ToolError, "Web search failed: rate limited"):
            await research.web("anything")


class TestArxivSearch(ResearchTestCase):
    async def test_a_new_query_goes_to_arxiv_then_comes_from_memory(self):
        first = await self.research.papers("sparse experts routing", max_results=2)
        self.assertFalse(first.from_memory)
        self.assertEqual([r["arxiv_id"] for r in first.results], ["2401.00001", "2401.00002"])
        again = await self.research.papers("sparse experts routing", max_results=2)
        self.assertTrue(again.from_memory)
        self.assertEqual(self.arxiv.queries, ["sparse experts routing"])

    async def test_the_count_is_clamped_to_1_to_10(self):
        self.assertEqual(len((await self.research.papers("one", max_results=0)).results), 1)

    async def test_web_and_arxiv_caches_do_not_answer_each_other(self):
        await self.research.web("sparse experts routing")
        found = await self.research.papers("sparse experts routing", max_results=2)
        self.assertFalse(found.from_memory)

    async def test_failures_become_tool_errors(self):
        class Broken:
            name = "broken"

            async def search(self, query, max_results):
                raise TimeoutError("slow")

        research = self.build(arxiv=Broken())
        with self.assertRaisesRegex(ToolError, "arXiv search failed"):
            await research.papers("x", max_results=3)


class TestListedAddresses(ResearchTestCase):
    async def test_nothing_is_listed_before_a_search(self):
        self.assertFalse(self.research.is_listed("https://example.org/moe"))

    async def test_web_results_are_listed_fresh_and_from_memory(self):
        await self.research.web("mixture of experts survey")
        self.assertTrue(self.research.is_listed("https://example.org/moe"))
        other = self.build()
        self.assertFalse(other.is_listed("https://example.org/moe"))  # per instance
        await other.web("mixture of experts survey")  # now answered from memory
        self.assertTrue(other.is_listed("https://example.org/moe"))

    async def test_arxiv_results_list_their_abstract_pages(self):
        await self.research.papers("sparse experts routing", max_results=2)
        self.assertTrue(self.research.is_listed("https://arxiv.org/abs/2401.00001"))
        other = self.build()
        await other.papers("sparse experts routing", max_results=2)
        self.assertTrue(other.is_listed("https://arxiv.org/abs/2401.00002"))

    async def test_a_user_approved_address_is_listed_from_then_on(self):
        self.research.list_address("https://example.com/page")
        self.assertTrue(self.research.is_listed("https://example.com/page"))
        self.assertFalse(self.research.is_listed("https://example.com/page?d=secret"))


class TestPages(ResearchTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        _Handler.requests = 0
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/lora?d=private"
        self.asked: list[str] = []

    async def asyncTearDown(self):
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        await super().asyncTearDown()

    def allow(self, url: str) -> None:
        self.asked.append(url)
        return None

    def refuse(self, url: str) -> str:
        self.asked.append(url)
        return "no, not that one"

    async def test_an_unlisted_address_asks_first_and_yes_fetches_and_stores_it(self):
        page = await self.research.page(self.url, self.allow)
        self.assertEqual(self.asked, [self.url])
        self.assertFalse(page.from_memory)
        self.assertIn("Low-rank adaptation", page.text)
        self.assertEqual(_Handler.requests, 1)
        self.assertTrue(self.research.is_listed(self.url))  # no second question
        self.assertTrue(self.memory.documents_for_url(self.url))

    async def test_no_never_contacts_the_site_and_passes_the_words_on(self):
        with self.assertRaisesRegex(ToolError, "did not approve.*not that one"):
            await self.research.page(self.url, self.refuse)
        self.assertEqual(_Handler.requests, 0)
        self.assertFalse(self.research.is_listed(self.url))

    async def test_a_listed_address_never_asks(self):
        self.research.list_address(self.url)
        await self.research.page(self.url, self.refuse)
        self.assertEqual(self.asked, [])
        self.assertEqual(_Handler.requests, 1)

    async def test_a_stored_page_needs_neither_a_question_nor_a_request(self):
        self.research.list_address(self.url)
        await self.research.page(self.url, self.refuse)
        again = await self.build().page(self.url, self.refuse)  # a fresh process, same memory
        self.assertTrue(again.from_memory)
        self.assertGreater(again.retrieved_at, 0)
        self.assertEqual((self.asked, _Handler.requests), ([], 1))

    async def test_only_public_http_addresses_are_allowed(self):
        research = Research(self.memory, self.web, self.arxiv, 5, self.http)  # no test allowance
        for url in ("file:///etc/passwd", self.url):
            with self.subTest(url=url), self.assertRaises(ToolError):
                await research.page(url, self.allow)
        self.assertEqual(self.asked, [])

    async def test_a_download_failure_is_a_tool_error(self):
        url = "http://127.0.0.1:1/nothing-listens-here"
        self.research.list_address(url)
        with self.assertRaisesRegex(ToolError, "Could not fetch"):
            await self.research.page(url, self.allow)

    async def test_a_page_without_text_is_an_error(self):
        class Empty(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body></body></html>")

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Empty)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addAsyncCleanup(asyncio.to_thread, server.shutdown)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        self.research.list_address(url)
        with self.assertRaisesRegex(ToolError, "No readable text"):
            await self.research.page(url, self.allow)


class TestPapers(ResearchTestCase):
    async def test_a_paper_is_looked_up_by_id(self):
        paper = await self.research.get_paper("2401.00002")
        self.assertEqual(paper.title, "Routing without load balancing")

    async def test_an_unknown_paper_is_an_error(self):
        with self.assertRaisesRegex(ToolError, "No arXiv paper 9999.99999"):
            await self.research.get_paper("9999.99999")

    async def test_a_failing_lookup_is_an_error(self):
        class Broken:
            name = "broken"

            async def get(self, paper_id):
                raise TimeoutError("slow")

        with self.assertRaisesRegex(ToolError, "arXiv lookup failed"):
            await self.build(arxiv=Broken()).get_paper("2401.00001")

    async def test_the_full_text_is_indexed_once_next_to_the_abstract(self):
        await self.research.papers("sparse experts routing", max_results=2)  # stores abstracts
        paper = self.arxiv.papers[0]
        before = len(self.memory.documents_for_url(paper.abs_url))
        text = "Section one. " + "Experts are routed by a learned gate. " * 120
        await self.research.index_paper(paper, text)
        indexed = len(self.memory.documents_for_url(paper.abs_url))
        self.assertGreater(indexed, before)
        await self.research.index_paper(paper, text)
        self.assertEqual(len(self.memory.documents_for_url(paper.abs_url)), indexed)

    async def test_empty_text_is_not_indexed(self):
        paper = self.arxiv.papers[0]
        await self.research.index_paper(paper, "")
        self.assertEqual(self.memory.documents_for_url(paper.abs_url), [])

    async def test_the_pdf_is_downloaded_through_the_guarded_client(self):
        class Pdf(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.end_headers()
                self.wfile.write(PDF)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Pdf)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addAsyncCleanup(asyncio.to_thread, server.shutdown)
        paper = Paper(
            "1", "T", [], "2024-01-01", "s", f"http://127.0.0.1:{server.server_address[1]}/p"
        )
        self.assertEqual(await self.research.paper_pdf(paper), PDF)
        strict = Research(self.memory, self.web, self.arxiv, 5, self.http)
        with self.assertRaises(ToolError):
            await strict.paper_pdf(paper)


if __name__ == "__main__":
    unittest.main()
