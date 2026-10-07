"""UI tests: the real ConversationPage in an offline NiceGUI client (no browser, no server).

Widgets are inspected directly after events flow through the page's real event pump.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from nicegui import Client, core, ui
from nicegui.page import page as nicegui_page

from tests.helpers import (
    ScriptedRecognizer,
    ToneSynthesizer,
    make_scripted_chat_model,
    make_tool_context,
    tool_call,
)
from voicemate.agent.graph import build_graph
from voicemate.agent.model_select import ModelChoice
from voicemate.agent.tools import build_tools
from voicemate.audio.vad import EnergyVAD
from voicemate.events import (
    ConfirmEvent,
    ErrorEvent,
    HoldEvent,
    Lane,
    LaneState,
    MetricsEvent,
    NotesChangedEvent,
    StatusEvent,
    ToolEvent,
    TranscriptEvent,
)
from voicemate.pipeline.orchestrator import ChatLine, VoiceSession
from voicemate.runtime import Runtime
from voicemate.tts.voices import VoiceBank
from voicemate.ui.app import ConversationPage, Server, handle_control_message


def texts(client: Client) -> list[str]:
    return [e.text for e in client.elements.values() if isinstance(e, ui.label)]


def buttons(client: Client) -> list[str]:
    return [e.text for e in client.elements.values() if isinstance(e, ui.button)]


class UITestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        core.loop = asyncio.get_running_loop()
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tool_context(self.tmp.name)
        self.tasks: list[asyncio.Task] = []

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        if hasattr(self, "session"):
            await self.session.close()
        await self.ctx.http.aclose()
        self.ctx.notes.close()
        self.tmp.cleanup()

    def runtime(self, responses=(), choice=None):
        voice = ToneSynthesizer(24000)
        graph = build_graph(
            make_scripted_chat_model(list(responses)),
            build_tools(self.ctx),
            self.ctx.config,
            self.ctx.memory,
            InMemorySaver(),
        )
        return Runtime(
            config=self.ctx.config,
            recognizer=ScriptedRecognizer([]),
            voices=VoiceBank({"en": voice, "hu": voice}, default_lang="en"),
            vad_factory=EnergyVAD,
            graph=graph,
            memory=self.ctx.memory,
            notes=self.ctx.notes,
            model_choice=choice,
        )

    def open_page(self, responses=(), choice=None, on_new_conversation=None) -> ConversationPage:
        self.session = VoiceSession(self.runtime(responses, choice))
        self.client = Client(nicegui_page("/"))
        with self.client:
            page = ConversationPage(self.session, self.ctx.config, on_new_conversation)
        self.tasks.append(asyncio.create_task(page.pump()))
        return page

    def apply(self, page: ConversationPage, event) -> None:
        with page.root:
            page.apply(event)

    async def settle(self):
        await asyncio.wait_for(self.session.wait_idle(), 10)
        await asyncio.sleep(0.05)  # let the pump render the last events


class TestLayout(UITestCase):
    async def test_header_rail_and_model(self):
        choice = ModelChoice("fast", "gemma4:e4b", True, False, "fits")
        page = self.open_page(choice=choice)
        shown = texts(self.client)
        self.assertIn("Ava", shown)
        self.assertIn("gemma4:e4b", shown)
        self.assertEqual(len(page.stations), 8)
        self.assertIn("Turning it into speech", shown)

    async def test_header_buttons(self):
        page = self.open_page()
        self.assertEqual(buttons(self.client)[:3], ["New conversation", "Hold", "Start microphone"])
        self.assertIn("vm-mic", page.mic.classes)  # the Ctrl+M shortcut clicks this button
        self.assertIn("vm-hold", page.hold.classes)  # and Ctrl+H this one

    async def test_hold_until_finished(self):
        page = self.open_page(["Got all of it."])
        with page.root:
            await page._toggle_hold()
        await asyncio.sleep(0.05)  # the page follows the session's HoldEvent
        self.assertTrue(self.session.holding)
        self.assertEqual(page.hold.text, "I'm finished")
        self.assertIn("is-on", page.hold.classes)
        page.input.value = "first part"
        await page.send()
        with page.root:
            await page._toggle_hold()
        await self.settle()
        self.assertFalse(self.session.holding)
        self.assertEqual(page.hold.text, "Hold")
        self.assertNotIn("is-on", page.hold.classes)
        self.assertIn("Got all of it.", [t.strip() for t in texts(self.client)])


class TestEvents(UITestCase):
    async def test_typed_message_and_streamed_reply(self):
        page = self.open_page(["Hello Adam, nice to meet you."])
        page.input.value = "Hi there"
        await page.send()
        await self.settle()
        shown = texts(self.client)
        self.assertIn("Hi there", shown)
        self.assertIn("Hello Adam, nice to meet you.", [t.strip() for t in shown])
        self.assertIn("Whole turn", page.latency.text)
        self.assertEqual(page.input.value, "")
        self.assertFalse(page.empty.visible)

    async def test_merged_question_replaces_the_cut_off_bubble(self):
        page = self.open_page()
        self.apply(page, TranscriptEvent("tell me a story", final=True, lang="en"))
        self.apply(
            page, TranscriptEvent("tell me a story, short", final=True, lang="en", merged=True)
        )
        user_bubbles = [t for t in texts(self.client) if t.startswith("tell me")]
        self.assertEqual(user_bubbles, ["tell me a story, short"])

    async def test_lanes_show_active_done_and_duration(self):
        page = self.open_page()
        self.apply(page, StatusEvent(Lane.THINKING, LaneState.ACTIVE, ts=100.0))
        station, ms, _ = page.stations[Lane.THINKING]
        self.assertIn("is-active", station.classes)
        self.apply(page, StatusEvent(Lane.THINKING, LaneState.DONE, ts=100.25))
        self.assertIn("is-done", station.classes)
        self.assertEqual(ms.text, "250 ms")
        self.apply(page, StatusEvent(Lane.TOOL, LaneState.ACTIVE, "web_search"))
        self.assertEqual(page.stations[Lane.TOOL][2].text, "web_search")
        self.apply(page, StatusEvent(Lane.LISTENING, LaneState.ACTIVE))  # a new turn resets
        self.assertNotIn("is-done", station.classes)
        self.assertEqual(ms.text, "")

    async def test_trace_badges_and_errors(self):
        page = self.open_page()
        self.apply(page, ToolEvent("web_search", "end", source="web", summary="moe survey"))
        self.apply(page, ToolEvent("web_search", "start"))  # starts are not traced
        self.apply(page, ErrorEvent("Turn failed: timed out"))
        shown = texts(self.client)
        self.assertIn("sent to web", shown)
        self.assertIn("web_search: moe survey", shown)
        self.assertIn("Turn failed: timed out", shown)

    async def test_metrics_and_notes_refresh(self):
        page = self.open_page()
        self.apply(page, MetricsEvent({"first_audio_s": 0.86, "total_s": 2.1, "tools": ["calc"]}))
        self.assertIn("First words after 0.9 s", page.latency.text)
        self.assertIn("Tools: calc", page.latency.text)
        self.ctx.notes.add("call Anna", "todo")
        self.apply(page, NotesChangedEvent())
        await asyncio.sleep(0.05)  # NiceGUI runs refreshes on the next loop iteration
        self.assertIn("call Anna", texts(self.client))

    async def test_microphone_toggle_messages(self):
        page = self.open_page()
        with page.root:
            page._mic_toggled(SimpleNamespace(args=True))
            self.assertIn("is-on", page.mic.classes)
            page._mic_toggled(SimpleNamespace(args=False))
            self.assertNotIn("is-on", page.mic.classes)
            self.assertEqual(page.mic.text, "Start microphone")


class TestConversation(UITestCase):
    async def test_reopened_page_shows_the_history(self):
        page = self.open_page()
        page.show_history(
            [
                ChatLine("user", "Tell me a story"),
                ChatLine("assistant", "Once upon a time", interrupted=True),
                ChatLine("user", "shorter please"),
            ]
        )
        self.assertEqual(
            [t for t in texts(self.client) if t in ("Tell me a story", "Once upon a time")],
            ["Tell me a story", "Once upon a time"],
        )
        bubble = next(
            e
            for e in self.client.elements.values()
            if isinstance(e, ui.label) and e.text == "Once upon a time"
        )
        self.assertIn("interrupted", bubble.classes)
        self.assertFalse(page.empty.visible)
        self.assertEqual(page.last_user.text, "shorter please")

    async def test_new_conversation_button_starts_a_fresh_thread(self):
        started = []
        page = self.open_page(on_new_conversation=lambda: started.append(True))
        with page.root:
            page._new_conversation()
        self.assertEqual(started, [True])


class TestConfirmation(UITestCase):
    async def test_dialog_opens_and_allow_replaces_the_file(self):
        target = Path(self.tmp.name) / "Documents" / "plan.md"
        target.write_text("old")
        call = tool_call("write_file", {"path": str(target), "content": "new"})
        page = self.open_page([call, "Replaced it."])
        await self.session.submit_text("Save my plan")
        await self.settle()
        self.assertTrue(page.dialog.value)
        self.assertIn("already exists", page.question.text)
        await page.answer("yes")
        await self.settle()
        self.assertFalse(page.dialog.value)
        self.assertEqual(target.read_text(), "new")

    async def test_hold_follows_the_session(self):
        page = self.open_page()
        self.apply(page, HoldEvent(True))
        self.assertEqual(page.hold.text, "I'm finished")
        self.apply(page, HoldEvent(False))  # e.g. the microphone was turned off
        self.assertEqual(page.hold.text, "Hold")

    async def test_closing_event_closes_the_dialog(self):
        page = self.open_page()
        self.apply(page, ConfirmEvent("Replace x?"))
        self.assertTrue(page.dialog.value)
        self.apply(page, ConfirmEvent("", open=False))
        self.assertFalse(page.dialog.value)


class TestServer(UITestCase):
    async def test_loading_page_while_models_load(self):
        server = Server(self.ctx.config)
        client = Client(nicegui_page("/"))
        with client:
            self.assertIsNone(await server.build_page())
        self.assertIn("Loading the speech and language models.", " ".join(texts(client)))

    def test_conversation_survives_a_restart_until_a_new_one_starts(self):
        server = Server(self.ctx.config)
        first = server.thread_id()
        self.assertEqual(first, server.thread_id())
        self.assertEqual(Server(self.ctx.config).thread_id(), first)  # e.g. after a restart
        server.new_conversation()
        self.assertNotEqual(server.thread_id(), first)
        self.assertEqual(Server(self.ctx.config).thread_id(), server.thread_id())

    async def test_open_session_restores_the_thread(self):
        server = Server(self.ctx.config, self.runtime(["Hi Adam."]))
        _, session = await server.open_session()
        await session.submit_text("Hello")
        await asyncio.wait_for(session.wait_idle(), 10)
        await session.close()
        sid, reopened = await server.open_session()
        self.assertIs(server.sessions[sid], reopened)
        self.assertEqual(reopened.thread_id, server.thread_id())
        lines = await reopened.restore()
        self.assertEqual([line.text for line in lines], ["Hello", "Hi Adam."])
        await server.shutdown()

    def test_fake_mic_needs_opt_in_and_a_plain_file_name(self):
        server = Server(self.ctx.config)
        folder = self.ctx.config.data_path / "fake_mic"
        folder.mkdir(parents=True)
        (folder / "hello.wav").write_bytes(b"RIFF")
        self.assertIsNone(server.fake_mic_name("hello.wav"))  # disabled by default
        server.config.ui.test_audio = True
        self.assertEqual(server.fake_mic_name("hello.wav"), "hello.wav")
        for bad in ("https://evil.example/cmd.wav", "../secret.wav", "missing.wav", None):
            with self.subTest(bad=bad):
                self.assertIsNone(server.fake_mic_name(bad))

    async def test_control_messages(self):
        session = VoiceSession(self.runtime())
        handle_control_message(session, '{"type": "playback", "state": "started"}')
        self.assertTrue(session.busy)
        handle_control_message(session, '{"type": "playback", "state": "ended"}')
        self.assertFalse(session.busy)
        events = session.bus.subscribe()
        handle_control_message(session, '{"type": "mic", "state": "off"}')
        stopped = events.get_nowait()
        self.assertEqual((stopped.lane, stopped.state), (Lane.LISTENING, LaneState.IDLE))
        handle_control_message(session, "not json")
        handle_control_message(session, '["a list"]')
        self.assertFalse(session.busy)


if __name__ == "__main__":
    unittest.main()
