import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.helpers import make_scripted_chat_model, make_tool_context, tool_call
from voicemate.agent.graph import build_graph
from voicemate.agent.tools import build_tools
from voicemate.events import ToolEvent
from voicemate.pipeline.conversation import (
    INTERRUPTED_MARK,
    INTERRUPTED_TOOL_RESULT,
    ChatLine,
    Conversation,
    ThreadStore,
)


class TestThreadStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "data" / "conversation.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_missing_file_starts_a_conversation_and_remembers_it(self):
        store = ThreadStore(self.path)
        first = store.current()
        self.assertTrue(first)
        self.assertEqual(store.current(), first)
        self.assertEqual(json.loads(self.path.read_text())["thread_id"], first)

    def test_a_restart_continues_the_same_conversation(self):
        first = ThreadStore(self.path).current()
        self.assertEqual(ThreadStore(self.path).current(), first)

    def test_a_new_conversation_replaces_the_current_one_for_everyone(self):
        store = ThreadStore(self.path)
        first = store.current()
        fresh = store.start_new()
        self.assertNotEqual(fresh, first)
        self.assertEqual(ThreadStore(self.path).current(), fresh)

    def test_unreadable_files_start_over_instead_of_crashing(self):
        self.path.parent.mkdir(parents=True)
        for broken in ("", "not json", "[]", '{"other": 1}', '{"thread_id": 5}'):
            with self.subTest(broken=broken):
                self.path.write_text(broken)
                self.assertTrue(ThreadStore(self.path).current())


class ConversationTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tool_context(self.tmp.name)

    async def asyncTearDown(self):
        await self.ctx.http.aclose()
        self.ctx.notes.close()
        self.tmp.cleanup()

    def graph(self, responses):
        return build_graph(
            make_scripted_chat_model(responses),
            build_tools(self.ctx),
            self.ctx.config,
            self.ctx.memory,
            InMemorySaver(),
        )

    def config(self, thread="t1"):
        return {"configurable": {"thread_id": thread}}


class TestLines(ConversationTestCase):
    async def test_a_new_thread_has_no_history(self):
        conversation = Conversation(self.graph(["x"]), "empty")
        self.assertEqual(await conversation.lines(), [])
        self.assertIsNone(await conversation.pending_question())

    async def test_lines_alternate_in_order_and_hide_tool_traffic(self):
        graph = self.graph([tool_call("get_time", {}), "It is noon.", "Fine."])
        config = self.config()
        await graph.ainvoke({"messages": [HumanMessage("What time is it?")], "lang": "en"}, config)
        await graph.ainvoke({"messages": [HumanMessage("Thanks")], "lang": "en"}, config)
        lines = await Conversation(graph, "t1").lines()
        self.assertEqual(
            lines,
            [
                ChatLine("user", "What time is it?"),
                ChatLine("assistant", "It is noon."),
                ChatLine("user", "Thanks"),
                ChatLine("assistant", "Fine."),
            ],
        )

    async def test_interrupted_replies_are_flagged_and_lose_the_marker(self):
        graph = self.graph(["x"])
        await graph.aupdate_state(
            self.config(),
            {
                "messages": [
                    HumanMessage("Tell me a story"),
                    AIMessage(f"Once upon a time {INTERRUPTED_MARK}"),
                ]
            },
            as_node="agent",
        )
        lines = await Conversation(graph, "t1").lines()
        self.assertEqual(lines[1], ChatLine("assistant", "Once upon a time", interrupted=True))

    async def test_threads_do_not_mix(self):
        graph = self.graph(["Hi."])
        await graph.ainvoke({"messages": [HumanMessage("Hello")], "lang": "en"}, self.config("a"))
        self.assertEqual(await Conversation(graph, "b").lines(), [])


class TestPendingQuestion(ConversationTestCase):
    async def test_a_paused_graph_exposes_its_question(self):
        target = Path(self.tmp.name) / "Documents" / "plan.md"
        target.write_text("old")
        call = tool_call("write_file", {"path": str(target), "content": "new"})
        graph = self.graph([call, "Done."])
        await graph.ainvoke({"messages": [HumanMessage("Save it")], "lang": "en"}, self.config())
        question = await Conversation(graph, "t1").pending_question()
        self.assertIn("already exists", question)
        self.assertIsNone(await Conversation(graph, "other").pending_question())


class TestClosingAnInterruptedTurn(ConversationTestCase):
    async def seed(self, *messages):
        await self.graph_.aupdate_state(
            self.config(), {"messages": list(messages)}, as_node="agent"
        )

    async def history(self):
        state = await self.graph_.aget_state(self.config())
        return state.values["messages"]

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.graph_ = self.graph(["x"])
        self.conversation = Conversation(self.graph_, "t1")

    def call_message(self, call_id="call_9", message_id="ai-with-call"):
        return AIMessage(
            "", tool_calls=[{"name": "get_time", "args": {}, "id": call_id}], id=message_id
        )

    async def test_call_message_is_kept_once_next_to_its_cancelled_result(self):
        await self.seed(HumanMessage("what time is it", id="u1"), self.call_message())
        await self.conversation.close_interrupted("")
        messages = await self.history()
        self.assertEqual([m.type for m in messages], ["human", "ai", "tool"])
        self.assertEqual(messages[1].tool_calls[0]["id"], messages[2].tool_call_id)
        self.assertEqual(messages[2].content, INTERRUPTED_TOOL_RESULT)

    async def test_answered_calls_are_left_alone(self):
        await self.seed(
            HumanMessage("time?", id="u1"),
            self.call_message("c1", "a1"),
            ToolMessage("noon", tool_call_id="c1"),
        )
        await self.conversation.close_interrupted("")
        messages = await self.history()
        self.assertEqual([m.type for m in messages], ["human", "ai", "tool"])
        self.assertEqual(messages[2].content, "noon")

    async def test_a_partial_reply_is_kept_and_marked(self):
        await self.seed(HumanMessage("story", id="u1"))
        await self.conversation.close_interrupted("  Once upon a time ")
        messages = await self.history()
        self.assertEqual(messages[-1].content, f"Once upon a time {INTERRUPTED_MARK}")

    async def test_closing_twice_changes_nothing_more(self):
        await self.seed(HumanMessage("what time is it", id="u1"), self.call_message())
        await self.conversation.close_interrupted("")
        once = [m.type for m in await self.history()]
        await self.conversation.close_interrupted("")
        self.assertEqual([m.type for m in await self.history()], once)

    async def test_a_clean_turn_needs_no_repair(self):
        await self.seed(HumanMessage("hi", id="u1"), AIMessage("hello", id="a1"))
        await self.conversation.close_interrupted("")
        self.assertEqual([m.type for m in await self.history()], ["human", "ai"])


class TestDiscard(ConversationTestCase):
    async def test_removes_a_cancelled_message_by_id(self):
        graph = build_graph(
            make_scripted_chat_model(["slow answer"], delay=5),
            build_tools(self.ctx),
            self.ctx.config,
            self.ctx.memory,
            InMemorySaver(),
        )
        conversation = Conversation(graph, "t1")

        async def consume():
            async for _ in conversation.say("drop me", "en", "d1", lambda event: None):
                pass

        task = asyncio.create_task(consume())
        for _ in range(300):  # wait until the message is checkpointed
            if await conversation.lines():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self.assertEqual([c.text for c in await conversation.lines()], ["drop me"])
        await conversation.discard("d1")
        self.assertEqual(await conversation.lines(), [])

    async def test_an_unknown_id_is_ignored(self):
        await Conversation(self.graph(["x"]), "t1").discard("missing")


class TestStreaming(ConversationTestCase):
    async def collect(self, stream) -> str:
        return "".join([chunk async for chunk in stream])

    async def test_say_streams_the_answer_text_and_records_the_message(self):
        conversation = Conversation(self.graph(["Hello there Adam."]), "t1")
        text = await self.collect(conversation.say("Hi", "en", "m1", lambda event: None))
        self.assertEqual(text.strip(), "Hello there Adam.")
        self.assertEqual(
            await conversation.lines(),
            [ChatLine("user", "Hi"), ChatLine("assistant", "Hello there Adam.")],
        )

    async def test_tool_traffic_is_not_streamed_but_events_reach_the_caller(self):
        events = []
        conversation = Conversation(self.graph([tool_call("get_time", {}), "It is noon."]), "t1")
        text = await self.collect(conversation.say("Time?", "en", "m1", events.append))
        self.assertEqual(text.strip(), "It is noon.")
        self.assertTrue(any(isinstance(e, ToolEvent) for e in events))

    async def test_a_question_ends_the_stream_and_resume_continues_it(self):
        target = Path(self.tmp.name) / "Documents" / "plan.md"
        target.write_text("old")
        call = tool_call("write_file", {"path": str(target), "content": "new"})
        conversation = Conversation(self.graph([call, "Replaced it."]), "t1")
        await self.collect(conversation.say("Save", "en", "m1", lambda event: None))
        self.assertIn("already exists", await conversation.pending_question())
        self.assertEqual(target.read_text(), "old")
        text = await self.collect(conversation.resume("yes", lambda event: None))
        self.assertEqual(text.strip(), "Replaced it.")
        self.assertEqual(target.read_text(), "new")
        self.assertIsNone(await conversation.pending_question())


class TestSessionIsFrameworkFree(unittest.TestCase):
    def test_the_session_imports_no_agent_framework(self):
        source = (
            Path(__file__).resolve().parents[2] / "voicemate" / "pipeline" / "orchestrator.py"
        ).read_text()
        for name in ("langgraph", "langchain"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
