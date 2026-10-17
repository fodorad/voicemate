import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.helpers import make_scripted_chat_model, make_tool_context, tool_call
from voicemate.agent.graph import build_graph
from voicemate.agent.tools import build_tools
from voicemate.pipeline.conversation import INTERRUPTED_MARK, ChatLine, Conversation, ThreadStore


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


if __name__ == "__main__":
    unittest.main()
