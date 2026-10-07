import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.helpers import make_scripted_chat_model, make_tool_context, tool_call
from voicemate.agent.graph import (
    ANSWER_NUDGE,
    NO_MORE_TOOLS,
    build_graph,
    recent_history,
    with_context,
)
from voicemate.agent.prompts import system_prompt, turn_context
from voicemate.agent.tools import build_tools
from voicemate.config import Config
from voicemate.events import Lane, StatusEvent, ToolEvent

NOW = datetime(2026, 9, 28, 21, 30).astimezone()


class GraphTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = make_tool_context(self.tmp.name)
        self.events = []

    async def asyncTearDown(self):
        await self.ctx.http.aclose()
        self.ctx.notes.close()
        self.tmp.cleanup()

    def graph(self, responses, max_tool_rounds=6, with_memory=True):
        self.llm = make_scripted_chat_model(responses)
        self.ctx.config.llm.max_tool_rounds = max_tool_rounds
        return build_graph(
            self.llm,
            build_tools(self.ctx),
            self.ctx.config,
            memory=self.ctx.memory if with_memory else None,
            checkpointer=InMemorySaver(),
            now=lambda: NOW,
        )

    def run_config(self, thread="t1"):
        return {"configurable": {"thread_id": thread, "emit": self.events.append}}

    async def turn(self, graph, text, lang="hu", thread="t1"):
        return await graph.ainvoke(
            {"messages": [HumanMessage(text)], "lang": lang}, self.run_config(thread)
        )

    def prompt_of(self, call_index):
        """Everything the model received in one call, joined."""
        return "\n".join(str(m.content) for m in self.llm.seen[call_index])


class TestGraph(GraphTestCase):
    async def test_plain_answer_in_requested_language(self):
        graph = self.graph(["Szia! Miben segíthetek?"])
        state = await self.turn(graph, "Szia")
        self.assertEqual(state["messages"][-1].content.strip(), "Szia! Miben segíthetek?")
        self.assertIn("Reply language: Hungarian", self.prompt_of(0))
        self.assertIn("2026-09-28 21:30", self.prompt_of(0))

    async def test_tool_round_trip_emits_start_and_end(self):
        graph = self.graph([tool_call("calculator", {"expression": "23*42"}), "It is 966."])
        state = await self.turn(graph, "Mennyi 23 szorozva 42?")
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(tool_messages[0].content, "23*42 = 966")
        phases = [(e.name, e.phase) for e in self.events if isinstance(e, ToolEvent)]
        self.assertEqual(phases, [("calculator", "start"), ("calculator", "end")])
        self.assertEqual(state["tool_rounds"], 1)

    async def test_empty_answer_after_a_tool_is_retried_once(self):
        graph = self.graph([tool_call("get_time", {}), "", "It is 12:43."])
        state = await self.turn(graph, "what time is it?", lang="en")
        self.assertEqual(state["messages"][-1].content.strip(), "It is 12:43.")
        self.assertIn(ANSWER_NUDGE, self.prompt_of(2))
        self.assertEqual(sum(m.type == "ai" for m in state["messages"]), 2)  # empty one dropped

    async def test_tool_errors_are_returned_to_the_model(self):
        graph = self.graph([tool_call("calculator", {"expression": "import os"}), "Sorry."])
        state = await self.turn(graph, "calc")
        tool_message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        self.assertIn("Error", tool_message.content)
        self.assertEqual(state["messages"][-1].content.strip(), "Sorry.")

    async def test_tool_budget_forces_an_answer(self):
        graph = self.graph([tool_call("get_time", {}), "Final."], max_tool_rounds=1)
        await self.turn(graph, "time?")
        self.assertNotIn(NO_MORE_TOOLS, self.prompt_of(0))
        self.assertIn(NO_MORE_TOOLS, self.prompt_of(1))

    async def test_history_is_kept_per_thread(self):
        graph = self.graph(["First.", "Second.", "Other."])
        await self.turn(graph, "one")
        await self.turn(graph, "two")
        await self.turn(graph, "three", thread="t2")
        humans = [m.content for m in self.llm.seen[1] if isinstance(m, HumanMessage)]
        self.assertEqual(humans[0], "one")  # history is stored without the context block
        self.assertTrue(humans[1].startswith("<context>") and humans[1].endswith("two"))
        humans_other = [m.content for m in self.llm.seen[2] if isinstance(m, HumanMessage)]
        self.assertEqual(len(humans_other), 1)
        self.assertTrue(humans_other[0].endswith("three"))

    async def test_recall_puts_memories_into_the_prompt(self):
        self.ctx.memory.add_fact("Adam is looking for machine learning jobs")
        graph = self.graph(["Sure."])
        await self.turn(graph, "machine learning jobs", lang="en")
        self.assertIn("Adam is looking for machine learning jobs", self.prompt_of(0))
        lanes = [e.lane for e in self.events if isinstance(e, StatusEvent)]
        self.assertEqual(lanes, [Lane.RECALLING, Lane.RECALLING])

    async def test_without_memory_there_is_no_recall(self):
        graph = self.graph(["Ok."], with_memory=False)
        await self.turn(graph, "hello", lang="en")
        self.assertNotIn("Relevant memory", self.prompt_of(0))

    async def test_streaming_yields_agent_tokens(self):
        graph = self.graph(["Hello there friend."])
        tokens = []
        async for chunk, meta in graph.astream(
            {"messages": [HumanMessage("hi")], "lang": "en"},
            self.run_config(),
            stream_mode="messages",
        ):
            if meta.get("langgraph_node") == "agent" and chunk.content:
                tokens.append(chunk.content)
        self.assertEqual("".join(tokens).strip(), "Hello there friend.")
        self.assertGreater(len(tokens), 1)


class TestHumanInTheLoop(GraphTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.target = Path(self.tmp.name) / "Documents" / "summary.md"
        self.target.write_text("old")

    async def run_until_question(self, answer_reply):
        call = tool_call("write_file", {"path": str(self.target), "content": "new"})
        graph = self.graph([call, answer_reply])
        await self.turn(graph, "Save the summary", lang="en")
        state = await graph.aget_state(self.run_config())
        return graph, state

    async def test_overwrite_pauses_for_the_user_and_resumes_on_yes(self):
        graph, state = await self.run_until_question("Done, I replaced it.")
        self.assertEqual(len(state.interrupts), 1)
        self.assertIn("already exists", state.interrupts[0].value["question"])
        self.assertEqual(self.target.read_text(), "old")
        final = await graph.ainvoke(Command(resume="yes, go ahead"), self.run_config())
        self.assertEqual(self.target.read_text(), "new")
        self.assertEqual(final["messages"][-1].content.strip(), "Done, I replaced it.")

    async def test_no_keeps_the_file_and_passes_the_words_to_the_model(self):
        graph, _ = await self.run_until_question("Okay, saved as summary2 instead.")
        final = await graph.ainvoke(Command(resume="no, call it summary2"), self.run_config())
        self.assertEqual(self.target.read_text(), "old")
        tool_result = next(m for m in final["messages"] if isinstance(m, ToolMessage))
        self.assertIn("no, call it summary2", tool_result.content)


class TestToolErrors(GraphTestCase):
    async def test_failed_tool_closes_its_lane(self):
        graph = self.graph([tool_call("read_file", {"path": "/etc/passwd"}), "Not allowed."])
        await self.turn(graph, "read it", lang="en")
        ends = [e for e in self.events if isinstance(e, ToolEvent) and e.phase == "end"]
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0].summary.startswith("failed"))


class TestHelpers(unittest.TestCase):
    def test_recent_history_starts_at_user_message(self):
        messages = [
            HumanMessage("a"),
            AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "1", "type": "tool_call"}]),
            ToolMessage("r", tool_call_id="1"),
            AIMessage("b"),
            HumanMessage("c"),
            AIMessage("d"),
        ]
        trimmed = recent_history(messages, limit=4)
        self.assertIsInstance(trimmed[0], HumanMessage)
        self.assertEqual([m.content for m in trimmed], ["c", "d"])

    def test_system_prompt_is_static(self):
        prompt = system_prompt(Config())
        self.assertIn("You are Ava", prompt)
        self.assertEqual(prompt, system_prompt(Config()))
        self.assertNotIn(str(NOW.year), prompt)

    def test_turn_context(self):
        context = turn_context("en", NOW, recall="- fact")
        self.assertIn("Reply language: English", context)
        self.assertIn("Relevant memory", context)
        self.assertNotIn("Relevant memory", turn_context("hu", NOW))

    def test_context_goes_on_newest_user_message_only(self):
        messages = [HumanMessage("a", id="1"), AIMessage("b"), HumanMessage("c", id="2")]
        out = with_context(messages, "<context>x</context>")
        self.assertEqual(out[0].content, "a")
        self.assertEqual(out[2].content, "<context>x</context>\n\nc")
        self.assertEqual(out[2].id, "2")
        self.assertEqual(messages[2].content, "c")


if __name__ == "__main__":
    unittest.main()
