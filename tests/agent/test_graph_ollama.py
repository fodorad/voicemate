import os
import tempfile
import unittest

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.helpers import make_tool_context


@unittest.skipUnless(os.environ.get("VOICEMATE_INTEGRATION"), "needs Ollama and the LLM")
class TestRealModel(unittest.IsolatedAsyncioTestCase):  # pragma: no cover - integration only
    async def test_hungarian_math_question_uses_calculator(self):
        from voicemate.agent.graph import build_graph, make_chat_model
        from voicemate.agent.tools import build_tools
        from voicemate.config import load_config

        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_tool_context(tmp)
            ctx.config.llm = load_config().llm
            graph = build_graph(
                make_chat_model(ctx.config.llm),
                build_tools(ctx),
                ctx.config,
                ctx.memory,
                InMemorySaver(),
            )
            state = await graph.ainvoke(
                {"messages": [HumanMessage("Mennyi 123456 szorozva 789-cel?")], "lang": "hu"},
                {"configurable": {"thread_id": "it"}},
            )
            await ctx.http.aclose()
            ctx.notes.close()
        tools = [m.name for m in state["messages"] if isinstance(m, ToolMessage)]
        self.assertIn("calculator", tools)
        answer = state["messages"][-1].content
        digits = "".join(ch for ch in answer if ch.isdigit())
        self.assertIn("97406784", digits)


if __name__ == "__main__":
    unittest.main()
