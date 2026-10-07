"""The LangGraph agent: ``recall → agent ⇄ tools``.

* ``recall`` embeds the user message, loads matching facts, past exchanges and research, and
  builds the per-turn context block (about 30-50 ms, so no speculation is needed).
* The system prompt is static and the context is attached only to the newest user message,
  so Ollama can reuse its KV cache for the long, tool-laden prompt prefix (see prompts.py).
* ``agent`` calls the chat model with all tools bound. After ``max_tool_rounds`` tool rounds
  the tools are unbound so the model has to answer.
* ``tools`` runs the requested tools; exceptions become error messages the model can read.

Conversation history is persisted per ``thread_id`` by the checkpointer. Events for the UI
are sent through the ``emit`` callback in ``config["configurable"]``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from voicemate.agent.prompts import system_prompt, turn_context
from voicemate.agent.tools.base import emitter
from voicemate.agent.tools.memory_tools import format_recall
from voicemate.config import Config, LLMConfig
from voicemate.events import Lane, LaneState, StatusEvent, ToolEvent
from voicemate.memory.store import MemoryStore

#: Most recent history messages sent to the model (older ones stay in the checkpoint).
HISTORY_MESSAGES: int = 24

#: Appended to the turn context when the model went silent after using a tool.
ANSWER_NUDGE: str = "You have the tool results now: answer the user's question in full."

#: Appended to the turn context once the tool budget is exhausted.
NO_MORE_TOOLS: str = "Tool budget for this turn is used up: answer now with what you have."


class AgentState(TypedDict):
    """Graph state.

    Attributes:
        messages: Conversation history (appended through the ``add_messages`` reducer).
        lang: Reply language of the current turn.
        context: Per-turn context block (time, language, memories) for the newest message.
        tool_rounds: Tool rounds used in the current turn.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    lang: str
    context: str
    tool_rounds: int


def make_chat_model(
    config: LLMConfig, model: str | None = None
) -> BaseChatModel:  # pragma: no cover - needs Ollama
    """The Ollama chat model with thinking disabled (latency matters more for voice).

    Args:
        config: LLM settings.
        model: Model tag; defaults to the preferred profile's model.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        keep_alive=config.keep_alive,
        reasoning=False,
    )


def recent_history(messages: Sequence[AnyMessage], limit: int = HISTORY_MESSAGES) -> list:
    """The last ``limit`` messages, starting at a user message.

    Starting at a user message keeps tool results together with the call that produced them.
    """
    recent = list(messages[-limit:])
    for i, message in enumerate(recent):
        if isinstance(message, HumanMessage):
            return recent[i:]
    return recent


def _is_empty(message: Any) -> bool:
    """True for a reply with neither text nor tool calls."""
    return not getattr(message, "tool_calls", None) and not str(message.content).strip()


def with_context(messages: list[AnyMessage], context: str) -> list[AnyMessage]:
    """Copy of ``messages`` whose newest user message is prefixed with ``context``."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        message = out[i]
        if isinstance(message, HumanMessage):
            out[i] = HumanMessage(f"{context}\n\n{message.content}", id=message.id)
            break
    return out


async def warm_up(llm: BaseChatModel, tools: list[BaseTool], config: Config) -> None:
    """Prefill the static prompt prefix so the first real turn hits Ollama's KV cache."""
    messages = [
        SystemMessage(system_prompt(config)),
        HumanMessage(f"{turn_context('en', datetime.now().astimezone())}\n\nHi"),
    ]
    await llm.bind_tools(tools).ainvoke(messages)


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    config: Config,
    memory: MemoryStore | None = None,
    checkpointer: Any = None,
    now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> Any:
    """Compile the agent graph.

    Args:
        llm: Chat model supporting tool calling.
        tools: Tools to bind.
        config: Application config.
        memory: Long-term memory used by the recall node (skipped when None).
        checkpointer: LangGraph checkpointer for per-thread history.
        now: Clock for the turn context.

    Returns:
        The compiled graph.
    """
    settings = config  # node functions receive the LangGraph run config as ``config``
    with_tools = llm.bind_tools(tools)
    max_rounds = settings.llm.max_tool_rounds

    async def recall(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        emit = emitter(config)
        user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        text = ""
        if memory is not None and user is not None:
            emit(StatusEvent(Lane.RECALLING, LaneState.ACTIVE))
            found = await asyncio.to_thread(memory.recall, str(user.content))
            text = "" if found.is_empty else format_recall(found)
            emit(StatusEvent(Lane.RECALLING, LaneState.DONE))
        lang = state.get("lang", settings.assistant.default_language)
        context = turn_context(lang, now(), text)
        return {"context": context, "tool_rounds": 0}

    async def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        emit = emitter(config)
        rounds = state.get("tool_rounds", 0)
        context = state.get("context", "")
        model = with_tools
        if rounds >= max_rounds:
            context = f"{context}\n{NO_MORE_TOOLS}"
            model = llm
        history = with_context(recent_history(state["messages"]), context)
        messages = [SystemMessage(system_prompt(settings)), *history]
        response = await model.ainvoke(messages, config)
        if rounds and _is_empty(response):
            # Small local models sometimes stop without answering after a tool round
            # (observed with gemma4:e4b); ask once more instead of leaving the user hanging.
            nudged = with_context(recent_history(state["messages"]), f"{context}\n{ANSWER_NUDGE}")
            response = await model.ainvoke(
                [SystemMessage(system_prompt(settings)), *nudged], config
            )
        calls = getattr(response, "tool_calls", None) or []
        for call in calls:
            emit(ToolEvent(call["name"], "start", dict(call.get("args", {}))))
        return {"messages": [response], "tool_rounds": rounds + (1 if calls else 0)}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    # ty cannot yet match a TypedDict class against LangGraph's TypedDictLike protocol bound.
    graph = StateGraph(AgentState)  # ty: ignore[invalid-argument-type]
    graph.add_node("recall", recall)
    graph.add_node("agent", agent)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        result = await tool_node.ainvoke(state, config)
        # Successful tools report their own result; failed ones are closed here so the UI's
        # tool lane and activity trace never hang on an error.
        emit = emitter(config)
        for message in result.get("messages", []):
            if isinstance(message, ToolMessage) and message.status == "error":
                summary = f"failed: {str(message.content)[:80]}"
                emit(ToolEvent(message.name or "tool", "end", summary=summary))
        return result

    graph.add_node("tools", run_tools)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
