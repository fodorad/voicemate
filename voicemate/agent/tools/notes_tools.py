"""Notes, todos and reminders tools."""

from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from voicemate.agent.tools.base import ToolContext, ToolError, emitter, report
from voicemate.events import NotesChangedEvent
from voicemate.memory.notes import Note


def format_notes(notes: list[Note]) -> str:
    """One line per note."""
    if not notes:
        return "No matching notes."
    return "\n".join(
        f"#{n.id} [{n.kind}{', done' if n.done else ''}] {n.text}"
        + (f" (due {n.due})" if n.due else "")
        for n in notes
    )


def build(ctx: ToolContext) -> list[BaseTool]:
    """Create ``add_note``, ``list_notes`` and ``complete_note``."""

    async def add_note(
        text: str, config: RunnableConfig, kind: str = "note", due: str | None = None
    ) -> str:
        """Save a note, todo item or reminder.

        Args:
            text: What to remember, in the user's words.
            kind: "note", "todo" or "reminder".
            due: Local due date/time in ISO format (e.g. "2026-10-02T09:00") for reminders
                and todos; compute it from today's date in the system prompt.
        """
        try:
            note = await asyncio.to_thread(ctx.notes.add, text, kind, due)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        report(config, "add_note", {"text": text, "kind": kind, "due": due}, "local")
        emitter(config)(NotesChangedEvent())
        return f"Saved {kind} #{note.id}" + (f" due {note.due}" if note.due else "") + "."

    async def list_notes(
        config: RunnableConfig, kind: str | None = None, query: str | None = None
    ) -> str:
        """List open notes, todos or reminders.

        Args:
            kind: Optional filter: "note", "todo" or "reminder".
            query: Optional text to search for.
        """
        if query:
            notes = await asyncio.to_thread(ctx.notes.search, query)
            notes = [n for n in notes if kind is None or n.kind == kind]
        else:
            notes = await asyncio.to_thread(ctx.notes.entries, kind)
        report(config, "list_notes", {"kind": kind, "query": query}, "local")
        return format_notes(notes)

    async def complete_note(note_id: int, config: RunnableConfig) -> str:
        """Mark a todo or reminder as done.

        Args:
            note_id: The number shown as #id by list_notes.
        """
        done = await asyncio.to_thread(ctx.notes.complete, int(note_id))
        if not done:
            raise ToolError(f"No note #{note_id}")
        report(config, "complete_note", {"note_id": note_id}, "local")
        emitter(config)(NotesChangedEvent())
        return f"Marked #{note_id} as done."

    return [
        StructuredTool.from_function(coroutine=add_note, name="add_note", parse_docstring=True),
        StructuredTool.from_function(coroutine=list_notes, name="list_notes", parse_docstring=True),
        StructuredTool.from_function(
            coroutine=complete_note, name="complete_note", parse_docstring=True
        ),
    ]
