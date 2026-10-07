"""Human-in-the-loop confirmation for risky tool actions.

A tool calls :func:`ask_user`, which pauses the LangGraph run with
:func:`langgraph.types.interrupt`. The voice session notices the pending interrupt, shows the
question in the UI and speaks it, and resumes the graph with whatever the user answers (a
button click or speech). The tool receives the user's own words, so a reply like "no, save
it as summary2.md" reaches the model instead of a bare yes/no.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langgraph.types import interrupt

#: Words that mean yes (English and Hungarian).
YES_WORDS: frozenset[str] = frozenset(
    "yes yeah yep sure ok okay go ahead do it allow confirm please igen persze mehet rendben "
    "jó oké csináld".split()
)

#: Words that mean no; any of them anywhere in the answer wins over a yes word.
NO_WORDS: frozenset[str] = frozenset(
    "no nope don't dont not never cancel stop deny wait nem ne mégse állj".split()
)

_WORD = re.compile(r"[^\W\d_]+(?:'[a-z]+)?", re.UNICODE)


@dataclass
class Confirmation:
    """A question the agent is waiting on.

    Attributes:
        question: What the user is asked (spoken and shown in the UI).
    """

    question: str


def is_affirmative(answer: str) -> bool:
    """True when ``answer`` clearly agrees (a yes word and no negation)."""
    words = [w.lower() for w in _WORD.findall(answer)]
    if any(w in NO_WORDS for w in words):
        return False
    return any(w in YES_WORDS for w in words)


def ask_user(question: str) -> str:
    """Pause the graph until the user answers ``question``; returns their answer text.

    Must be called inside a LangGraph run with a checkpointer (e.g. from a tool).
    """
    answer = interrupt({"question": question})
    return str(answer)
