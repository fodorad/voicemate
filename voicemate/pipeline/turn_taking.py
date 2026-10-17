"""Turn-taking rules: what each thing the user does means for the conversation.

:class:`TurnTaking` is a small state machine with no event loop and no I/O. The voice session
tells it what happened (an utterance, typed text, a button, a cancelled turn, ...) as plain
values and executes the :data:`Decision` it returns. That keeps the rules (merging a
cut-off question, answering a confirmation, holding until the user is finished, barge-in)
testable without a graph, a microphone or a voice.
"""

from __future__ import annotations

from dataclasses import dataclass

#: A question cut off before any answer is merged with the next utterance only if that
#: utterance ends within this many seconds; later speech is a new question.
MERGE_WINDOW_S: float = 20.0


@dataclass(frozen=True)
class Begin:
    """Start a turn.

    Attributes:
        text: The user's words (a merged question already joined).
        merged: ``text`` continues a question that was cut off.
        resume: The user's words answer a pending confirmation (the turn resumes the run).
        discard: Id of the cut-off message to remove from the history (when merged).
        shown: The UI already shows ``text`` (it was collected during a hold).
    """

    text: str
    merged: bool = False
    resume: str | None = None
    discard: str | None = None
    shown: bool = False


@dataclass(frozen=True)
class Collect:
    """Keep the words; the user is holding.

    Attributes:
        text: Everything collected so far, joined.
        parts: How many separate inputs it holds.
    """

    text: str
    parts: int


@dataclass(frozen=True)
class Interrupt:
    """Stop what the assistant is doing (barge-in, or the user took the floor).

    Returned for input while a turn is running too: nothing was consumed, so the caller
    cancels the turn and presents the same input again.
    """


@dataclass(frozen=True)
class Ignore:
    """Nothing to do."""


Decision = Begin | Collect | Interrupt | Ignore


@dataclass
class _CutOff:
    """A user message whose turn was cancelled before anything was said."""

    message_id: str
    text: str
    at: float


class TurnTaking:
    """Decides what user input means. Times are monotonic seconds supplied by the caller.

    Args:
        merge_window_s: See :data:`MERGE_WINDOW_S`.

    Attributes:
        holding: While True, input is collected instead of answered.
        question: The confirmation the agent is waiting on, if any.
    """

    def __init__(self, merge_window_s: float = MERGE_WINDOW_S) -> None:
        self.merge_window_s = merge_window_s
        self.holding = False
        self.question: str | None = None
        self._held: list[str] = []
        self._cut_off: _CutOff | None = None

    # ── input ─────────────────────────────────────────────────────────────────

    def on_utterance(self, text: str, ended_at: float, turn_running: bool) -> Decision:
        """The user said ``text``; it may continue a question that was just cut off."""
        if self.holding:
            return self._collect(text)
        return self._begin(text, ended_at, turn_running, may_merge=True)

    def on_typed(self, text: str, now: float, turn_running: bool) -> Decision:
        """The user typed ``text`` (typed input never merges)."""
        if self.holding:
            return self._collect(text)
        return self._begin(text, now, turn_running, may_merge=False)

    def on_answer(self, text: str, now: float, turn_running: bool) -> Decision:
        """The user answered the pending question with a button (works while holding)."""
        if self.question is None:
            return Ignore()
        return self._begin(text, now, turn_running, may_merge=False)

    def on_speech_start(self, busy: bool) -> Decision:
        """The user started speaking: barge in if the assistant is busy."""
        return Interrupt() if busy else Ignore()

    # ── hold ──────────────────────────────────────────────────────────────────

    def hold(self, busy: bool) -> Decision:
        """The user wants time to think: collect input; a busy assistant must stop."""
        self.holding = True
        return Interrupt() if busy else Ignore()

    def release(self, turn_running: bool = False) -> Decision:
        """The user is finished: everything collected becomes one turn."""
        if not self.holding:
            return Ignore()
        if self._held and turn_running:
            return Interrupt()
        self.holding = False
        held, self._held = self._held, []
        if not held:
            return Ignore()
        return self._begin(" ".join(held), 0.0, False, may_merge=False, shown=True)

    # ── events from the session ───────────────────────────────────────────────

    def question_asked(self, question: str) -> None:
        """The agent paused and waits for the user's answer."""
        self.question = question

    def turn_cancelled(
        self, message_id: str, text: str, at: float, spoke: bool, resumed: bool
    ) -> None:
        """A turn was cancelled; remember its question if nothing was said and it was new."""
        if not spoke and not resumed:
            self._cut_off = _CutOff(message_id, text, at)

    # ── internals ─────────────────────────────────────────────────────────────

    def _collect(self, text: str) -> Collect:
        self._held.append(text)
        return Collect(" ".join(self._held), len(self._held))

    def _begin(
        self,
        text: str,
        at: float,
        turn_running: bool,
        may_merge: bool,
        shown: bool = False,
    ) -> Begin | Interrupt:
        if turn_running:  # the running turn must stop (and may leave a cut-off question)
            return Interrupt()
        cut_off, self._cut_off = self._cut_off, None
        if self.question is not None:  # the user's words answer the pending question
            self.question = None
            return Begin(text, resume=text, shown=shown)
        if may_merge and cut_off is not None and at - cut_off.at <= self.merge_window_s:
            return Begin(
                f"{cut_off.text} {text}", merged=True, discard=cut_off.message_id, shown=shown
            )
        return Begin(text, shown=shown)
