"""Decides when speech counts as a command (pure policy, no I/O).

In ``always`` mode every utterance is a command. In ``wake_word`` mode only an utterance that
starts with the assistant's name ("Ava, what should I pick") is, and the name is stripped; all
other speech is dropped by the caller before anything is shown, logged or stored, so Ava can
stay on while Adam talks to friends. The wake word is a transcript prefix: the speech is
transcribed locally first and discarded when it is not addressed to Ava.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voicemate.config import Config

#: Greetings that may come before the name ("hey Ava").
GREETINGS: tuple[str, ...] = ("hey", "hi", "hello", "ok", "okay")

_MODES = ("always", "wake_word")


@dataclass(frozen=True)
class Activation:
    """The activation rule.

    Attributes:
        mode: ``always`` or ``wake_word``.
        names: Accepted spellings of the assistant's name (lower case).
    """

    mode: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Reject unknown modes."""
        if self.mode not in _MODES:
            raise ValueError(f"Unknown activation mode {self.mode!r}; use one of {_MODES}")

    @classmethod
    def from_config(cls, config: Config) -> Activation:
        """The rule configured in ``[activation]`` for the configured persona name."""
        names = (config.assistant.name, *config.activation.variants)
        return cls(config.activation.mode, tuple(n.strip().lower() for n in names if n.strip()))

    @property
    def needs_wake_word(self) -> bool:
        """Whether speech must start with the name to count."""
        return self.mode == "wake_word"

    def command(self, text: str) -> str | None:
        """The command in ``text``, or ``None`` when it is not addressed to the assistant.

        Args:
            text: A transcript.

        Returns:
            In ``always`` mode the text itself; in ``wake_word`` mode the text after the name
            (``None`` when there is no name or nothing follows it).
        """
        if not self.needs_wake_word:
            return text
        greeting = "|".join(GREETINGS)
        name = "|".join(re.escape(n) for n in self.names)
        match = re.match(
            rf"^\s*(?:(?:{greeting})[\s,.!?]+)?(?:{name})(?![\w'-])[\s,.:;!?-]*(?P<rest>.*)$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        rest = match.group("rest").strip() if match else ""
        return rest or None
