"""A global hold-to-talk key: hold a key combination anywhere on the computer to speak.

Everything that does not touch the operating system lives here: reading a combination such as
``cmd+alt+a`` and turning raw key events into "the hold started" and "the hold ended"
(:class:`HoldDetector`). Listening to the keyboard is an adapter behind
:class:`HotkeyListener`; the macOS one is in :mod:`voicemate.hotkey_macos`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Protocol

#: The modifier keys, by the names a combination uses.
MODIFIERS: frozenset[str] = frozenset({"cmd", "alt", "ctrl", "shift"})
#: Other spellings of the modifiers.
_ALIASES: dict[str, str] = {
    "command": "cmd", "meta": "cmd", "super": "cmd",
    "option": "alt", "opt": "alt",
    "control": "ctrl",
}  # fmt: skip


class HotkeyError(RuntimeError):
    """The hotkey cannot be set up; the message says what to fix."""


@dataclass(frozen=True)
class Combo:
    """A key combination.

    Attributes:
        modifiers: The modifier keys that must be down (``cmd``, ``alt``, ``ctrl``, ``shift``).
        key: The one other key: a letter, a digit or ``space``.
    """

    modifiers: frozenset[str]
    key: str


def parse_combo(text: str) -> Combo:
    """Read a combination such as ``"cmd+alt+a"`` (``option`` and ``command`` work too).

    Raises:
        HotkeyError: If it has no modifier, no key, or an unknown part.
    """
    parts = [_ALIASES.get(part.strip().lower(), part.strip().lower()) for part in text.split("+")]
    modifiers = frozenset(part for part in parts if part in MODIFIERS)
    keys = [part for part in parts if part not in MODIFIERS]
    valid = len(keys) == 1 and (len(keys[0]) == 1 and keys[0].isalnum() or keys[0] == "space")
    if not modifiers or not valid:
        raise HotkeyError(
            f"The hotkey {text!r} must be one or more of cmd, alt, ctrl, shift plus one "
            "letter, digit or 'space' (for example 'cmd+alt+a')."
        )
    return Combo(modifiers, keys[0])


class HoldDetector:
    """Turns key events into the start and the end of a hold of one combination.

    A key held down repeats its key-down event; only the first one starts the hold. The hold
    ends when the key is released or when a modifier of the combination is let go.

    Args:
        combo: The combination to watch.
    """

    def __init__(self, combo: Combo) -> None:
        self.combo = combo
        self.holding = False

    def key(self, key: str, down: bool, modifiers: Collection[str]) -> bool | None:
        """A key went down or up while ``modifiers`` were held.

        Returns:
            ``True`` when the hold starts, ``False`` when it ends, otherwise ``None``.
        """
        if key != self.combo.key:
            return None
        if down:
            if self.holding or set(modifiers) != self.combo.modifiers:
                return None
            self.holding = True
            return True
        if self.holding:
            self.holding = False
            return False
        return None

    def modifiers_changed(self, modifiers: Collection[str]) -> bool | None:
        """The held modifier keys changed; ``False`` when that ends the hold."""
        if self.holding and not self.combo.modifiers <= set(modifiers):
            self.holding = False
            return False
        return None


class HotkeyListener(Protocol):
    """Listens to the keyboard for one combination, wherever the focus is."""

    def start(self, on_hold: Callable[[bool], None]) -> None:
        """Start listening; ``on_hold(True)`` when the hold starts, ``on_hold(False)`` when it ends.

        ``on_hold`` is called from the listener's own thread.

        Raises:
            HotkeyError: If the keyboard cannot be listened to (a missing permission).
        """
        ...

    def stop(self) -> None:
        """Stop listening."""
        ...
