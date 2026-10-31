"""The macOS hotkey adapter: a listen-only keyboard event tap on a background thread.

Needs *Input Monitoring* for the app that runs Ava (System Settings → Privacy & Security).
The tap only listens, so the keys still reach the focused app. Key codes are the physical keys
of an ANSI keyboard.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from voicemate.hotkey import Combo, HoldDetector, HotkeyError

logger = logging.getLogger(__name__)

PERMISSION_HINT = (
    "Allow Input Monitoring for the app that runs Ava (System Settings → Privacy & Security → "
    "Input Monitoring), then restart it."
)

#: Key codes of the keys a combination may use.
KEY_CODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
    "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46, "space": 49,
}  # fmt: skip
_KEYS = {code: key for key, code in KEY_CODES.items()}
#: How long :meth:`MacHotkeyListener.start` waits for the tap to come up, seconds.
START_TIMEOUT_S = 3.0


class MacHotkeyListener:
    """Listens for one combination with a Quartz event tap.

    Args:
        combo: The combination to hold.
    """

    def __init__(self, combo: Combo) -> None:
        if combo.key not in KEY_CODES:
            raise HotkeyError(f"The key {combo.key!r} cannot be used as a hotkey here.")
        self.combo = combo
        self._loop = None
        self._thread: threading.Thread | None = None

    def start(self, on_hold: Callable[[bool], None]) -> None:
        """Start the tap thread and wait until it is listening."""
        ready = threading.Event()
        failure: list[str] = []
        self._thread = threading.Thread(
            target=self._run, args=(on_hold, ready, failure), name="hotkey", daemon=True
        )
        self._thread.start()
        if not ready.wait(START_TIMEOUT_S) or failure:
            raise HotkeyError(
                failure[0] if failure else f"The hotkey did not start. {PERMISSION_HINT}"
            )

    def stop(self) -> None:
        """Stop the tap thread."""
        import Quartz

        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(
        self, on_hold: Callable[[bool], None], ready: threading.Event, failure: list[str]
    ) -> None:
        import Quartz

        detector = HoldDetector(self.combo)
        flag_masks = {
            "cmd": Quartz.kCGEventFlagMaskCommand,
            "alt": Quartz.kCGEventFlagMaskAlternate,
            "ctrl": Quartz.kCGEventFlagMaskControl,
            "shift": Quartz.kCGEventFlagMaskShift,
        }
        tap = None

        def callback(_proxy, kind, event, _refcon):
            if kind in (
                Quartz.kCGEventTapDisabledByTimeout,
                Quartz.kCGEventTapDisabledByUserInput,
            ):
                Quartz.CGEventTapEnable(tap, True)  # the system switched it off: turn it on again
                return event
            flags = Quartz.CGEventGetFlags(event)
            held = {name for name, mask in flag_masks.items() if flags & mask}
            if kind == Quartz.kCGEventFlagsChanged:
                changed = detector.modifiers_changed(held)
            else:
                code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                key = _KEYS.get(code)
                changed = detector.key(key, kind == Quartz.kCGEventKeyDown, held) if key else None
            if changed is not None:
                on_hold(changed)
            return event

        mask = sum(
            Quartz.CGEventMaskBit(kind)
            for kind in (
                Quartz.kCGEventKeyDown,
                Quartz.kCGEventKeyUp,
                Quartz.kCGEventFlagsChanged,
            )
        )
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            callback,
            None,
        )
        if tap is None:
            failure.append(f"The keyboard cannot be listened to. {PERMISSION_HINT}")
            ready.set()
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        logger.info("Hotkey %s is listening", self.combo)
        ready.set()
        Quartz.CFRunLoopRun()
