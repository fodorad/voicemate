"""The pipeline rail: what each lane shows, as plain data.

The page draws one station per pipeline lane. Which state a station is in, how long the stage
took and which tool is running follow a few rules (a new turn resets the rail, a finished lane
shows its duration, and so on). :class:`Rail` holds those rules with no UI framework, so they
are tested without a browser; the page only renders the :class:`LaneView` it returns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from voicemate.events import Lane, LaneState, StatusEvent


@dataclass(frozen=True)
class LaneView:
    """What one station shows.

    Attributes:
        state: Idle, active or done.
        duration: Text such as ``"250 ms"`` once a started lane is done; empty otherwise.
        detail: Extra text under the lane, the running tool's name.
    """

    state: LaneState = LaneState.IDLE
    duration: str = ""
    detail: str = ""


class Rail:
    """Lane events in, per-lane views out.

    Args:
        lanes: The lanes the page draws; events for other lanes are ignored.
    """

    def __init__(self, lanes: Iterable[Lane]) -> None:
        self._views: dict[Lane, LaneView] = {lane: LaneView() for lane in lanes}
        self._started: dict[Lane, float] = {}

    def view(self, lane: Lane) -> LaneView:
        """What ``lane`` shows now."""
        return self._views[lane]

    def apply(self, event: StatusEvent) -> dict[Lane, LaneView]:
        """Apply one event.

        Returns:
            The views that changed: the event's lane, or every lane when a new turn starts
            listening (the rail resets first).
        """
        if event.lane not in self._views:
            return {}
        changed: dict[Lane, LaneView] = {}
        if event.lane is Lane.LISTENING and event.state is LaneState.ACTIVE:
            for lane in self._views:
                self._views[lane] = LaneView()
            self._started.clear()
            changed = dict(self._views)
        view = replace(self._views[event.lane], state=event.state)
        if event.state is LaneState.IDLE:
            view = LaneView()
        elif event.state is LaneState.ACTIVE:
            self._started.setdefault(event.lane, event.ts)
            view = replace(view, duration="")
        elif event.lane in self._started:
            view = replace(view, duration=f"{(event.ts - self._started[event.lane]) * 1000:.0f} ms")
        if event.lane is Lane.TOOL and event.detail:
            view = replace(view, detail=event.detail)
        self._views[event.lane] = view
        changed[event.lane] = view
        return changed
