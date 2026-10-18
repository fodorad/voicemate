import unittest
from pathlib import Path

from voicemate.events import Lane, LaneState, StatusEvent
from voicemate.ui.rail import LaneView, Rail

LANES = (Lane.LISTENING, Lane.TRANSCRIBING, Lane.THINKING, Lane.TOOL)


def event(lane: Lane, state: LaneState, ts: float, detail: str = "") -> StatusEvent:
    return StatusEvent(lane, state, detail, ts=ts)


class TestRail(unittest.TestCase):
    def setUp(self):
        self.rail = Rail(LANES)

    def test_every_lane_starts_idle_and_blank(self):
        for lane in LANES:
            self.assertEqual(self.rail.view(lane), LaneView(LaneState.IDLE, "", ""))

    def test_a_finished_lane_shows_how_long_it_took(self):
        self.rail.apply(event(Lane.THINKING, LaneState.ACTIVE, 100.0))
        changed = self.rail.apply(event(Lane.THINKING, LaneState.DONE, 100.25))
        self.assertEqual(changed, {Lane.THINKING: LaneView(LaneState.DONE, "250 ms", "")})
        self.assertEqual(self.rail.view(Lane.THINKING).duration, "250 ms")

    def test_an_active_lane_clears_the_duration_text(self):
        self.rail.apply(event(Lane.THINKING, LaneState.ACTIVE, 1.0))
        self.rail.apply(event(Lane.THINKING, LaneState.DONE, 1.5))
        self.rail.apply(event(Lane.THINKING, LaneState.ACTIVE, 2.0))
        self.assertEqual(self.rail.view(Lane.THINKING), LaneView(LaneState.ACTIVE, "", ""))

    def test_the_first_start_counts_when_a_lane_goes_active_twice(self):
        self.rail.apply(event(Lane.TOOL, LaneState.ACTIVE, 10.0))
        self.rail.apply(event(Lane.TOOL, LaneState.ACTIVE, 10.5))
        self.rail.apply(event(Lane.TOOL, LaneState.DONE, 11.0))
        self.assertEqual(self.rail.view(Lane.TOOL).duration, "1000 ms")

    def test_done_without_a_start_has_no_duration(self):
        self.rail.apply(event(Lane.THINKING, LaneState.DONE, 5.0))
        self.assertEqual(self.rail.view(Lane.THINKING), LaneView(LaneState.DONE, "", ""))

    def test_idle_clears_the_duration_and_the_detail(self):
        self.rail.apply(event(Lane.TOOL, LaneState.ACTIVE, 1.0, "web_search"))
        self.rail.apply(event(Lane.TOOL, LaneState.DONE, 2.0, "web_search"))
        self.rail.apply(event(Lane.TOOL, LaneState.IDLE, 3.0))
        self.assertEqual(self.rail.view(Lane.TOOL), LaneView(LaneState.IDLE, "", ""))

    def test_a_tool_event_with_a_detail_names_the_tool(self):
        self.rail.apply(event(Lane.TOOL, LaneState.ACTIVE, 1.0, "web_search"))
        self.assertEqual(self.rail.view(Lane.TOOL).detail, "web_search")
        self.rail.apply(event(Lane.TOOL, LaneState.DONE, 2.0))  # no detail: keeps the name
        self.assertEqual(self.rail.view(Lane.TOOL).detail, "web_search")

    def test_only_the_tool_lane_takes_a_detail(self):
        self.rail.apply(event(Lane.THINKING, LaneState.ACTIVE, 1.0, "ignored"))
        self.assertEqual(self.rail.view(Lane.THINKING).detail, "")

    def test_a_new_turn_resets_every_lane_and_forgets_start_times(self):
        self.rail.apply(event(Lane.THINKING, LaneState.ACTIVE, 1.0))
        self.rail.apply(event(Lane.THINKING, LaneState.DONE, 2.0))
        self.rail.apply(event(Lane.TOOL, LaneState.ACTIVE, 1.0, "calc"))
        changed = self.rail.apply(event(Lane.LISTENING, LaneState.ACTIVE, 50.0))
        self.assertEqual(set(changed), set(LANES))
        self.assertEqual(changed[Lane.LISTENING].state, LaneState.ACTIVE)
        for lane in (Lane.TRANSCRIBING, Lane.THINKING, Lane.TOOL):
            self.assertEqual(changed[lane], LaneView(LaneState.IDLE, "", ""))
        self.rail.apply(event(Lane.THINKING, LaneState.DONE, 51.0))
        self.assertEqual(self.rail.view(Lane.THINKING).duration, "")  # the old start is gone

    def test_an_ordinary_event_reports_only_its_own_lane(self):
        changed = self.rail.apply(event(Lane.TRANSCRIBING, LaneState.ACTIVE, 1.0))
        self.assertEqual(list(changed), [Lane.TRANSCRIBING])

    def test_unknown_lanes_are_ignored(self):
        self.assertEqual(
            Rail((Lane.LISTENING,)).apply(event(Lane.SPEAKING, LaneState.ACTIVE, 1.0)), {}
        )

    def test_the_rail_needs_no_ui_framework(self):
        source = (Path(__file__).resolve().parents[2] / "voicemate" / "ui" / "rail.py").read_text()
        self.assertNotIn("nicegui", source)


if __name__ == "__main__":
    unittest.main()
