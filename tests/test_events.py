import unittest

from voicemate.events import (
    EventBus,
    Lane,
    LaneState,
    StatusEvent,
    TokenEvent,
    TranscriptEvent,
)


class TestEventBus(unittest.IsolatedAsyncioTestCase):
    async def test_every_subscriber_receives_events_in_order(self):
        bus = EventBus()
        first, second = bus.subscribe(), bus.subscribe()
        events = [
            StatusEvent(Lane.LISTENING, LaneState.ACTIVE),
            TranscriptEvent("szia", final=True, lang="hu"),
            TokenEvent("Hello"),
        ]
        for event in events:
            bus.publish(event)
        for queue in (first, second):
            received = [queue.get_nowait() for _ in events]
            self.assertEqual(received, events)

    async def test_unsubscribed_queue_receives_nothing(self):
        bus = EventBus()
        queue = bus.subscribe()
        bus.unsubscribe(queue)
        bus.publish(TokenEvent("x"))
        self.assertTrue(queue.empty())

    async def test_full_queue_drops_oldest_instead_of_blocking(self):
        bus = EventBus(maxsize=2)
        queue = bus.subscribe()
        for text in ("a", "b", "c"):
            bus.publish(TokenEvent(text))
        self.assertEqual([queue.get_nowait().text for _ in range(2)], ["b", "c"])


class TestStatusEvent(unittest.TestCase):
    def test_timestamp_is_set_automatically(self):
        event = StatusEvent(Lane.TOOL, LaneState.ACTIVE, detail="web_search")
        self.assertGreater(event.ts, 0)
        self.assertEqual(event.detail, "web_search")


if __name__ == "__main__":
    unittest.main()
