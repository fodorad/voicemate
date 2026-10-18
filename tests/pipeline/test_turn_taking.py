import unittest

from tests.helpers import called_names, imported_modules, module_source, uses_async_syntax
from voicemate.pipeline.turn_taking import (
    MERGE_WINDOW_S,
    Begin,
    Collect,
    Ignore,
    Interrupt,
    TurnTaking,
)


def cut_off(policy: TurnTaking, text="tell me a story", at=100.0, message_id="m1", **kwargs):
    """The previous turn was cancelled before anything was said."""
    policy.turn_cancelled(message_id, text, at, spoke=False, resumed=False, **kwargs)


class TestStartingTurns(unittest.TestCase):
    def setUp(self):
        self.policy = TurnTaking()

    def test_plain_utterance_starts_a_turn(self):
        self.assertEqual(
            self.policy.on_utterance("hello", ended_at=1.0, turn_running=False), Begin("hello")
        )

    def test_a_running_turn_must_be_interrupted_first_and_nothing_is_consumed(self):
        cut_off(self.policy)
        self.assertEqual(
            self.policy.on_utterance("but short", ended_at=101.0, turn_running=True), Interrupt()
        )
        # The session cancels the turn, then presents the same input again.
        self.assertEqual(
            self.policy.on_utterance("but short", ended_at=101.0, turn_running=False),
            Begin("tell me a story but short", merged=True, discard="m1"),
        )

    def test_a_running_turn_blocks_typed_text_and_answers_too(self):
        self.assertEqual(self.policy.on_typed("x", now=1.0, turn_running=True), Interrupt())
        self.policy.question_asked("Replace it?")
        self.assertEqual(self.policy.on_answer("yes", now=1.0, turn_running=True), Interrupt())
        self.assertEqual(self.policy.question, "Replace it?")

    def test_typed_text_starts_a_turn_too(self):
        self.assertEqual(self.policy.on_typed("hello", now=1.0, turn_running=False), Begin("hello"))


class TestMergingACutOffQuestion(unittest.TestCase):
    def setUp(self):
        self.policy = TurnTaking()

    def test_the_next_utterance_within_the_window_continues_the_question(self):
        cut_off(self.policy)
        decision = self.policy.on_utterance("but short", ended_at=105.0, turn_running=False)
        self.assertEqual(decision, Begin("tell me a story but short", merged=True, discard="m1"))

    def test_the_window_boundary(self):
        cut_off(self.policy, at=100.0)
        inside = self.policy.on_utterance("x", ended_at=100.0 + MERGE_WINDOW_S, turn_running=False)
        self.assertTrue(inside.merged)
        cut_off(self.policy, at=100.0)
        outside = self.policy.on_utterance("x", ended_at=100.1 + MERGE_WINDOW_S, turn_running=False)
        self.assertEqual(outside, Begin("x"))

    def test_typed_text_never_merges_but_still_consumes_the_cut_off_question(self):
        cut_off(self.policy)
        self.assertEqual(
            self.policy.on_typed("hello", now=101.0, turn_running=False), Begin("hello")
        )
        again = self.policy.on_utterance("later", ended_at=102.0, turn_running=False)
        self.assertEqual(again, Begin("later"))

    def test_a_question_is_merged_only_once(self):
        cut_off(self.policy)
        self.policy.on_utterance("a", ended_at=101.0, turn_running=False)
        self.assertEqual(
            self.policy.on_utterance("b", ended_at=102.0, turn_running=False), Begin("b")
        )

    def test_nothing_is_kept_when_something_was_already_said(self):
        self.policy.turn_cancelled("m1", "q", 100.0, spoke=True, resumed=False)
        self.assertEqual(
            self.policy.on_utterance("x", ended_at=101.0, turn_running=False), Begin("x")
        )

    def test_nothing_is_kept_when_the_cancelled_turn_was_a_resume(self):
        self.policy.turn_cancelled("m1", "yes", 100.0, spoke=False, resumed=True)
        self.assertEqual(
            self.policy.on_utterance("x", ended_at=101.0, turn_running=False), Begin("x")
        )


class TestConfirmation(unittest.TestCase):
    def setUp(self):
        self.policy = TurnTaking()
        self.policy.question_asked("Replace it?")

    def test_the_next_input_answers_the_question(self):
        self.assertEqual(self.policy.question, "Replace it?")
        decision = self.policy.on_utterance("yes go ahead", ended_at=1.0, turn_running=False)
        self.assertEqual(decision, Begin("yes go ahead", resume="yes go ahead"))
        self.assertIsNone(self.policy.question)
        self.assertEqual(
            self.policy.on_utterance("next", ended_at=2.0, turn_running=False), Begin("next")
        )

    def test_an_answer_never_merges_with_a_cut_off_question(self):
        cut_off(self.policy)
        decision = self.policy.on_utterance("no", ended_at=101.0, turn_running=False)
        self.assertEqual(decision, Begin("no", resume="no"))

    def test_a_button_answer_resumes_even_while_holding(self):
        self.policy.hold(busy=False)
        decision = self.policy.on_answer("yes", now=1.0, turn_running=False)
        self.assertEqual(decision, Begin("yes", resume="yes"))

    def test_a_button_answer_without_a_question_is_ignored(self):
        policy = TurnTaking()
        self.assertEqual(policy.on_answer("yes", now=1.0, turn_running=False), Ignore())


class TestHold(unittest.TestCase):
    def setUp(self):
        self.policy = TurnTaking()

    def test_pauses_are_collected_into_one_growing_question(self):
        self.policy.hold(busy=False)
        self.assertTrue(self.policy.holding)
        first = self.policy.on_utterance("First thought.", ended_at=1.0, turn_running=False)
        second = self.policy.on_typed("And a typed one.", now=2.0, turn_running=False)
        self.assertEqual(first, Collect("First thought.", parts=1))
        self.assertEqual(second, Collect("First thought. And a typed one.", parts=2))

    def test_release_sends_everything_as_one_turn(self):
        self.policy.hold(busy=False)
        self.policy.on_utterance("One.", ended_at=1.0, turn_running=False)
        self.policy.on_utterance("Two.", ended_at=2.0, turn_running=False)
        decision = self.policy.release()
        self.assertEqual(decision, Begin("One. Two.", shown=True))
        self.assertFalse(self.policy.holding)

    def test_release_with_a_running_turn_interrupts_it_first(self):
        self.policy.hold(busy=False)
        self.policy.on_utterance("One.", ended_at=1.0, turn_running=False)
        self.assertEqual(self.policy.release(turn_running=True), Interrupt())
        self.assertTrue(self.policy.holding)
        self.assertEqual(self.policy.release(), Begin("One.", shown=True))

    def test_release_answers_a_pending_question_with_the_collected_words(self):
        self.policy.question_asked("Replace it?")
        self.policy.hold(busy=False)
        self.policy.on_typed("yes please", now=1.0, turn_running=False)
        self.assertEqual(
            self.policy.release(), Begin("yes please", resume="yes please", shown=True)
        )

    def test_release_with_nothing_collected_does_nothing(self):
        self.policy.hold(busy=False)
        self.assertEqual(self.policy.release(), Ignore())
        self.assertEqual(self.policy.release(), Ignore())  # releasing twice is harmless
        self.assertFalse(self.policy.holding)

    def test_release_without_a_hold_does_nothing(self):
        self.assertEqual(self.policy.release(), Ignore())

    def test_hold_while_busy_asks_to_cancel_the_running_turn(self):
        self.assertEqual(self.policy.hold(busy=True), Interrupt())
        self.assertTrue(self.policy.holding)

    def test_hold_while_idle_just_holds(self):
        self.assertEqual(self.policy.hold(busy=False), Ignore())

    def test_release_discards_any_cut_off_question_instead_of_merging(self):
        cut_off(self.policy)
        self.policy.hold(busy=False)
        self.policy.on_utterance("fresh", ended_at=101.0, turn_running=False)
        self.assertEqual(self.policy.release(), Begin("fresh", shown=True))
        self.assertEqual(
            self.policy.on_utterance("x", ended_at=102.0, turn_running=False), Begin("x")
        )


class TestBargeIn(unittest.TestCase):
    def test_speech_while_busy_interrupts(self):
        policy = TurnTaking()
        self.assertEqual(policy.on_speech_start(busy=True), Interrupt())
        self.assertEqual(policy.on_speech_start(busy=False), Ignore())


class TestPurity(unittest.TestCase):
    def test_the_module_needs_no_event_loop_or_io(self):
        source = module_source("pipeline/turn_taking.py")
        self.assertEqual(imported_modules(source), {"__future__", "dataclasses"})
        self.assertFalse(uses_async_syntax(source))
        self.assertNotIn("open", called_names(source))


if __name__ == "__main__":
    unittest.main()
