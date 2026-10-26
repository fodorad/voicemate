import unittest

from tests.helpers import ScriptedSkill, imported_modules, module_source
from voicemate.pipeline.skills import SkillAnswer, first_answer


class TestFirstAnswer(unittest.TestCase):
    def test_skills_are_tried_in_order_and_the_first_answer_wins(self):
        first, second = (
            ScriptedSkill({"a": "from first"}),
            ScriptedSkill({"a": "x", "b": "from second"}),
        )
        self.assertEqual(first_answer([first, second], "a"), SkillAnswer("from first", "en"))
        self.assertEqual(first_answer([first, second], "b"), SkillAnswer("from second", "en"))
        self.assertEqual(second.asked, ["b"])  # the second skill was not asked about "a"

    def test_no_skill_means_no_answer(self):
        self.assertIsNone(first_answer([], "a"))
        self.assertIsNone(first_answer([ScriptedSkill({})], "a"))


class TestSeams(unittest.TestCase):
    def test_the_session_knows_the_skill_protocol_but_no_skill_internals(self):
        imports = imported_modules(module_source("pipeline/orchestrator.py"))
        self.assertIn("voicemate.pipeline.skills", imports)
        self.assertFalse([m for m in imports if m.startswith("voicemate.hots")])

    def test_the_hots_skill_needs_no_language_model_stack(self):
        for name in ("advisor", "intents", "knowledge", "skill", "draft", "roster", "pages"):
            modules = {m.split(".")[0] for m in imported_modules(module_source(f"hots/{name}.py"))}
            with self.subTest(module=name):
                self.assertTrue(modules.isdisjoint({"langchain", "langchain_core", "langgraph"}))


if __name__ == "__main__":
    unittest.main()
