import unittest

from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.clock_calc import safe_eval


class TestSafeEval(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(safe_eval("23*42"), 966)
        self.assertEqual(safe_eval("(3+4)/7"), 1)
        self.assertEqual(safe_eval("2^10"), 1024)
        self.assertAlmostEqual(safe_eval("sqrt(2)**2"), 2.0)

    def test_percent_and_modulo(self):
        self.assertAlmostEqual(safe_eval("17% * 340"), 57.8)
        self.assertEqual(safe_eval("10 % 3"), 1)

    def test_rejects_code_and_huge_exponents(self):
        for expression in ("__import__('os')", "open('x')", "(1).real", "9**9**9", "1/0", "2 +"):
            with self.subTest(expression=expression), self.assertRaises(ToolError):
                safe_eval(expression)


if __name__ == "__main__":
    unittest.main()
