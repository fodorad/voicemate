import unittest

from tests.helpers import (
    attributes_of,
    called_names,
    imported_modules,
    module_source,
    uses_async_syntax,
)

SNIPPET = '''
"""A docstring that mentions langgraph and open( and nicegui."""
# a comment about import asyncio
import os.path
from collections import deque
from langgraph.types import Command

def f(memory):
    def inner():
        from nicegui import ui
    return memory.facts.add("x") + self.memory.research.cached_search("y")
'''


class TestSourceReaders(unittest.TestCase):
    def test_imports_come_from_the_syntax_tree_not_from_text(self):
        names = imported_modules(SNIPPET)
        self.assertEqual(names, {"os.path", "collections", "langgraph.types", "nicegui"})
        self.assertNotIn("asyncio", names)  # only mentioned in a comment

    def test_async_syntax_is_detected_and_text_does_not_fool_it(self):
        self.assertFalse(uses_async_syntax("x = 'await async def'  # await"))
        for code in (
            "async def f(): pass",
            "async def f():\n    await g()",
            "async def f():\n    async for x in y: pass",
            "async def f():\n    async with a: pass",
        ):
            with self.subTest(code=code):
                self.assertTrue(uses_async_syntax(code))

    def test_called_names_include_functions_and_methods(self):
        self.assertEqual(called_names(SNIPPET), {"add", "cached_search"})
        self.assertIn("open", called_names("open('f')"))
        self.assertNotIn("open", called_names("'open('  # open("))

    def test_attributes_of_an_owner_works_for_names_and_chains(self):
        self.assertEqual(attributes_of(SNIPPET, "memory"), {"facts", "research"})
        self.assertEqual(attributes_of("a.b.c.d", "b"), {"c"})
        self.assertEqual(attributes_of("x = 1", "memory"), set())

    def test_module_source_reads_a_package_file(self):
        self.assertIn("class Rail", module_source("ui/rail.py"))


if __name__ == "__main__":
    unittest.main()
