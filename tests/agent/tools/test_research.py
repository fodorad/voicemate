import tempfile
import unittest
from pathlib import Path

from tests.helpers import HashingEmbedder, StaticArxivBackend, StaticSearchBackend
from voicemate.agent.tools.base import ToolError
from voicemate.agent.tools.research import Research
from voicemate.config import MemoryConfig
from voicemate.memory.store import MemoryStore


class FailingBackend:
    name = "broken"

    async def search(self, query: str, max_results: int):
        raise ConnectionError("rate limited")


class ResearchTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(
            Path(self.tmp.name),
            HashingEmbedder(),
            MemoryConfig(search_cache_similarity=0.9, recall_min_similarity=0.2),
        )
        self.web = StaticSearchBackend()
        self.arxiv = StaticArxivBackend()
        self.research = Research(self.memory, self.web, self.arxiv, max_results=5)

    def tearDown(self):
        self.tmp.cleanup()


class TestWebSearch(ResearchTestCase):
    async def test_a_new_query_goes_to_the_backend(self):
        found = await self.research.web("mixture of experts survey")
        self.assertFalse(found.from_memory)
        self.assertEqual(found.backend, "static")
        self.assertEqual([r["url"] for r in found.results][:1], ["https://example.org/moe"])
        self.assertEqual(self.web.queries, ["mixture of experts survey"])

    async def test_the_same_query_is_then_answered_from_memory(self):
        await self.research.web("mixture of experts survey")
        again = await self.research.web("mixture of experts survey")
        self.assertTrue(again.from_memory)
        self.assertEqual(again.query, "mixture of experts survey")
        self.assertGreater(again.retrieved_at, 0)
        self.assertEqual(self.web.queries, ["mixture of experts survey"])  # no second request

    async def test_results_become_recallable_documents(self):
        await self.research.web("mixture of experts survey")
        hits = self.memory.search_documents("mixture of experts routes tokens")
        self.assertIn("https://example.org/moe", [h.meta.get("url") for h in hits])

    async def test_max_results_limits_the_backend(self):
        research = Research(self.memory, self.web, self.arxiv, max_results=1)
        self.assertEqual(len((await research.web("anything at all")).results), 1)

    async def test_backend_failures_become_tool_errors(self):
        research = Research(self.memory, FailingBackend(), self.arxiv, max_results=5)
        with self.assertRaisesRegex(ToolError, "Web search failed: rate limited"):
            await research.web("anything")


class TestArxivSearch(ResearchTestCase):
    async def test_a_new_query_goes_to_arxiv_then_comes_from_memory(self):
        first = await self.research.papers("sparse experts routing", max_results=2)
        self.assertFalse(first.from_memory)
        self.assertEqual([r["arxiv_id"] for r in first.results], ["2401.00001", "2401.00002"])
        again = await self.research.papers("sparse experts routing", max_results=2)
        self.assertTrue(again.from_memory)
        self.assertEqual(self.arxiv.queries, ["sparse experts routing"])

    async def test_the_count_is_clamped_to_1_to_10(self):
        self.assertEqual(len((await self.research.papers("one", max_results=0)).results), 1)

    async def test_web_and_arxiv_caches_do_not_answer_each_other(self):
        await self.research.web("sparse experts routing")
        found = await self.research.papers("sparse experts routing", max_results=2)
        self.assertFalse(found.from_memory)

    async def test_failures_become_tool_errors(self):
        class Broken:
            name = "broken"

            async def search(self, query, max_results):
                raise TimeoutError("slow")

        research = Research(self.memory, self.web, Broken(), max_results=5)
        with self.assertRaisesRegex(ToolError, "arXiv search failed"):
            await research.papers("x", max_results=3)


class TestListedAddresses(ResearchTestCase):
    async def test_nothing_is_listed_before_a_search(self):
        self.assertFalse(self.research.is_listed("https://example.org/moe"))

    async def test_web_results_are_listed_fresh_and_from_memory(self):
        await self.research.web("mixture of experts survey")
        self.assertTrue(self.research.is_listed("https://example.org/moe"))
        other = Research(self.memory, self.web, self.arxiv, max_results=5)
        self.assertFalse(other.is_listed("https://example.org/moe"))  # per instance
        await other.web("mixture of experts survey")  # now answered from memory
        self.assertTrue(other.is_listed("https://example.org/moe"))

    async def test_arxiv_results_list_their_abstract_pages(self):
        await self.research.papers("sparse experts routing", max_results=2)
        self.assertTrue(self.research.is_listed("https://arxiv.org/abs/2401.00001"))
        other = Research(self.memory, self.web, self.arxiv, max_results=5)
        await other.papers("sparse experts routing", max_results=2)
        self.assertTrue(other.is_listed("https://arxiv.org/abs/2401.00002"))

    async def test_a_user_approved_address_is_listed_from_then_on(self):
        self.research.list_address("https://example.com/page")
        self.assertTrue(self.research.is_listed("https://example.com/page"))
        self.assertFalse(self.research.is_listed("https://example.com/page?d=secret"))


if __name__ == "__main__":
    unittest.main()
