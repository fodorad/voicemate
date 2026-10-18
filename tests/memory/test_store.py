import tempfile
import unittest
from pathlib import Path

from tests.helpers import HashingEmbedder
from voicemate.config import MemoryConfig
from voicemate.memory.store import DAY, MemoryStore


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        config = MemoryConfig(
            search_cache_similarity=0.9,
            fact_min_similarity=0.3,
            recall_min_similarity=0.3,
            fact_dedupe_similarity=0.95,
        )
        self.store = MemoryStore(Path(self.tmp.name), HashingEmbedder(), config, clock=self.clock)

    def tearDown(self):
        self.tmp.cleanup()

    # ── facts ────────────────────────────────────────────────────────────────

    def test_fact_roundtrip(self):
        self.store.facts.add("Adam is looking for machine learning jobs in Budapest")
        hits = self.store.facts.search("machine learning jobs")
        self.assertEqual(len(hits), 1)
        self.assertIn("Budapest", hits[0].text)
        self.assertGreater(hits[0].similarity, 0.3)

    def test_identical_fact_replaces_instead_of_duplicating(self):
        self.store.facts.add("Adam likes green tea")
        replaced = self.store.facts.add("Adam likes green tea")
        self.assertTrue(replaced)
        self.assertEqual(self.store.stats()["facts"], 1)

    def test_unrelated_query_returns_nothing(self):
        self.store.facts.add("Adam likes green tea")
        self.assertEqual(self.store.facts.search("quantum chromodynamics lattice"), [])

    # ── episodes ─────────────────────────────────────────────────────────────

    def test_episode_is_recalled_with_date(self):
        self.store.episodes.add(
            "what is retrieval augmented generation", "RAG combines ...", "en", "t1"
        )
        hits = self.store.episodes.search("retrieval augmented generation")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].meta["thread_id"], "t1")
        self.assertEqual(hits[0].created, self.clock.now)

    # ── research cache ───────────────────────────────────────────────────────

    def test_search_cache_hit_and_expiry(self):
        results = [{"title": "MoE survey", "url": "https://a.b", "snippet": "experts"}]
        self.store.research.store_search("mixture of experts survey", "web", results, ttl_days=30)
        hit = self.store.research.cached_search("mixture of experts survey", "web")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.results, results)
        self.assertEqual(hit.retrieved_at, self.clock.now)
        self.assertIsNone(self.store.research.cached_search("mixture of experts survey", "arxiv"))
        self.clock.now += 31 * DAY
        self.assertIsNone(self.store.research.cached_search("mixture of experts survey", "web"))

    def test_different_query_misses_cache(self):
        self.store.research.store_search("mixture of experts survey", "web", [], ttl_days=30)
        self.assertIsNone(self.store.research.cached_search("best pizza in gyöngyös", "web"))

    def test_documents_by_url_and_semantic_search(self):
        self.store.research.add_documents(
            ["LoRA trains low rank adapters.", "Full fine tuning updates every weight."],
            url="https://x.org/lora",
            title="LoRA",
            source="page",
            ttl_days=30,
        )
        chunks = self.store.research.documents_for_url("https://x.org/lora")
        self.assertEqual(len(chunks), 2)
        hits = self.store.research.search_documents("low rank adapters LoRA")
        self.assertEqual(hits[0].meta["url"], "https://x.org/lora")
        self.clock.now += 40 * DAY
        self.assertEqual(self.store.research.documents_for_url("https://x.org/lora"), [])
        self.assertEqual(self.store.research.search_documents("low rank adapters LoRA"), [])

    def test_document_chunks_come_back_in_order(self):
        chunks = [f"chunk number {i}" for i in range(25)]
        self.store.research.add_documents(chunks, url="u", title="t", source="page", ttl_days=1)
        self.assertEqual([h.text for h in self.store.research.documents_for_url("u")], chunks)

    def test_url_with_quote_is_escaped(self):
        url = "https://x.org/it's"
        self.store.research.add_documents(["quoted"], url=url, title="t", source="page", ttl_days=1)
        self.assertEqual(len(self.store.research.documents_for_url(url)), 1)

    # ── combined recall ──────────────────────────────────────────────────────

    def test_recall_combines_all_kinds(self):
        self.store.facts.add("Adam studies speech recognition")
        self.store.episodes.add("tell me about speech recognition", "Sure ...", "en", "t")
        self.store.research.add_documents(
            ["Parakeet is a speech recognition model."], "u", "t", "page", ttl_days=30
        )
        recall = self.store.recall("speech recognition")
        self.assertEqual(len(recall.facts), 1)
        self.assertEqual(len(recall.episodes), 1)
        self.assertEqual(len(recall.documents), 1)
        self.assertFalse(recall.is_empty)
        self.assertTrue(self.store.recall("zzz qqq").is_empty)

    def test_store_reopens_existing_tables(self):
        self.store.facts.add("persisted fact")
        reopened = MemoryStore(Path(self.tmp.name), HashingEmbedder(), MemoryConfig())
        self.assertEqual(reopened.stats()["facts"], 1)

    def test_every_concept_survives_a_reopen(self):
        self.store.facts.add("Adam studies speech recognition")
        self.store.episodes.add("what is CTC", "A loss for alignment-free training.", "en", "t9")
        self.store.research.store_search("ctc loss survey", "web", [{"url": "u"}], ttl_days=30)
        self.store.research.add_documents(["CTC sums over alignments."], "u", "t", "web", 30)
        reopened = MemoryStore(
            Path(self.tmp.name), HashingEmbedder(), self.store._index.config, clock=self.clock
        )
        self.assertEqual(
            reopened.stats(), {"facts": 1, "episodes": 1, "searches": 1, "documents": 1}
        )
        self.assertTrue(reopened.facts.search("speech recognition"))
        self.assertTrue(reopened.episodes.search("what is CTC"))
        self.assertIsNotNone(reopened.research.cached_search("ctc loss survey", "web"))
        self.assertEqual(len(reopened.research.documents_for_url("u")), 1)


class TestCallersSeeOnlyTheirSlice(unittest.TestCase):
    """Each caller reaches memory through one concept interface (or recall)."""

    ROOT = Path(__file__).resolve().parents[2] / "voicemate"

    def uses(self, relative: str) -> set[str]:
        source = (self.ROOT / relative).read_text()
        return {c for c in ("facts", "episodes", "research") if f"memory.{c}." in source}

    def test_research_touches_only_research_memory(self):
        self.assertEqual(self.uses("agent/tools/research.py"), {"research"})

    def test_the_memory_tools_touch_only_facts(self):
        self.assertEqual(self.uses("agent/tools/memory_tools.py"), {"facts"})

    def test_the_session_touches_only_episodes(self):
        self.assertEqual(self.uses("pipeline/orchestrator.py"), {"episodes"})


if __name__ == "__main__":
    unittest.main()
