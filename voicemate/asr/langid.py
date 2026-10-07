"""Hungarian/English language identification for transcripts.

Parakeet does not report the spoken language, and generic detectors misclassify Hungarian
sentences that are full of English technical terms ("Keress rá a transformer
architecture-ökre"). The grammar words decide which language a sentence is in, so this
detector scores function words and Hungarian-only letters first and only asks lingua when
those signals tie.
"""

from __future__ import annotations

import re

#: Supported language codes.
LANGUAGES: tuple[str, str] = ("hu", "en")

#: Hungarian function words and very frequent short words.
HU_WORDS: frozenset[str] = frozenset(
    """a az egy és is hogy nem van volt lesz mi mit mik mennyi milyen melyik hol hova honnan
    mikor miért hogyan ki kit kinek ez ezt azt ezek azok itt ott rá ra meg de vagy ha mert
    csak még már most kérlem kérem légyszi légy szia köszönöm köszi igen nekem neked nekünk
    nálam tőlem velem engem téged mondd mondj keress keresd nézd nézz foglald írd olvasd
    jegyezd mennyit hány kell lehet tudsz tudod szeretnék szeretném holnap ma tegnap
    """.split()
)

#: English function words and very frequent short words.
EN_WORDS: frozenset[str] = frozenset(
    """the is are was were be been what which who whom whose where when why how this that
    these those it its of to in on for with from at by and or but not do does did can could
    would should will shall my your our me you we they he she i please find search tell
    give show explain summarize what's it's there their about
    """.split()
)

#: Letters that occur in Hungarian but not in English.
HU_LETTERS: frozenset[str] = frozenset("áéíóöőúüű")

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


class LanguageDetector:
    """Decide between Hungarian and English for one utterance.

    Args:
        default: Language returned when there is no signal at all.
    """

    def __init__(self, default: str = "en") -> None:
        self.default = default
        self._lingua = None

    def _lingua_detect(self, text: str) -> str | None:
        if self._lingua is None:
            from lingua import Language, LanguageDetectorBuilder

            self._lingua = LanguageDetectorBuilder.from_languages(
                Language.HUNGARIAN, Language.ENGLISH
            ).build()
        language = self._lingua.detect_language_of(text)
        if language is None:
            return None
        return "hu" if language.name == "HUNGARIAN" else "en"

    def detect(self, text: str, previous: str | None = None) -> str:
        """Return ``"hu"`` or ``"en"`` for ``text``.

        Args:
            text: Transcript.
            previous: Language of the previous turn; short utterances without a clear signal
                (e.g. "ok") keep it.
        """
        words = [w.lower() for w in _WORD.findall(text)]
        fallback = previous or self.default
        if not words:
            return fallback
        hu = sum(w in HU_WORDS for w in words)
        en = sum(w in EN_WORDS for w in words)
        # Only function words decide; Hungarian letters inside proper nouns (Gyöngyös) do not
        # make an English sentence Hungarian, so letters are only a tie-breaker.
        if hu != en:
            return "hu" if hu > en else "en"
        if any(ch in HU_LETTERS for w in words for ch in w):
            return "hu"
        if len(words) <= 2:
            return fallback
        return self._lingua_detect(text) or fallback
