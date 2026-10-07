"""Turn LLM output into text a TTS voice can read aloud.

The LLM is asked to answer in plain spoken style, but markdown, URLs and emoji still slip
through; reading "asterisk asterisk" or a URL aloud is useless, so they are stripped here.
The full text (with links) is still shown in the UI.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

_CODE_BLOCK = re.compile(r"```.*?(```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"(https?://|www\.)\S+")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_LIST_MARK = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
# Emphasis markers only count next to non-word characters, so "2*3*4" stays intact.
_EMPHASIS = re.compile(r"(?<![\w*])(\*{1,3}|_{2,3})(\S(?:.*?\S)?)\1(?![\w*])")
_TABLE_PIPE = re.compile(r"\s*\|\s*")
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff\U00002b00-\U00002bff"
    "\U0000fe0f\U0000200d]"
)
_SPACES = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")


def speakable(text: str, pronunciation: Mapping[str, str] | None = None) -> str:
    """Strip markup from ``text`` and apply pronunciation replacements.

    Args:
        text: Raw LLM output (possibly markdown).
        pronunciation: Whole-word replacements, e.g. ``{"Ava": "Éva"}``.

    Returns:
        Single-line plain text.
    """
    text = _CODE_BLOCK.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARK.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _TABLE_PIPE.sub(" ", text)
    text = _EMOJI.sub("", text)
    for word, spoken in (pronunciation or {}).items():
        text = re.sub(rf"\b{re.escape(word)}\b", spoken, text)
    text = _SPACES.sub(" ", text).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)
