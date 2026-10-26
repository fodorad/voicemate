"""Parsers for the saved Icy Veins pages (plain dataclasses out, no knowledge-base logic).

The pages are saved by Adam from his browser into the pages folder. Each parser reads one kind
of page: a hero guide, a tier list, or a text page (map guides, abilities, talents).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

_GUIDE_HREF = re.compile(r"/heroes/([a-z0-9-]+)-build-guide")
_MAP_SLUG = re.compile(r"^map-([a-z0-9-]+)$")
_HERO_SLUG = re.compile(r"hp_50x50_([a-z0-9-]+)")
_ROLE_CLASS = re.compile(r"htl_role_([a-z]+)")
_TIER_ID = re.compile(r"^tier-([sabcd])$")
_MAP_RATINGS = ("stronger", "average", "weaker")


class PageError(ValueError):
    """A saved page does not have the expected structure."""


@dataclass(frozen=True)
class HeroGuide:
    """What a hero's build-guide page says about the draft.

    Attributes:
        slug: Hero slug.
        name: Display name.
        role: Role from the page breadcrumb (singular), e.g. ``Melee Assassin``.
        modified: The page's ``dateModified`` (ISO 8601), empty when absent.
        strengths: Strength bullet points.
        weaknesses: Weakness bullet points.
        synergies: Slugs of heroes that work well with this one.
        counters: Slugs of heroes that counter this one.
        maps: Map slug → ``stronger`` | ``average`` | ``weaker``.
        synergy_text: The page's explanation of the synergies.
        counter_text: The page's explanation of the counters.
    """

    slug: str
    name: str
    role: str
    modified: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    synergies: tuple[str, ...]
    counters: tuple[str, ...]
    maps: dict[str, str] = field(default_factory=dict)
    synergy_text: str = ""
    counter_text: str = ""


@dataclass(frozen=True)
class TierEntry:
    """One hero in a tier list.

    Attributes:
        hero: Hero slug.
        tier: ``S``, ``A``, ``B``, ``C`` or ``D``.
        role: The list's role group (``tank``, ``offlaner``, ...).
        ban: Whether the list recommends banning the hero.
    """

    hero: str
    tier: str
    role: str
    ban: bool


@dataclass(frozen=True)
class TierList:
    """A general or map-specific tier list."""

    entries: tuple[TierEntry, ...]
    modified: str = ""


@dataclass(frozen=True)
class TextPage:
    """A page kept as text sections (map guides, abilities, talents).

    Attributes:
        title: The page title without the site suffix.
        sections: ``(heading, text)`` pairs in page order; empty sections are dropped.
        modified: The page's ``dateModified``, empty when absent.
    """

    title: str
    sections: tuple[tuple[str, str], ...]
    modified: str = ""


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _clean(text: str) -> str:
    """Collapse whitespace and close the gap markup leaves before punctuation."""
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", text)).strip()


def _classes(tag: Tag) -> str:
    return " ".join(tag.get_attribute_list("class"))


def _json_ld(soup: BeautifulSoup, kind: str) -> dict:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("@type") == kind:
            return data
    return {}


def _modified(soup: BeautifulSoup) -> str:
    return str(_json_ld(soup, "Article").get("dateModified", ""))


def _portrait_slugs(container: Tag | None) -> tuple[str, ...]:
    """Hero slugs of the portrait links inside ``container``, in page order, without repeats."""
    if container is None:
        return ()
    slugs: list[str] = []
    for link in container.select("a[href]"):
        match = _GUIDE_HREF.search(str(link["href"]))
        if match and link.select_one("img.hero_portrait") and match.group(1) not in slugs:
            slugs.append(match.group(1))
    return tuple(slugs)


def _bullets(container: Tag | None) -> tuple[str, ...]:
    if container is None:
        return ()
    return tuple(_clean(li.get_text(" ")) for li in container.select("li") if _clean(li.get_text()))


def _explanation(container: Tag | None) -> str:
    """The prose paragraphs next to the portraits of a synergy or counter block."""
    if container is None:
        return ""
    return " ".join(_clean(p.get_text(" ")) for p in container.select("p") if _clean(p.get_text()))


def parse_guide(html: str, slug: str) -> HeroGuide:
    """Parse a hero's build-guide page.

    Args:
        html: The saved page.
        slug: The hero slug the page belongs to (from its file name).

    Raises:
        PageError: If the page has no hero name or no synergies/counters block.
    """
    soup = _soup(html)
    crumbs = _json_ld(soup, "BreadcrumbList").get("itemListElement", [])
    names = [item.get("item", {}).get("name", "") for item in crumbs]
    if len(names) < 4 or not names[-1]:
        raise PageError(f"{slug}: no breadcrumb with the hero name")
    block = soup.select_one(".heroes_synergies_counters")
    if block is None:
        raise PageError(f"{slug}: no synergies and counters block")
    synergies = block.select_one(".heroes_synergies")
    counters = block.select_one(".heroes_counters")
    maps: dict[str, str] = {}
    for rating in _MAP_RATINGS:
        for link in soup.select(f".heroes_maps_{rating} a[data-heroes-tooltip]"):
            match = _MAP_SLUG.match(str(link["data-heroes-tooltip"]))
            if match:
                maps[match.group(1)] = rating
    return HeroGuide(
        slug=slug,
        name=names[-1],
        role=names[-2].removesuffix("s"),
        modified=_modified(soup),
        strengths=_bullets(soup.select_one(".strengths_weaknesses .strengths")),
        weaknesses=_bullets(soup.select_one(".strengths_weaknesses .weaknesses")),
        synergies=_portrait_slugs(synergies),
        counters=_portrait_slugs(counters),
        maps=maps,
        synergy_text=_explanation(synergies),
        counter_text=_explanation(counters),
    )


def parse_tier_list(html: str) -> TierList:
    """Parse a tier-list page into hero → tier entries.

    Raises:
        PageError: If no heroes are found.
    """
    soup = _soup(html)
    entries: list[TierEntry] = []
    for group in soup.select(".htl_l"):
        heading = group.find_previous("h2", id=_TIER_ID)
        role_span = group.select_one("[class*=htl_role_]")
        role_match = _ROLE_CLASS.search(_classes(role_span)) if role_span else None
        if heading is None:
            continue
        tier = str(heading["id"]).removeprefix("tier-").upper()
        for icon in group.select("[class*=hp_50x50_]"):
            classes = _classes(icon)
            hero = _HERO_SLUG.search(classes)
            if hero:
                entries.append(
                    TierEntry(
                        hero=hero.group(1),
                        tier=tier,
                        role=role_match.group(1) if role_match else "",
                        ban="htl_ban_true" in classes,
                    )
                )
    if not entries:
        raise PageError("tier list without heroes")
    return TierList(tuple(entries), _modified(soup))


def parse_text_page(html: str) -> TextPage:
    """Parse a text page into titled sections (one per ``h2``).

    Raises:
        PageError: If the page has no title.
    """
    soup = _soup(html)
    if soup.title is None or not soup.title.string:
        raise PageError("page without a title")
    title = soup.title.string.split(" - ")[0].strip()
    sections: list[tuple[str, str]] = []
    for heading in soup.select("h2"):
        container = heading.find_parent(class_="heading_container") or heading
        parts: list[str] = []
        for sibling in container.find_next_siblings():
            if sibling.find("h2") or sibling.name == "h2":
                break
            parts.append(sibling.get_text(" "))
        text = _clean(" ".join(parts))
        if text:
            sections.append((_clean(heading.get_text()), text))
    return TextPage(title, tuple(sections), _modified(soup))
