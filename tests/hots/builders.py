"""Small hand-written pages that mimic the structure of the saved Icy Veins pages.

They are written for the tests (no page content is copied), so parsers and the knowledge base
can be tested on a tiny, exactly known roster.
"""

from __future__ import annotations

import json
from pathlib import Path


def _portraits(slugs: list[str]) -> str:
    return "".join(
        f'<div><a href="//www.icy-veins.com/heroes/{slug}-build-guide">'
        f'<img class="hero_portrait" title="{slug}" alt="{slug} Portrait"></a></div>'
        for slug in slugs
    )


def guide_html(
    slug: str,
    name: str,
    role: str = "Tanks",
    *,
    modified: str = "2026-01-01T00:00:00+00:00",
    strengths: tuple[str, ...] = ("Tough",),
    weaknesses: tuple[str, ...] = ("Slow",),
    synergies: tuple[str, ...] = (),
    counters: tuple[str, ...] = (),
    maps: dict[str, str] | None = None,
) -> str:
    """A hero build-guide page."""
    article = {"@type": "Article", "dateModified": modified}
    crumbs = {
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"item": {"name": n}} for n in ("Home", "Heroes of the Storm", role, name)
        ],
    }
    ratings = ""
    for rating in ("stronger", "average", "weaker"):
        links = "".join(
            f'<a data-heroes-tooltip="map-{m}" href="//www.icy-veins.com/heroes/{m}-map-guide">'
            f'<img alt="{m}"></a>'
            for m, value in (maps or {}).items()
            if value == rating
        )
        ratings += (
            f'<div class="heroes_maps_{rating}">'
            f'<div class="heroes_maps_content">{links}</div></div>'
        )
    lis = lambda items: "".join(f"<li><img alt='+'><span>{t}</span></li>" for t in items)  # noqa: E731
    return f"""<html><head><title>{name} Build Guide - Heroes of the Storm - Icy Veins</title>
<script type="application/ld+json">{json.dumps(article)}</script>
<script type="application/ld+json">{json.dumps(crumbs)}</script></head><body>
<div class="strengths_weaknesses"><div class="strengths"><ul>{lis(strengths)}</ul></div>
<div class="weaknesses"><ul>{lis(weaknesses)}</ul></div></div>
<div class="heroes_synergies_counters">
<div class="heroes_synergies"><div class="heroes_synergies_counters_content">
<div>{_portraits(list(synergies))}</div>
<div><p>{name} likes friends.</p></div></div></div>
<div class="heroes_counters"><div class="heroes_synergies_counters_content">
<div>{_portraits(list(counters))}</div>
<div><p>{name} fears foes.</p></div></div></div></div>
<div class="heroes_maps"><div class="heroes_maps_rankings">{ratings}</div></div>
</body></html>"""


def tier_html(groups: dict[tuple[str, str], list[tuple[str, bool]]]) -> str:
    """A tier list; ``groups`` maps ``(tier letter, role)`` to ``[(hero slug, ban?)]``."""
    body = ""
    for tier in "sabcd":
        body += (
            f'<div class="heading_container"><h2 id="tier-{tier}">Tier {tier.upper()}</h2></div>'
        )
        body += '<div class="htl">'
        for (letter, role), heroes in groups.items():
            if letter.lower() != tier:
                continue
            body += f'<div class="htl_l"><div><span class="htl_role_{role}"></span></div><div>'
            for slug, ban in heroes:
                body += (
                    f'<span><a href="//www.icy-veins.com/heroes/{slug}-build-guide">'
                    f'<span class="hp_50x50 hp_50x50_{slug} htl_ban_{str(ban).lower()}"></span>'
                    f"<span>{slug}</span></a></span>"
                )
            body += "</div></div>"
        body += "</div>"
    head = "<title>Tier List - Heroes of the Storm - Icy Veins</title>"
    return f"<html><head>{head}</head><body>{body}</body></html>"


def text_html(title: str, sections: dict[str, str]) -> str:
    """A text page with one ``h2`` section per entry."""
    body = "".join(
        f'<div class="heading_container"><h2>{heading}</h2></div><p>{text}</p>'
        for heading, text in sections.items()
    )
    head = f"<title>{title} - Heroes of the Storm - Icy Veins</title>"
    return f"<html><head>{head}</head><body>{body}</body></html>"


def write_pages(folder: Path, pages: dict[str, str]) -> None:
    """Write ``{file stem: html}`` into ``folder`` as ``heroes_<stem>.html``."""
    folder.mkdir(parents=True, exist_ok=True)
    for stem, html in pages.items():
        (folder / f"heroes_{stem}.html").write_text(html, encoding="utf-8")
