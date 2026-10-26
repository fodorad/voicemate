"""The draft as the session knows it: who the enemy and allies picked, bans and the map."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Draft:
    """One draft in progress.

    Attributes:
        enemy: Hero slugs the enemy picked, in pick order.
        ally: Hero slugs on Adam's team, in pick order.
        banned: Hero slugs already banned (by either side).
        map_slug: The map being played, if known.
    """

    enemy: list[str] = field(default_factory=list)
    ally: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)
    map_slug: str | None = None

    def side(self, name: str) -> list[str]:
        """The list for ``enemy``, ``ally`` or ``banned``."""
        return {"enemy": self.enemy, "ally": self.ally, "banned": self.banned}[name]

    def add(self, name: str, heroes: list[str]) -> None:
        """Add heroes to a side (heroes already there are kept once)."""
        side = self.side(name)
        side.extend(h for h in heroes if h not in side)

    def remove(self, name: str, heroes: list[str]) -> list[str]:
        """Remove heroes from a side; returns the ones that were actually there."""
        side = self.side(name)
        gone = [h for h in heroes if h in side]
        side[:] = [h for h in side if h not in gone]
        return gone

    def reset(self) -> None:
        """Start a new draft."""
        self.enemy.clear()
        self.ally.clear()
        self.banned.clear()
        self.map_slug = None
