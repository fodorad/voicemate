"""Summaries of the per-turn metrics log (``make report``).

This is the Stage-6 feedback loop: when a budget from docs/sdlc/intent.md is breached, the
report says so, and the finding goes back into the intent as new work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Latency budgets from intent.md (seconds, p50).
BUDGETS: dict[str, float] = {"chat_first_audio_s": 2.5, "tool_first_audio_s": 8.0}

#: Maximum acceptable share of failed turns.
MAX_ERROR_RATE: float = 0.05


@dataclass
class Report:
    """Aggregated turn metrics.

    Attributes:
        turns: Number of turns.
        chat_first_audio: p50/p95 first-audio latency of turns without tools.
        tool_first_audio: p50/p95 first-audio latency of turns with tools.
        error_rate: Share of failed turns.
        tool_counts: Calls per tool.
        breaches: Human-readable budget violations.
    """

    turns: int = 0
    chat_first_audio: tuple[float, float] | None = None
    tool_first_audio: tuple[float, float] | None = None
    error_rate: float = 0.0
    tool_counts: dict[str, int] = field(default_factory=dict)
    breaches: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Plain-text summary."""

        def fmt(pair: tuple[float, float] | None) -> str:
            return "n/a" if pair is None else f"p50 {pair[0]:.2f} s, p95 {pair[1]:.2f} s"

        lines = [
            f"Turns: {self.turns}",
            f"First audio, chat turns: {fmt(self.chat_first_audio)}",
            f"First audio, tool turns: {fmt(self.tool_first_audio)}",
            f"Failed turns: {self.error_rate:.1%}",
        ]
        if self.tool_counts:
            tools = ", ".join(f"{k} {v}" for k, v in sorted(self.tool_counts.items()))
            lines.append(f"Tool calls: {tools}")
        lines.append("Budget breaches: " + ("; ".join(self.breaches) if self.breaches else "none"))
        return "\n".join(lines)


def _percentiles(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, 50)), float(np.percentile(array, 95))


def summarize(log_path: Path) -> Report:
    """Aggregate ``turns.jsonl``; a missing file yields an empty report."""
    report = Report()
    if not log_path.exists():
        return report
    chat, tools = [], []
    failed = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        turn = json.loads(line)
        report.turns += 1
        failed += bool(turn.get("failed"))
        for name in turn.get("tools", []):
            report.tool_counts[name] = report.tool_counts.get(name, 0) + 1
        if "first_audio_s" in turn and not turn.get("failed"):
            (tools if turn.get("tools") else chat).append(turn["first_audio_s"])
    report.chat_first_audio = _percentiles(chat)
    report.tool_first_audio = _percentiles(tools)
    report.error_rate = failed / report.turns if report.turns else 0.0
    if report.chat_first_audio and report.chat_first_audio[0] > BUDGETS["chat_first_audio_s"]:
        report.breaches.append(
            f"chat first audio p50 {report.chat_first_audio[0]:.2f} s > "
            f"{BUDGETS['chat_first_audio_s']} s"
        )
    if report.tool_first_audio and report.tool_first_audio[0] > BUDGETS["tool_first_audio_s"]:
        report.breaches.append(
            f"tool first audio p50 {report.tool_first_audio[0]:.2f} s > "
            f"{BUDGETS['tool_first_audio_s']} s"
        )
    if report.error_rate > MAX_ERROR_RATE:
        report.breaches.append(f"error rate {report.error_rate:.1%} > {MAX_ERROR_RATE:.0%}")
    return report
