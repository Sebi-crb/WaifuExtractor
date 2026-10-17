"""Pick a character's lines and turn them into padded, merged time ranges."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TypeVar

from .subs import Line

T = TypeVar("T")


@dataclass
class NameStats:
    display: str
    solo: int = 0
    shared: int = 0
    seconds: float = 0.0
    episodes: set[str] = field(default_factory=set)


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def name_catalog(episodes: dict[str, list[Line]]) -> list[NameStats]:
    """Every speaker name across all episodes, grouped case-insensitively, busiest first."""
    stats: dict[str, NameStats] = {}
    for label, lines in episodes.items():
        for line in lines:
            for sp in line.speakers:
                st = stats.setdefault(sp.casefold(), NameStats(sp))
                if len(line.speakers) == 1:
                    st.solo += 1
                    st.seconds += line.end - line.start
                else:
                    st.shared += 1
                st.episodes.add(label)
    return sorted(stats.values(), key=lambda s: (-s.solo, s.display.casefold()))


def _overlap(a: Line, b: Line) -> float:
    return min(a.end, b.end) - max(a.start, b.start)


def build_segments(
    lines: list[Line],
    keys: set[str],
    *,
    pad: float = 0.1,
    merge_gap: float = 0.3,
    include_shared: bool = False,
    keep_overlap: bool = False,
    max_overlap: float = 0.15,
) -> list[Segment]:
    mine: list[Line] = []
    others: list[Line] = []
    for line in lines:
        speakers = {s.casefold() for s in line.speakers}
        if not speakers & keys:
            others.append(line)
        elif speakers <= keys or include_shared:
            mine.append(line)

    if not keep_overlap:
        mine = [m for m in mine if all(_overlap(m, o) <= max_overlap for o in others)]

    merged: list[Segment] = []
    for line in mine:
        start, end = max(0.0, line.start - pad), line.end + pad
        if merged and start - merged[-1].end <= merge_gap:
            merged[-1].end = max(merged[-1].end, end)
            merged[-1].text += " " + line.text
        else:
            merged.append(Segment(start, end, line.text))
    return merged


# Clip lengths that work best as voice references: long enough to carry the voice,
# short enough to be one utterance. Tiers are used in order until the budget is filled.
CLONE_TIERS = [(3.0, 12.0), (1.5, 20.0), (0.0, float("inf"))]


def pick_for_cloning(plan: list[tuple[T, list[Segment]]], target: float, seed: int = 0) -> list[tuple[T, list[Segment]]]:
    """Choose about `target` seconds of clips, preferring good clone lengths and spreading them
    across episodes (round-robin) and through each episode (seeded shuffle, so runs repeat)."""
    rng = random.Random(seed)
    chosen: dict[int, set[int]] = {i: set() for i in range(len(plan))}
    total = 0.0
    for low, high in CLONE_TIERS:
        queues = []
        for i, (_, segs) in enumerate(plan):
            idx = [j for j, s in enumerate(segs) if low <= s.duration <= high and j not in chosen[i]]
            rng.shuffle(idx)
            queues.append((i, idx))
        while total < target and any(q for _, q in queues):
            for i, q in queues:
                if q and total < target:
                    j = q.pop()
                    chosen[i].add(j)
                    total += plan[i][1][j].duration
        if total >= target:
            break
    return [(key, [s for j, s in enumerate(segs) if j in chosen[i]])
            for i, (key, segs) in enumerate(plan) if chosen[i]]
