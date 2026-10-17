"""Pick a character's lines and turn them into padded, merged time ranges."""

from __future__ import annotations

from dataclasses import dataclass, field

from .subs import Line


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
