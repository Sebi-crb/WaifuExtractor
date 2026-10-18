"""Find lines a character says in subtitles that don't name the speakers: by search, or from a list of quotes.

Her picked lines then teach the voice matcher what she sounds like, in place of reference clips.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import TypeVar

from .subs import Line

T = TypeVar("T")

NOT_WORD = re.compile(r"[\W_]+")
SIMILAR = 0.85  # how close a quote has to be to a subtitle line (typos, punctuation, small wording changes)
PART = 0.6  # a quote inside a longer line must cover this much of it; the rest may be someone else
MIN_PIECE = 12  # a line inside a longer quote (subs often split one sentence) needs this many letters
SHORT_WORDS = 4  # quotes this short that turn up in several places are probably said by others too
SHORT_CHARS = 8  # the same for scripts without spaces


def norm(text: str) -> str:
    """Lowercase, no punctuation, single spaces: what's left to compare."""
    return NOT_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def search(texts: list[tuple[T, list[Line]]], query: str) -> list[tuple[T, Line]]:
    """Lines containing the query, ignoring case and punctuation, in episode order."""
    q = norm(query)
    if not q:
        return []
    return [(ep, line) for ep, lines in texts for line in lines if q in norm(line.text)]


def _matches(quote: str, line: str) -> bool:
    if not line:
        return False
    if quote in line:
        return len(quote) >= PART * len(line)
    if line in quote:
        return len(line.replace(" ", "")) >= MIN_PIECE
    sm = SequenceMatcher(None, quote, line, autojunk=False)
    return sm.real_quick_ratio() >= SIMILAR and sm.quick_ratio() >= SIMILAR and sm.ratio() >= SIMILAR


def _short(quote: str) -> bool:
    return len(quote.split()) < SHORT_WORDS if " " in quote else len(quote) < SHORT_CHARS


def match_quotes(texts: list[tuple[T, list[Line]]], quotes: list[str]
                 ) -> tuple[list[tuple[T, Line]], list[str], list[str]]:
    """Subtitle lines matching the quotes. Returns (lines, quotes found nowhere, short quotes found in
    several places and skipped). The same moment in two subtitle tracks counts as one place."""
    normed = [(ep, line, norm(line.text)) for ep, lines in texts for line in lines]
    found, missing, common = [], [], []
    for raw in quotes:
        q = norm(raw)
        if not q:
            continue
        hits = [(ep, line) for ep, line, text in normed if _matches(q, text)]
        places = []
        for ep, line in hits:
            if not any(ep is e and same_moment(line, l) for e, l in places):
                places.append((ep, line))
        if not hits:
            missing.append(raw)
        elif len(places) > 1 and _short(q):
            common.append(raw)
        else:
            found += hits
    return found, missing, common


def same_moment(a: Line, b: Line) -> bool:
    """Do two lines (maybe from different subtitle tracks) mostly cover the same time?"""
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return overlap > 0.5 * min(a.end - a.start, b.end - b.start)
