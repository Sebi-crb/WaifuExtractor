"""Extract text subtitle tracks (cached) and turn them into speaker-tagged lines."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from .probe import MediaInfo, Stream

CACHE_DIR = Path.home() / ".cache" / "waifu-extractor" / "subs"

# Styles used for typesetting, karaoke and credits rather than spoken dialogue.
SIGN_STYLE = re.compile(r"sign|song|^op\b|^ed\b|title|kara|lyric|insert|typeset|credit|logo|eyecatch", re.I)
JUNK_NAMES = {"sign", "signs", "song", "songs", "op", "ed", "note", "tl note", "tln", "title", "text", "credits", "logo"}
# "Aqua & Kazuma", "Aqua/Kazuma", "Aqua and Kazuma" -> several speakers on one line.
SPEAKER_SPLIT = re.compile(r"\s*(?:&|/|\+|,|\band\b)\s*", re.I)
OVERRIDE_TAG = re.compile(r"\{[^}]*\}")
# Karaoke timing tags (\k, \kf, \ko, \K) only appear in song lines.
KARAOKE_TAG = re.compile(r"\\[kK][fo]?\d")


@dataclass(frozen=True)
class Line:
    start: float
    end: float
    speakers: tuple[str, ...]
    text: str


@dataclass
class Track:
    stream: Stream
    lines: list[Line]

    @property
    def named(self) -> int:
        return len(self.lines)


def _cache_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]


def _sub_file(cache: Path, stream: Stream) -> Path:
    return cache / f"{stream.index}.{'ass' if stream.codec in ('ass', 'ssa') else 'srt'}"


def extract_subs(info: MediaInfo) -> tuple[dict[int, Path], bool]:
    """Dump every text subtitle stream in one ffmpeg pass. Returns (files, was_cached)."""
    streams = info.text_subs
    cache = CACHE_DIR / _cache_key(info.path)
    files = {s.index: _sub_file(cache, s) for s in streams}
    if (cache / "done").exists():
        return files, True
    if not streams:
        return {}, False
    cache.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(info.path)]
    for s in streams:
        codec = "copy" if s.codec in ("ass", "ssa", "subrip", "srt") else "srt"
        cmd += ["-map", f"0:{s.index}", "-c:s", codec, str(files[s.index])]
    subprocess.run(cmd, check=True, capture_output=True)
    (cache / "done").touch()
    return files, False


def clean_text(raw: str) -> str:
    text = OVERRIDE_TAG.sub("", raw).replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    return " ".join(text.split())


def parse_track(path: Path) -> list[Line]:
    """Spoken lines that carry a speaker name. Unnamed, sign and karaoke events are dropped."""
    try:
        subs = pysubs2.load(str(path), encoding="utf-8")
    except Exception:
        return []
    lines = []
    for ev in subs.events:
        # Karaoke templater output sets Effect ("fx", "template", ...); dialogue leaves it empty.
        if ev.is_comment or ev.end <= ev.start or ev.effect.strip() or KARAOKE_TAG.search(ev.text):
            continue
        name = ev.name.strip()
        if not name or name.isdigit() or name.startswith("[") or name.casefold() in JUNK_NAMES:
            continue
        if SIGN_STYLE.search(ev.style):
            continue
        text = clean_text(ev.text)
        if not text:
            continue
        speakers = tuple(p for p in (s.strip() for s in SPEAKER_SPLIT.split(name)) if p)
        lines.append(Line(ev.start / 1000, ev.end / 1000, speakers, text))
    lines.sort(key=lambda l: l.start)
    return lines


def load_tracks(info: MediaInfo, files: dict[int, Path]) -> list[Track]:
    return [Track(s, parse_track(files[s.index])) for s in info.text_subs if files.get(s.index, Path()).exists()]


def best_track(tracks: list[Track]) -> Track | None:
    """The track with the most speaker-named lines, or None if no track names anyone."""
    best = max(tracks, key=lambda t: t.named, default=None)
    return best if best and best.named else None
