"""Extract text subtitle tracks (cached) and turn them into dialogue lines, speaker-tagged where the subs say who talks."""

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
# OP/ED also as a word inside a name ("Konosuba-OP") or glued to a lyrics language ("edeng", "edrom").
SIGN_STYLE = re.compile(r"sign|song|(?:^|[\W_])(?:op|ed)(?:$|[\W_\d]|eng|rom|kan|jp)|title|kara|lyric|insert"
                        r"|typeset|credit|logo|eyecatch", re.I)
JUNK_NAMES = {"sign", "signs", "song", "songs", "op", "ed", "note", "tl note", "tln", "title", "text", "credits", "logo"}
# Notes fansubbers leave in the name field for each other (translation check, edit, honorific doubt...).
# The line is still dialogue, just without a speaker.
STAFF_NOTE = re.compile(r"^(?:(?:mega)?tlc|tl|edit(?:or)?|qc|ts|timing|hon|honorifics?|check|note|todo|"
                        r"\?+)\??$", re.I)
STAFF_SPLIT = re.compile(r"[\s;,/&+]+")
MIN_NAMED_SHARE = 0.3  # a track naming fewer of its lines than this isn't speaker-tagged (real ones: ~99 %)
# "Aqua & Kazuma", "Aqua/Kazuma", "Aqua and Kazuma" -> several speakers on one line.
SPEAKER_SPLIT = re.compile(r"\s*(?:&|/|\+|,|\band\b)\s*", re.I)
OVERRIDE_TAG = re.compile(r"\{[^}]*\}")
# Karaoke timing tags (\k, \kf, \ko, \K) only appear in song lines.
KARAOKE_TAG = re.compile(r"\\[kK][fo]?\d")
# Tracks that only carry signs and songs; their timing isn't dialogue.
SIGN_TRACK = re.compile(r"sign|song|forced", re.I)
MIN_DIALOGUE = 30  # a track with fewer lines than this isn't a full dialogue track
# Text placed on screen by hand is a sign, not someone talking; ♪ marks sung lines.
POSITIONED = re.compile(r"\\(?:pos|move|org|i?clip)\(")


@dataclass(frozen=True)
class Line:
    start: float
    end: float
    speakers: tuple[str, ...]  # empty when the subs don't say who talks
    text: str
    typeset: bool = False  # positioned or sung: no use as dialogue timing


@dataclass
class Track:
    stream: Stream
    lines: list[Line]

    @property
    def named(self) -> int:
        return sum(1 for line in self.lines if line.speakers)


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
    """Spoken lines, with their speaker names where the subs give them. Sign and karaoke events are dropped."""
    try:
        subs = pysubs2.load(str(path), encoding="utf-8")
    except Exception:
        return []
    # A style mostly used for positioned text is a sign style, even on its few unpositioned lines.
    placed: dict[str, list[int]] = {}
    for ev in subs.events:
        counts = placed.setdefault(ev.style, [0, 0])
        counts[0] += bool(POSITIONED.search(ev.text))
        counts[1] += 1
    sign_styles = {style for style, (pos, total) in placed.items() if pos * 2 >= total}
    lines = []
    for ev in subs.events:
        # Karaoke templater output sets Effect ("fx", "template", ...); dialogue leaves it empty.
        if ev.is_comment or ev.end <= ev.start or ev.effect.strip() or KARAOKE_TAG.search(ev.text):
            continue
        name = ev.name.strip()
        if name and all(STAFF_NOTE.match(part) for part in STAFF_SPLIT.split(name) if part):
            name = ""
        if name and (name.isdigit() or name.startswith("[") or name.casefold() in JUNK_NAMES):
            continue
        if SIGN_STYLE.search(ev.style):
            continue
        text = clean_text(ev.text)
        if not text:
            continue
        speakers = tuple(p for p in (s.strip() for s in SPEAKER_SPLIT.split(name)) if p)
        typeset = ev.style in sign_styles or bool(POSITIONED.search(ev.text)) or "♪" in text
        lines.append(Line(ev.start / 1000, ev.end / 1000, speakers, text, typeset))
    lines.sort(key=lambda l: l.start)
    return lines


def load_tracks(info: MediaInfo, files: dict[int, Path]) -> list[Track]:
    return [Track(s, parse_track(files[s.index])) for s in info.text_subs if files.get(s.index, Path()).exists()]


def best_track(tracks: list[Track]) -> Track | None:
    """The track with the most speaker-named lines, or None if no track names its speakers. A few names
    on a mostly unnamed track are left-over notes or one-off labels, not speaker tags."""
    best = max(tracks, key=lambda t: t.named, default=None)
    return best if best and best.named and best.named >= MIN_NAMED_SHARE * len(best.lines) else None


def spoken_tracks(tracks: list[Track]) -> list[Track]:
    """Every dialogue track, named or not, reduced to its spoken lines. Typeset lines and duplicate
    layers of the same event are left out."""
    out = []
    for t in tracks:
        if SIGN_TRACK.search(t.stream.title):
            continue
        seen, spoken = set(), []
        for line in t.lines:
            if not line.typeset and (line.start, line.end, line.text) not in seen:
                seen.add((line.start, line.end, line.text))
                spoken.append(line)
        if len(spoken) >= MIN_DIALOGUE:
            out.append(Track(t.stream, spoken))
    return out


def dialogue_track(tracks: list[Track]) -> Track | None:
    """The fullest spoken track: its timing says where people talk."""
    return max(spoken_tracks(tracks), key=lambda t: len(t.lines), default=None)
