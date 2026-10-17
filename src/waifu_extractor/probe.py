"""ffprobe wrapper: which video, audio and subtitle streams a file has."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUB_CODECS = {"ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"}
MEDIA_SUFFIXES = {".mkv", ".mp4", ".m4v", ".webm"}


@dataclass
class Stream:
    index: int  # absolute stream index in the file (0:N)
    kind_index: int  # index among streams of the same type (0:a:N)
    codec: str
    language: str
    title: str
    start_time: float
    default: bool
    attached_pic: bool = False

    def label(self) -> str:
        parts = [self.language or "und", self.codec]
        if self.title:
            parts.append(f'"{self.title}"')
        return " ".join(parts)


@dataclass
class MediaInfo:
    path: Path
    video: list[Stream] = field(default_factory=list)
    audio: list[Stream] = field(default_factory=list)
    subs: list[Stream] = field(default_factory=list)

    @property
    def label(self) -> str:
        return episode_label(self.path)

    @property
    def main_video(self) -> Stream | None:
        return next((v for v in self.video if not v.attached_pic), None)

    @property
    def text_subs(self) -> list[Stream]:
        return [s for s in self.subs if s.codec in TEXT_SUB_CODECS]


def episode_label(path: Path) -> str:
    m = re.search(r"S(\d{1,2})E(\d{1,3})", path.name, re.I)
    if m:
        return f"S{int(m[1]):02d}E{int(m[2]):02d}"
    return re.sub(r"[^\w.-]+", "_", path.stem)[:60]


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = MediaInfo(path)
    counters = {"video": 0, "audio": 0, "subtitle": 0}
    for s in json.loads(out.stdout).get("streams", []):
        kind = s.get("codec_type")
        if kind not in counters:
            continue
        tags = s.get("tags", {})
        disp = s.get("disposition", {})
        stream = Stream(
            index=s["index"],
            kind_index=counters[kind],
            codec=s.get("codec_name", "unknown"),
            language=tags.get("language", ""),
            title=tags.get("title", ""),
            start_time=float(s.get("start_time") or 0.0),
            default=bool(disp.get("default")),
            attached_pic=bool(disp.get("attached_pic")),
        )
        counters[kind] += 1
        {"video": info.video, "audio": info.audio, "subtitle": info.subs}[kind].append(stream)
    return info


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES)
    raise FileNotFoundError(target)
