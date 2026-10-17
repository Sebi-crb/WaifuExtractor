"""ffmpeg work: decode audio, cut clips, concatenate strips."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from .probe import MediaInfo, Stream
from .segments import Segment

VAAPI_DEVICE = "/dev/dri/renderD128"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{res.stderr.strip()[-2000:]}")


# --- audio -----------------------------------------------------------------

def decode_audio(info: MediaInfo, audio: Stream, out: Path, mono: bool) -> Path:
    """Decode one audio stream of the whole episode to 16-bit PCM wav (one read of the file)."""
    # Fixed rate/layout so wavs from different episodes can be joined into one strip.
    run(["ffmpeg", "-v", "error", "-y", "-i", str(info.path), "-map", f"0:a:{audio.kind_index}",
         "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1" if mono else "2", str(out)])
    return out


def decode_spans(info: MediaInfo, audio: Stream, segs: list[Segment], out: Path, mono: bool,
                 context: float = 1.0) -> Path:
    """Like decode_audio, but seeks to each clip (plus `context` seconds each side) instead of reading
    the whole file. Much faster over a network share when an episode only has a few clips.
    The wav keeps episode timing: gaps are silence, and it ends after the last clip."""
    rate, channels = 48000, 1 if mono else 2
    frame = channels * 2
    spans: list[list[float]] = []
    for seg in segs:
        a, b = max(0.0, seg.start - context), seg.end + context
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    with wave.open(str(out), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        pos = 0
        for a, b in spans:
            pcm = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(info.path),
                 "-map", f"0:a:{audio.kind_index}", "-f", "s16le", "-ar", str(rate), "-ac", str(channels), "-"],
                capture_output=True, check=True).stdout
            start = max(0, int((a - audio.start_time) * rate))
            if start > pos:
                w.writeframes(b"\0" * ((start - pos) * frame))
                pos = start
            elif start < pos:  # overlaps what's written already
                pcm = pcm[(pos - start) * frame:]
            w.writeframes(pcm)
            pos += len(pcm) // frame
    return out


class WavSource:
    """Random access into a decoded episode wav. `offset` is the audio stream's start time."""

    def __init__(self, path: Path, offset: float = 0.0):
        self.w = wave.open(str(path), "rb")
        self.params = self.w.getparams()
        self.offset = offset

    def read(self, start: float, end: float) -> bytes:
        rate, total = self.params.framerate, self.params.nframes
        a = min(total, max(0, int((start - self.offset) * rate)))
        b = min(total, max(a, int((end - self.offset) * rate)))
        self.w.setpos(a)
        return self.w.readframes(b - a)

    def silence(self, seconds: float) -> bytes:
        return b"\0" * (int(seconds * self.params.framerate) * self.params.nchannels * self.params.sampwidth)

    def close(self) -> None:
        self.w.close()


def write_wav(path: Path, params: wave._wave_params, frames: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(frames)


# --- video -----------------------------------------------------------------

def cut_video(
    info: MediaInfo,
    seg: Segment,
    out: Path,
    *,
    video: Stream,
    audio: Stream,
    accurate: bool,
    encoder: str,
    audio_file: Path | None = None,
    audio_offset: float = 0.0,
) -> None:
    """Cut one clip. `audio_file` replaces the soundtrack with a processed episode wav."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if accurate and encoder == "vaapi":
        cmd += ["-vaapi_device", VAAPI_DEVICE]
    cmd += ["-ss", f"{seg.start:.3f}", "-i", str(info.path)]
    if audio_file:
        cmd += ["-ss", f"{max(0.0, seg.start - audio_offset):.3f}", "-i", str(audio_file)]
    cmd += ["-t", f"{seg.duration:.3f}", "-map", f"0:v:{video.kind_index}",
            "-map", "1:a:0" if audio_file else f"0:a:{audio.kind_index}"]

    if not accurate:
        cmd += ["-c:v", "copy"]
    elif encoder == "vaapi":
        cmd += ["-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "20"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]

    if accurate or audio_file:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "copy"]
    run(cmd + ["-avoid_negative_ts", "make_zero", "-sn", "-dn", str(out)])


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip() or 0)


def concat_videos(parts: list[Path], out: Path) -> None:
    listing = out.with_suffix(".concat.txt")
    listing.write_text("".join("file '{}'\n".format(str(p).replace("'", "'\\''")) for p in parts))
    try:
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    finally:
        listing.unlink(missing_ok=True)
