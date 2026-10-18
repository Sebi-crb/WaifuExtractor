"""Optional audio effects: music removal and de-reverb (audio-separator models), loudness normalisation (ffmpeg)."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Callable

from .cutter import run, write_wav
from .segments import Segment

MODEL_DIR = Path.home() / ".cache" / "waifu-extractor" / "models"
DEFAULT_LUFS = -20.0
CONTEXT = 1.0  # seconds of surrounding audio kept around each clip, so excerpt seams stay out of the clips

EFFECTS = {
    "music": "Remove music and sound effects",
    "dereverb": "Remove reverb and echo",
    "normalize": "Normalize loudness",
}
ORDER = ["music", "dereverb", "normalize"]
# Folder names when every step is saved: what the clips are after that effect.
STEP_NAMES = {"music": "no-music", "dereverb": "no-reverb", "normalize": "normalized"}
SEPARATIONS = {"music", "dereverb"}

# effect -> (model file, regex matching the stem we keep in audio-separator's output names)
MODELS = {
    # BS-Roformer: vocals SDR 13.0 vs 10.8 for Demucs htdemucs_ft
    "music": ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", re.compile(r"\(vocals\)", re.I)),
    "dereverb": ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", re.compile(r"\(no[ _-]?reverb\)", re.I)),
}
INSTALL_HINT = "install the [fx] extra plus PyTorch for your GPU (see README)"


def missing() -> dict[str, str]:
    """Effects that can't run in this environment, with the reason."""
    if importlib.util.find_spec("audio_separator") is None:
        return {e: INSTALL_HINT for e in SEPARATIONS}
    return {}


def gpu_env() -> dict[str, str]:
    """ROCm only ships kernels for gfx1030 among RDNA2 cards; the rest (6600/6650/6700...) run fine posing as it."""
    env = dict(os.environ)
    env.setdefault("MIOPEN_FIND_MODE", "FAST")  # stops MIOpen re-tuning kernels for every new input length
    if "HSA_OVERRIDE_GFX_VERSION" not in env:
        for props in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
            m = re.search(r"gfx_target_version (\d+)", props.read_text())
            if m and 100301 <= int(m[1]) <= 100309:
                env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
                break
    return env


def separate(wav: Path, effect: str, workdir: Path, mono: bool) -> Path:
    """Run one separation model over a wav; returns the kept stem in the decode format."""
    model, stem = MODELS[effect]
    out = workdir / "separated"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    exe = Path(sys.executable).with_name("audio-separator")
    run([str(exe), str(wav), "--model_filename", model, "--model_file_dir", str(MODEL_DIR),
         "--output_dir", str(out), "--output_format", "WAV", "--log_level", "warning"], env=gpu_env())
    kept = [p for p in out.iterdir() if stem.search(p.name)]
    if not kept:
        raise RuntimeError(f"{model} gave no matching stem, got: {', '.join(p.name for p in out.iterdir())}")
    result = wav.with_name(f"{wav.stem}.{effect}.wav")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(kept[0]), "-c:a", "pcm_s16le", "-ar", "48000",
         "-ac", "1" if mono else "2", str(result)])
    shutil.rmtree(out)
    return result


def _spans(segs: list[Segment], offset: float, rate: int, nframes: int) -> list[list[int]]:
    """Frame ranges of the episode wav that clips will use, widened by CONTEXT and merged."""
    spans: list[list[int]] = []
    for seg in segs:
        a = max(0, int((seg.start - offset - CONTEXT) * rate))
        b = min(nframes, int((seg.end - offset + CONTEXT) * rate))
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        elif b > a:
            spans.append([a, b])
    return spans


def process_excerpts(wav: Path, segs: list[Segment], offset: float, effects: set[str],
                     workdir: Path, mono: bool, status: Callable[[str], None] = lambda _: None,
                     every_step: bool = False) -> list[Path]:
    """Run the separation models only over the parts of the episode that get cut.

    The models run at roughly real time on a mid-range GPU, so a whole episode per model is far too
    slow when a character speaks for a few minutes. The processed audio is written back at its
    original position in an episode-length wav (silence elsewhere), so cutting works unchanged.
    Returns that wav after the last model, or with `every_step` one after each model, in order.
    """
    with wave.open(str(wav), "rb") as w:
        params = w.getparams()
        spans = _spans(segs, offset, params.framerate, params.nframes)
        chunks = []
        for a, b in spans:
            w.setpos(a)
            chunks.append(w.readframes(b - a))
    excerpt = workdir / "excerpt.wav"
    write_wav(excerpt, params, b"".join(chunks))
    minutes = sum(b - a for a, b in spans) / params.framerate / 60
    # Music removal first: de-reverb models are trained on isolated vocals.
    steps = [e for e in ("music", "dereverb") if e in effects]
    outs = []
    for effect in steps:
        status(f"{EFFECTS[effect].lower()} on {minutes:.1f} min of audio (AI model)")
        processed = separate(excerpt, effect, workdir, mono)
        excerpt.unlink()
        excerpt = processed
        if every_step or effect == steps[-1]:
            outs.append(_put_back(excerpt, spans, params, wav.with_name(f"{wav.stem}.{effect}.wav")))
    excerpt.unlink()
    return outs


def _put_back(excerpt: Path, spans: list[list[int]], params: wave._wave_params, out: Path) -> Path:
    """Processed excerpt -> episode-length wav with each span back at its place."""
    frame = params.nchannels * params.sampwidth
    with wave.open(str(excerpt), "rb") as src, wave.open(str(out), "wb") as dst:
        dst.setparams(params)
        pos = 0
        for a, b in spans:
            dst.writeframes(b"\0" * ((a - pos) * frame))
            # Resampling inside the model can shift the length by a few samples; pad or trim to fit.
            data = src.readframes(b - a).ljust((b - a) * frame, b"\0")
            dst.writeframes(data[:(b - a) * frame])
            pos = b
        dst.writeframes(b"\0" * ((params.nframes - pos) * frame))
    return out


# --- loudness ----------------------------------------------------------------

def _loudnorm(target: float) -> str:
    return f"loudnorm=I={target:g}:TP=-1.5:LRA=11"


def measure(input_args: list[str], target: float = DEFAULT_LUFS) -> dict | None:
    """First loudnorm pass. None when the clip is too short or silent to measure."""
    res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *input_args,
                          "-af", _loudnorm(target) + ":print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr)
    if not m:
        return None
    stats = json.loads(m[0])
    try:
        if not math.isfinite(float(stats["input_i"])):
            return None
    except ValueError:
        return None
    return stats


def loudnorm_filter(stats: dict | None, target: float) -> str | None:
    """Second pass with the measured values; linear mode keeps it a plain gain change, no pumping."""
    if stats is None:
        return None
    return (f"{_loudnorm(target)}:measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
            f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
            f":offset={stats['target_offset']}:linear=true")


def normalize_frames(frames: bytes, params: wave._wave_params, workdir: Path, target: float) -> bytes:
    """Normalize one clip's PCM frames; returns them unchanged if the clip can't be measured."""
    src, dst = workdir / "norm-in.wav", workdir / "norm-out.wav"
    write_wav(src, params, frames)
    af = loudnorm_filter(measure(["-i", str(src)], target), target)
    if af is None:
        return frames
    # loudnorm resamples to 192 kHz internally, so pin rate and layout back
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af, "-ar", str(params.framerate),
         "-ac", str(params.nchannels), "-c:a", "pcm_s16le", str(dst)])
    with wave.open(str(dst), "rb") as w:
        return w.readframes(w.getnframes())
