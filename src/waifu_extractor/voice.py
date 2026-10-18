"""Find a character by her voice where no subtitle names the speaker.

Every stretch of speech gets a voice fingerprint (WeSpeaker ResNet34, run through onnxruntime). A small
classifier learns her voice against everyone else's: from episodes whose subs do name the speakers, or,
with only reference clips of her, against all speech in the episodes being searched (lines that already
look like hers are taken out of that pile and the classifier retrained). Measured on KonoSuba S1 with the
names hidden, lines scoring 0.8+ were hers ~98 % of the time; 0.3-0.8 is a mix worth listening to.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # the [voice] extra isn't installed; missing() says so
    np = None

from .probe import MediaInfo, Stream
from .subs import Line, _cache_key

MODEL_URL = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx"
MODEL_PATH = Path.home() / ".cache" / "waifu-extractor" / "models" / "wespeaker" / "voxceleb_resnet34_LM.onnx"
CACHE_DIR = Path.home() / ".cache" / "waifu-extractor" / "voice"
INSTALL_HINT = "install the [voice] extra (see README)"

RATE = 16000
MIN_LEN = 0.4  # shorter speech can't be fingerprinted reliably
MAX_LEN = 20.0  # longer isn't one utterance (and costs minutes to fingerprint)
MAX_TRAIN_EPISODES = 6  # named episodes used to learn the voices; more adds time, not accuracy
PU_ROUNDS = 3  # retrain rounds in reference-only mode
REF_WINDOW = 8.0  # long reference speech is split into windows of this length
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".opus", ".ogg", ".m4a", ".aac", ".mkv", ".mp4", ".webm"}


def missing() -> str | None:
    """Why voice matching can't run here, or None if it can."""
    for mod in ("numpy", "onnxruntime", "sklearn", "silero_vad"):
        if importlib.util.find_spec(mod) is None:
            return INSTALL_HINT
    return None


# --- audio -------------------------------------------------------------------

def decode16(path: Path, stream: Stream | None = None) -> np.ndarray:
    """One audio stream as 16 kHz mono float32. Sample 0 is the stream's start_time."""
    mapping = f"0:a:{stream.kind_index}" if stream else "0:a:0"
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", mapping, "-f", "s16le",
                          "-ar", str(RATE), "-ac", "1", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


def speech_spans(audio: np.ndarray) -> list[tuple[float, float]]:
    """Where someone is talking (Silero VAD), in seconds from the start of `audio`."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=True)  # small; one per call keeps threads from sharing its state
    stamps = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=RATE, return_seconds=True,
                                   min_speech_duration_ms=int(MIN_LEN * 1000), min_silence_duration_ms=250,
                                   max_speech_duration_s=15, speech_pad_ms=60)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


# --- fingerprints --------------------------------------------------------------

def _mel_banks(bins: int = 80, fft: int = 512, low: float = 20.0, high: float = RATE / 2) -> np.ndarray:
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (bins + 1)
    freqs = mel(np.arange(fft // 2) * RATE / fft)
    left = lo + np.arange(bins)[:, None] * delta
    up = (freqs - left) / delta
    down = (left + 2 * delta - freqs) / delta
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))  # Kaldi leaves the Nyquist bin out


@functools.cache
def _window() -> tuple[np.ndarray, np.ndarray]:
    return _mel_banks(), 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)


def fbank(clip: np.ndarray) -> np.ndarray:
    """80-bin Kaldi log-mel filterbank (25 ms / 10 ms), mean-normalised: what WeSpeaker was trained on.
    Matches torchaudio.compliance.kaldi.fbank with dither 0 and a hamming window."""
    mel, hamming = _window()
    x = clip.astype(np.float64) * 32768
    n = 1 + (len(x) - 400) // 160
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160][:n].copy()
    frames -= frames.mean(1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    power = np.abs(np.fft.rfft(frames * hamming, 512)) ** 2
    feats = np.log(np.maximum(power @ mel.T, np.finfo(np.float32).eps))
    return (feats - feats.mean(0)).astype(np.float32)


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        part = MODEL_PATH.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, part)
        part.rename(MODEL_PATH)
    return MODEL_PATH


class Fingerprinter:
    """WeSpeaker ResNet34 on CPU. One instance can be shared by threads (onnxruntime run() is thread-safe)."""

    def __init__(self) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(ensure_model()), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        e = self.session.run(None, {self.input: fbank(clip)[None]})[0][0]
        return e / np.linalg.norm(e)

    def spans(self, audio: np.ndarray, spans: list[tuple[float, float]], offset: float = 0.0) -> np.ndarray:
        """One fingerprint per span (episode seconds); NaN rows where there's too little audio."""
        out = np.full((len(spans), 256), np.nan, dtype=np.float32)
        for i, (a, b) in enumerate(spans):
            clip = audio[max(0, int((a - offset) * RATE)):max(0, int((b - offset) * RATE))]
            if MIN_LEN * RATE <= len(clip) <= MAX_LEN * RATE and np.abs(clip).max() > 1e-3:
                out[i] = self(clip)
        return out


@dataclass
class Prints:
    starts: np.ndarray
    ends: np.ndarray
    emb: np.ndarray  # NaN rows: not fingerprinted

    @property
    def ok(self) -> np.ndarray:
        return ~np.isnan(self.emb[:, 0])


def episode_prints(fp: Fingerprinter, info: MediaInfo, audio: Stream,
                   spans: list[tuple[float, float]] | None) -> Prints:
    """Fingerprints for the given spans, or for every stretch of speech the VAD finds when `spans` is None.
    Cached per file, audio track and span list, so trying another character on the same show is quick."""
    tag = "vad" if spans is None else hashlib.sha1(np.asarray(spans).tobytes()).hexdigest()[:12]
    cache = CACHE_DIR / f"{_cache_key(info.path)}_{audio.index}_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        return Prints(d["starts"], d["ends"], d["emb"])
    pcm = decode16(info.path, audio)
    if spans is None:
        spans = [(a + audio.start_time, b + audio.start_time) for a, b in speech_spans(pcm)]
    emb = fp.spans(pcm, spans, audio.start_time)
    prints = Prints(np.array([a for a, _ in spans]), np.array([b for _, b in spans]), emb)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, starts=prints.starts, ends=prints.ends, emb=prints.emb)
    return prints


def reference_prints(fp: Fingerprinter, path: Path) -> np.ndarray:
    """Fingerprints of a clip that holds only her voice: its speech, cut into windows of up to REF_WINDOW s."""
    pcm = decode16(path)
    windows = []
    for a, b in speech_spans(pcm):
        while b - a >= 1.0:
            windows.append((a, min(b, a + REF_WINDOW)))
            a += REF_WINDOW
    emb = fp.spans(pcm, windows)
    return emb[~np.isnan(emb[:, 0])]


def reference_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


# --- learning her voice ----------------------------------------------------------

def overlapping(lines: list[Line], max_overlap: float = 0.15) -> list[bool]:
    """Per line (sorted by start): does another line run over it, probably someone else talking at the
    same time? Unnamed lines always count as different speakers."""
    flags = [False] * len(lines)
    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if b.start >= a.end:
                break
            if min(a.end, b.end) - b.start > max_overlap and (not a.speakers or a.speakers != b.speakers):
                flags[i] = flags[j] = True
    return flags


def train(pos: np.ndarray, neg: np.ndarray | None, pool: np.ndarray):
    """A scorer: fingerprint rows -> probability she is speaking.

    With `neg` (other speakers from named episodes) it's a plain classifier. Without, the episodes'
    own speech (`pool`) stands in for 'not her'; since some of it *is* her, lines that score as hers
    are taken out of that pile and the classifier retrained a few times."""
    from sklearn.linear_model import LogisticRegression

    def fit(negatives: np.ndarray):
        X = np.concatenate([pos, negatives])
        y = np.r_[np.ones(len(pos)), np.zeros(len(negatives))]
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(X, y)

    if neg is not None and len(neg):
        clf = fit(neg)
        return lambda rows: clf.predict_proba(rows)[:, 1]
    keep = np.ones(len(pool), dtype=bool)
    clf = None
    for _ in range(PU_ROUNDS + 1):
        clf = fit(pool[keep])
        keep = clf.predict_proba(pool)[:, 1] < 0.5
        if keep.sum() < 10:
            break
    return lambda rows: clf.predict_proba(rows)[:, 1]


# --- voice groups ------------------------------------------------------------------

GROUPS = 15  # more groups than speakers: one voice split over two groups is fine (tick both), two in one isn't
MAX_GROUPED = 4000  # spans grouped at most (a random sample); the classifier then scores all of them
GROUP_CORE = 0.7  # share of a ticked group, closest to its average voice, that teaches the classifier


def voice_groups(rows: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Group fingerprints by voice. Returns (indexes of the rows grouped, their group labels).

    Spectral clustering on a nearest-neighbour graph: on KonoSuba it kept Aqua, Megumin and Chris apart,
    where plain similarity cut-offs merged them into one group."""
    from sklearn.cluster import SpectralClustering

    idx = np.arange(len(rows))
    if len(idx) > MAX_GROUPED:
        idx = np.sort(np.random.default_rng(seed).choice(idx, MAX_GROUPED, replace=False))
    k = min(GROUPS, max(2, len(idx) // 10))
    labels = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=10,
                                random_state=seed).fit_predict(rows[idx])
    return idx, labels


def by_closeness(rows: np.ndarray) -> np.ndarray:
    """Row order, most typical of the group's voice first."""
    centre = rows.mean(0)
    return np.argsort(-(rows @ centre))


def group_core(rows: np.ndarray) -> np.ndarray:
    """The part of a group closest to its average voice: leaves out the odd line of someone else."""
    return rows[by_closeness(rows)[:max(1, round(len(rows) * GROUP_CORE))]]
