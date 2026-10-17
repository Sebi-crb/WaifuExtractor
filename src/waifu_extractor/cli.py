"""waifu-extract: pull one character's lines out of anime episodes using speaker-named subtitles."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import cutter
from .probe import MediaInfo, Stream, collect_files, probe
from .segments import NameStats, Segment, build_segments, name_catalog
from .subs import Line, Track, best_track, extract_subs, load_tracks

SHOWN_NAMES = 40


@dataclass
class Episode:
    info: MediaInfo
    label: str
    track: Track | None

    @property
    def lines(self) -> list[Line]:
        return self.track.lines if self.track else []


# --- small helpers ---------------------------------------------------------

def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}m{seconds % 60:06.3f}s"


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_") or "character"


def interactive() -> bool:
    return sys.stdin.isatty()


def choose(prompt: str, options: list[tuple[str, str]], default: str) -> str:
    """Numbered menu; returns the chosen option's value."""
    print(f"\n{prompt}")
    for i, (value, label) in enumerate(options, 1):
        print(f"  {i}) {label}{'  (default)' if value == default else ''}")
    values = [v for v, _ in options]
    while True:
        ans = input("> ").strip().lower()
        if not ans:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return values[int(ans) - 1]
        if ans in values:
            return ans
        print("  pick a number from the list")


def yes_no(prompt: str, default: bool) -> bool:
    ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not ans else ans.startswith("y")


def need(value, flag: str):
    """A choice that wasn't given as a flag and can't be asked because there's no terminal."""
    if value is None and not interactive():
        die(f"{flag} is required when not running in a terminal")
    return value


# --- scanning --------------------------------------------------------------

def scan(files: list[Path], jobs: int) -> list[Episode]:
    print(f"Scanning {len(files)} file(s) for subtitle tracks (first scan reads each whole file)...")

    def work(path: Path) -> tuple[MediaInfo, list[Track], bool]:
        info = probe(path)
        sub_files, cached = extract_subs(info)
        return info, load_tracks(info, sub_files), cached

    results: dict[Path, tuple[MediaInfo, list[Track]]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(work, p): p for p in files}
        for n, fut in enumerate(as_completed(futures), 1):
            path = futures[fut]
            try:
                info, tracks, cached = fut.result()
            except Exception as exc:  # one broken file shouldn't stop the batch
                print(f"  [{n}/{len(files)}] {path.name}: skipped ({exc})")
                continue
            results[path] = (info, tracks)
            print(f"  [{n}/{len(files)}] {info.label}{' (cached)' if cached else ''}")

    episodes, seen = [], set()
    for path in files:
        if path not in results:
            continue
        info, tracks = results[path]
        label = info.label
        while label in seen:
            label += "_2"
        seen.add(label)
        episodes.append(Episode(info, label, best_track(tracks)))
    return episodes


def print_scan(episodes: list[Episode]) -> None:
    print("\nSpeaker-named subtitle track per episode:")
    for ep in episodes:
        if ep.track:
            print(f"  {ep.label:<12} {ep.track.named:>4} named lines  <- {ep.track.stream.label()}")
        else:
            subs = len(ep.info.text_subs)
            print(f"  {ep.label:<12} no speaker names in {subs} text track(s) - skipped")


def print_catalog(catalog: list[NameStats], limit: int | None = None) -> None:
    shown = catalog[:limit] if limit else catalog
    width = max((len(s.display) for s in shown), default=4)
    print("\nCharacters found:")
    for i, st in enumerate(shown, 1):
        shared = f"  (+{st.shared} shared)" if st.shared else ""
        print(f"  {i:>3}) {st.display:<{width}}  {st.solo:>5} lines  {fmt_dur(st.seconds):>8}  "
              f"{len(st.episodes):>3} ep{shared}")
    if limit and len(catalog) > limit:
        print(f"  ... {len(catalog) - limit} more with fewer lines (type a name to pick one of them)")


# --- choices ---------------------------------------------------------------

def similar_names(chosen: set[str], catalog: list[NameStats]) -> list[NameStats]:
    """Spelling variants such as Kaz / Kazuma: one name is a prefix of the other."""
    out = []
    for st in catalog:
        key = st.display.casefold()
        if key not in chosen and any(key.startswith(c) or c.startswith(key) for c in chosen if len(c) >= 3 and len(key) >= 3):
            out.append(st)
    return out


def resolve_names(raw: str, catalog: list[NameStats]) -> list[NameStats]:
    by_key = {s.display.casefold(): s for s in catalog}
    picked = []
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        if part.isdigit() and 1 <= int(part) <= len(catalog):
            picked.append(catalog[int(part) - 1])
        elif part.casefold() in by_key:
            picked.append(by_key[part.casefold()])
        else:
            raise ValueError(part)
    return picked


def pick_character(args, catalog: list[NameStats]) -> list[NameStats]:
    if args.character:
        try:
            picked = resolve_names(",".join(args.character), catalog)
        except ValueError as exc:
            die(f"no speaker called {exc} - run with --list to see the names")
        for st in similar_names({p.display.casefold() for p in picked}, catalog):
            print(f"note: similar name {st.display!r} ({st.solo} lines) not included; add it with --character")
        return picked

    need(None, "--character")
    print_catalog(catalog, SHOWN_NAMES)
    while True:
        raw = input("\nWhich character? Numbers or names, comma-separated for spelling variants (e.g. 2,7): ")
        try:
            picked = resolve_names(raw, catalog)
        except ValueError as exc:
            print(f"  no speaker called {exc}")
            continue
        if picked:
            break
    for st in similar_names({p.display.casefold() for p in picked}, catalog):
        if yes_no(f"Also include {st.display!r} ({st.solo} lines, {len(st.episodes)} ep)? Looks like the same character.", True):
            picked.append(st)
    return picked


def default_audio(info: MediaInfo) -> Stream:
    return next((a for a in info.audio if a.language == "jpn"), None) \
        or next((a for a in info.audio if a.default), None) or info.audio[0]


def pick_audio(args, sample: MediaInfo) -> tuple[str | None, int]:
    """Audio preference as (language, index); language wins when an episode has it."""
    if not sample.audio:
        die(f"{sample.path.name} has no audio track")
    if args.audio_lang or args.audio_track is not None:
        return args.audio_lang, args.audio_track or 0
    default = default_audio(sample)
    if len(sample.audio) > 1 and interactive():
        options = [(str(a.kind_index), a.label()) for a in sample.audio]
        idx = int(choose("Which audio track?", options, str(default.kind_index)))
        return sample.audio[idx].language or None, idx
    return default.language or None, default.kind_index


def resolve_audio(info: MediaInfo, lang: str | None, idx: int) -> Stream | None:
    if lang:
        match = next((a for a in info.audio if a.language == lang), None)
        if match:
            return match
    if idx < len(info.audio):
        return info.audio[idx]
    return info.audio[0] if info.audio else None


def unique_dir(base: Path) -> Path:
    path, n = base, 2
    while path.exists() and any(path.iterdir()):
        path = base.with_name(f"{base.name}_{n}")
        n += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- extraction ------------------------------------------------------------

def extract(args, plan: list[tuple[Episode, list[Segment]]], name: str, mode: str, output: str,
            accurate: bool, isolate: bool, audio_pref: tuple[str | None, int]) -> Path:
    outdir = unique_dir(args.out / f"{safe_name(name)}_{mode}_{output}")
    work = Path(tempfile.mkdtemp(prefix=".work-", dir=outdir))
    clips_dir = outdir if output == "clips" else work / "parts"
    clips_dir.mkdir(exist_ok=True)
    ext = "wav" if mode == "audio" else "mkv"
    strip_path = outdir / f"{safe_name(name)}.{ext}"

    meta = open(outdir / "metadata.tsv", "w", newline="")
    writer = csv.writer(meta, delimiter="\t")
    writer.writerow(["file", "episode", "source", "start", "end", "duration"]
                    + (["strip_start", "strip_end"] if output == "strip" else []) + ["text"])

    strip_wav: wave.Wave_write | None = None
    strip_pos = 0.0
    parts: list[Path] = []
    total = sum(len(s) for _, s in plan)
    done = 0
    try:
        for ep, segs in plan:
            audio = resolve_audio(ep.info, *audio_pref)
            video = ep.info.main_video
            if audio is None or (mode == "video" and video is None):
                print(f"  {ep.label}: missing {'audio' if audio is None else 'video'} stream, skipped")
                done += len(segs)
                continue

            wav = None
            if mode == "audio" or isolate:
                print(f"  {ep.label}: decoding audio ({audio.label()})...")
                wav = cutter.decode_audio(ep.info, audio, work / f"{ep.label}.wav", args.mono)
                if isolate:
                    print(f"  {ep.label}: isolating vocals with Demucs...")
                    raw, wav = wav, cutter.isolate_vocals(wav, work / "demucs", args.mono)
                    raw.unlink()
                    shutil.rmtree(work / "demucs", ignore_errors=True)

            src = cutter.WavSource(wav, audio.start_time) if mode == "audio" else None
            for n, seg in enumerate(segs, 1):
                clip = f"{ep.label}_{n:04d}_{fmt_ts(seg.start)}.{ext}"
                row = [clip if output == "clips" else strip_path.name, ep.label, ep.info.path.name,
                       f"{seg.start:.3f}", f"{seg.end:.3f}", f"{seg.duration:.3f}"]
                if src:
                    frames = src.read(seg.start, seg.end)
                    if output == "clips":
                        cutter.write_wav(clips_dir / clip, src.params, frames)
                    else:
                        if strip_wav is None:
                            strip_wav = wave.open(str(strip_path), "wb")
                            strip_wav.setparams(src.params)
                        strip_wav.writeframes(frames + src.silence(args.gap))
                else:
                    cutter.cut_video(ep.info, seg, clips_dir / clip, video=video, audio=audio,
                                     accurate=accurate, encoder=args.encoder,
                                     vocals=wav, vocals_offset=audio.start_time)
                    parts.append(clips_dir / clip)
                if output == "strip":
                    # Fast video cuts start on a keyframe, so the clip can be longer than the line.
                    length = seg.duration if src else cutter.duration(parts[-1])
                    row += [f"{strip_pos:.3f}", f"{strip_pos + length:.3f}"]
                    strip_pos += length + (args.gap if src else 0)
                writer.writerow(row + [seg.text])
                done += 1
                print(f"\r  {ep.label}: {n}/{len(segs)} clips   (total {done}/{total})", end="", flush=True)
            print()
            if src:
                src.close()
            if wav:
                wav.unlink(missing_ok=True)

        if strip_wav:
            strip_wav.close()
        if mode == "video" and output == "strip" and parts:
            print("  joining clips into one video...")
            cutter.concat_videos(parts, strip_path)
    finally:
        meta.close()
        shutil.rmtree(work, ignore_errors=True)
    return outdir


# --- entry point -----------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="waifu-extract",
        description="Cut one character's lines out of anime episodes, using subtitle tracks that name the speaker.",
    )
    p.add_argument("path", type=Path, help="an episode file, or a folder (searched recursively)")
    p.add_argument("--list", action="store_true", help="only print the characters found and exit")
    p.add_argument("-c", "--character", action="append",
                   help="speaker name(s) or list numbers, comma-separated; repeatable (e.g. -c Kaz,Kazuma)")
    p.add_argument("-m", "--mode", choices=["audio", "video"], help="wav audio or mkv video")
    p.add_argument("-O", "--output", choices=["clips", "strip"], help="one file per line, or one long file")
    p.add_argument("-o", "--out", type=Path, default=Path("waifu-extract"), help="output root (default: ./waifu-extract)")
    cut = p.add_mutually_exclusive_group()
    cut.add_argument("--accurate", dest="cut", action="store_const", const="accurate",
                     help="re-encode video for frame-accurate cuts (default)")
    cut.add_argument("--fast", dest="cut", action="store_const", const="fast",
                     help="stream-copy video: instant, but cuts snap to keyframes")
    p.add_argument("--encoder", choices=["x264", "vaapi"], default="x264", help="encoder for accurate video cuts")
    p.add_argument("--audio-lang", help="audio track language, e.g. jpn or eng (default: jpn if present)")
    p.add_argument("--audio-track", type=int, help="audio track number (0 = first audio track)")
    p.add_argument("--isolate-vocals", action=argparse.BooleanOptionalAction, default=None,
                   help="strip music/SFX with Demucs (needs the [vocals] extra)")
    p.add_argument("--mono", action="store_true", help="mix audio down to mono")
    p.add_argument("--pad", type=float, default=0.1, help="seconds added before and after each line (default 0.1)")
    p.add_argument("--merge-gap", type=float, default=0.3,
                   help="join lines closer together than this many seconds (default 0.3)")
    p.add_argument("--gap", type=float, default=0.3, help="silence between lines in an audio strip (default 0.3)")
    p.add_argument("--min-length", type=float, default=0.0, help="drop clips shorter than this many seconds")
    p.add_argument("--include-shared", action="store_true",
                   help="also use lines credited to several speakers at once (e.g. 'Aqua & Kazuma')")
    p.add_argument("--keep-overlap", action="store_true",
                   help="keep lines that overlap another speaker's line (dropped by default)")
    p.add_argument("--limit", type=int, help="only extract the first N clips (handy for a preview)")
    p.add_argument("-j", "--jobs", type=int, default=3, help="files scanned in parallel (default 3)")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)


def run(args) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            die(f"{tool} not found on PATH")
    try:
        files = collect_files(args.path)
    except FileNotFoundError:
        die(f"{args.path} does not exist")
    if not files:
        die(f"no video files in {args.path}")

    episodes = scan(files, args.jobs)
    print_scan(episodes)
    named = [e for e in episodes if e.track]
    if not named:
        die("no subtitle track in these files names its speakers, so there is nothing to cut by")
    catalog = name_catalog({e.label: e.lines for e in named})
    if args.list:
        print_catalog(catalog)
        return

    picked = pick_character(args, catalog)
    keys = {p.display.casefold() for p in picked}
    name = picked[0].display if len(picked) == 1 else max(picked, key=lambda p: len(p.display)).display

    mode = args.mode or need(None, "--mode") or choose(
        "Output type?", [("audio", "audio only (.wav)"), ("video", "video with sound (.mkv)")], "audio")
    output = args.output or need(None, "--output") or choose(
        "Cut or one strip?", [("clips", "a folder with one clip per line"), ("strip", "one long file")], "clips")
    accurate = True
    if mode == "video":
        cut = args.cut
        if cut is None and interactive():
            cut = choose("Video cutting?", [("accurate", "re-encode, frame-accurate (slower)"),
                                            ("fast", "stream copy, instant, snaps to keyframes")], "accurate")
        accurate = cut != "fast"

    isolate = args.isolate_vocals
    if isolate is None:
        isolate = interactive() and cutter.demucs_available() and yes_no("Isolate vocals with Demucs (slow)?", False)
    if isolate and not cutter.demucs_available():
        die("--isolate-vocals needs Demucs: install with  uv tool install 'waifu-extractor[vocals]'  (see README)")

    audio_pref = pick_audio(args, named[0].info)

    plan = []
    for ep in named:
        segs = build_segments(ep.lines, keys, pad=args.pad, merge_gap=args.merge_gap,
                              include_shared=args.include_shared, keep_overlap=args.keep_overlap)
        segs = [s for s in segs if s.duration >= args.min_length]
        if segs:
            plan.append((ep, segs))
    if args.limit:
        left, trimmed = args.limit, []
        for ep, segs in plan:
            if left <= 0:
                break
            trimmed.append((ep, segs[:left]))
            left -= len(trimmed[-1][1])
        plan = trimmed

    count = sum(len(s) for _, s in plan)
    if not count:
        die(f"no usable lines for {name} (try --keep-overlap or --include-shared)")
    seconds = sum(seg.duration for _, segs in plan for seg in segs)
    print(f"\n{name} ({', '.join(p.display for p in picked)}): {count} clips, {fmt_dur(seconds)} "
          f"across {len(plan)} episode(s) -> {mode} {output}"
          + (", vocals isolated" if isolate else "")
          + ("" if mode == "audio" else (", frame-accurate" if accurate else ", fast cut")))
    if not args.yes and interactive() and not yes_no("Start?", True):
        return

    outdir = extract(args, plan, name, mode, output, accurate, isolate, audio_pref)
    print(f"\nDone: {outdir}")


if __name__ == "__main__":
    main()
