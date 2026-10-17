"""waifu-extract: pull one character's lines out of anime episodes using speaker-named subtitles."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import __version__, cutter, fx, ui
from .probe import MediaInfo, Stream, collect_files, probe
from .segments import NameStats, Segment, build_segments, name_catalog
from .subs import Line, Track, best_track, extract_subs, load_tracks


@dataclass
class Episode:
    info: MediaInfo
    label: str
    track: Track | None

    @property
    def lines(self) -> list[Line]:
        return self.track.lines if self.track else []


class Abort(Exception):
    """Stop with an error message."""


# --- small helpers ---------------------------------------------------------

def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}m{seconds % 60:06.3f}s"


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_") or "character"


def need(flag: str) -> None:
    """A choice that wasn't given as a flag and can't be asked because there's no terminal."""
    if not ui.interactive():
        raise Abort(f"{flag} is required when not running in a terminal")


# --- scanning --------------------------------------------------------------

def scan(files: list[Path], jobs: int) -> list[Episode]:
    def work(path: Path) -> tuple[MediaInfo, list[Track], bool]:
        info = probe(path)
        sub_files, cached = extract_subs(info)
        return info, load_tracks(info, sub_files), cached

    results: dict[Path, tuple[MediaInfo, list[Track]]] = {}
    cached_count = 0
    with ui.progress(transient=True) as bar, ThreadPoolExecutor(max_workers=jobs) as pool:
        task = bar.add_task("Reading subtitles", total=len(files))
        futures = {pool.submit(work, p): p for p in files}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                info, tracks, cached = fut.result()
                results[path] = (info, tracks)
                cached_count += cached
            except Exception as exc:  # one broken file shouldn't stop the batch
                ui.warn(f"{path.name}: skipped ({exc})")
            bar.advance(task)

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
    fresh = len(results) - cached_count
    ui.ok(f"Scanned {len(results)} file(s)" + (f" [{ui.DIM}]({cached_count} cached, {fresh} read)[/]" if cached_count else ""))
    return episodes


def report_scan(episodes: list[Episode], verbose: bool) -> None:
    named = [e for e in episodes if e.track]
    if verbose:
        ui.table([("Episode", "left"), ("Named lines", "right"), ("Subtitle track", "left")],
                 [(e.label, e.track.named if e.track else "-", e.track.stream.label() if e.track
                   else f"[{ui.DIM}]no speaker names in {len(e.info.text_subs)} text track(s)[/]") for e in episodes])
        ui.console.print()
    tracks = {e.track.stream.title or e.track.stream.label() for e in named}
    if named:
        ui.ok(f"{len(named)} episode(s) have speaker names [{ui.DIM}]· from {', '.join(sorted(tracks))}[/]")
    for e in episodes:
        if not e.track:
            ui.warn(f"{e.label}: no subtitle track names its speakers, skipped")


def report_catalog(catalog: list[NameStats]) -> None:
    ui.table([("#", "right"), ("Character", "left"), ("Lines", "right"), ("Talk time", "right"),
              ("Episodes", "right"), ("Shared", "right")],
             [(i, st.display, st.solo, fmt_dur(st.seconds), len(st.episodes), st.shared or "")
              for i, st in enumerate(catalog, 1)])


# --- choices ---------------------------------------------------------------

def is_variant(a: str, b: str) -> bool:
    """Kaz / Kazuma: the shorter name (3+ letters) starts the longer and the rest is plain letters.
    Rules out combined or labelled credits such as 'Aqua-Kaz' or 'Aqua(ReviewA)'."""
    short, long = sorted((a, b), key=len)
    return len(short) >= 3 and long != short and long.startswith(short) and long[len(short):].isalpha()


def similar_names(chosen: set[str], catalog: list[NameStats]) -> list[NameStats]:
    return [st for st in catalog if st.display.casefold() not in chosen
            and any(is_variant(st.display.casefold(), c) for c in chosen)]


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
            raise Abort(f"no speaker called {part!r} - run with --list to see the names")
    return picked


def character_title(st: NameStats, width: int) -> str:
    return f"{st.display:<{width}}  {st.solo:>5} lines  {fmt_dur(st.seconds):>8}  {len(st.episodes):>3} ep"


def pick_character(args, catalog: list[NameStats]) -> list[NameStats]:
    if args.character:
        picked = resolve_names(",".join(args.character), catalog)
        for st in similar_names({p.display.casefold() for p in picked}, catalog):
            ui.warn(f"Similar name {st.display!r} ({st.solo} lines) not included; add it with -c")
        return picked

    need("--character")
    width = min(24, max(len(s.display) for s in catalog))
    by_key = {st.display.casefold(): st for st in catalog}
    keys = ui.search("Which character?", [(key, character_title(st, width)) for key, st in by_key.items()],
                     answer=lambda titles: ", ".join(t[:width].strip() for t in titles))
    similar = similar_names(set(keys), catalog)
    if similar:
        keys += ui.checkbox(
            "These look like the same character under another spelling. Include them?",
            [(st.display.casefold(), character_title(st, width)) for st in similar],
            checked={st.display.casefold() for st in similar},
            answer=lambda titles: ", ".join(t[:width].strip() for t in titles) or "none",
        )
    return [by_key[k] for k in keys]


def pick_effects(args) -> set[str]:
    flags = {"music": args.remove_music, "dereverb": args.dereverb, "normalize": args.normalize}
    missing = fx.missing()
    if any(v is not None for v in flags.values()) or not ui.interactive():
        effects = {e for e, on in flags.items() if on}
    else:
        effects = set(ui.checkbox(
            "Audio effects?",
            [(e, fx.EFFECTS[e]) for e in fx.ORDER],
            unavailable={e: "not installed, see README" for e in missing},
        ))
    for e in effects & missing.keys():
        raise Abort(f"{fx.EFFECTS[e]} is not installed: {missing[e]}")
    return effects


def default_audio(info: MediaInfo) -> Stream:
    return next((a for a in info.audio if a.language == "jpn"), None) \
        or next((a for a in info.audio if a.default), None) or info.audio[0]


def pick_audio(args, sample: MediaInfo) -> tuple[str | None, int]:
    """Audio preference as (language, index); language wins when an episode has it."""
    if not sample.audio:
        raise Abort(f"{sample.path.name} has no audio track")
    if args.audio_lang or args.audio_track is not None:
        return args.audio_lang, args.audio_track or 0
    default = default_audio(sample)
    if len(sample.audio) > 1 and ui.interactive():
        idx = ui.select("Which audio track?", [(a.kind_index, a.label()) for a in sample.audio],
                        default=default.kind_index)
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
            accurate: bool, effects: set[str], audio_pref: tuple[str | None, int]) -> Path:
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
    try:
        with ui.progress() as bar:
            task = bar.add_task("Starting", total=sum(len(s) for _, s in plan))

            def status(label: str, msg: str) -> None:
                bar.update(task, description=f"[bold]{label}[/] [{ui.DIM}]{msg}[/]")

            for ep, segs in plan:
                audio = resolve_audio(ep.info, *audio_pref)
                video = ep.info.main_video
                if audio is None or (mode == "video" and video is None):
                    ui.warn(f"{ep.label}: missing {'audio' if audio is None else 'video'} stream, skipped")
                    bar.advance(task, len(segs))
                    continue

                wav = None
                if mode == "audio" or effects:
                    status(ep.label, f"decoding {audio.language or 'audio'} track")
                    wav = cutter.decode_audio(ep.info, audio, work / f"{ep.label}.wav", args.mono)
                    if effects & fx.SEPARATIONS:
                        raw, wav = wav, fx.process_excerpts(wav, segs, audio.start_time, effects, work,
                                                            args.mono, lambda m: status(ep.label, m))
                        raw.unlink()

                status(ep.label, "cutting")
                src = cutter.WavSource(wav, audio.start_time) if wav else None
                for n, seg in enumerate(segs, 1):
                    clip = f"{ep.label}_{n:04d}_{fmt_ts(seg.start)}.{ext}"
                    row = [clip if output == "clips" else strip_path.name, ep.label, ep.info.path.name,
                           f"{seg.start:.3f}", f"{seg.end:.3f}", f"{seg.duration:.3f}"]
                    frames = b""
                    if src:
                        frames = src.read(seg.start, seg.end)
                        if "normalize" in effects:
                            frames = fx.normalize_frames(frames, src.params, work, args.loudness)
                    if mode == "audio":
                        if output == "clips":
                            cutter.write_wav(clips_dir / clip, src.params, frames)
                        else:
                            if strip_wav is None:
                                strip_wav = wave.open(str(strip_path), "wb")
                                strip_wav.setparams(src.params)
                            strip_wav.writeframes(frames + src.silence(args.gap))
                    else:
                        # With effects, the clip's processed audio goes in as its own wav starting at seg.start.
                        clip_audio = None
                        if src:
                            clip_audio = work / "clip-audio.wav"
                            cutter.write_wav(clip_audio, src.params, frames)
                        cutter.cut_video(ep.info, seg, clips_dir / clip, video=video, audio=audio,
                                         accurate=accurate, encoder=args.encoder,
                                         audio_file=clip_audio, audio_offset=seg.start)
                        parts.append(clips_dir / clip)
                    if output == "strip":
                        # Fast video cuts start on a keyframe, so the clip can be longer than the line.
                        length = seg.duration if mode == "audio" else cutter.duration(parts[-1])
                        row += [f"{strip_pos:.3f}", f"{strip_pos + length:.3f}"]
                        strip_pos += length + (args.gap if mode == "audio" else 0)
                    writer.writerow(row + [seg.text])
                    bar.advance(task)
                if src:
                    src.close()
                if wav:
                    wav.unlink(missing_ok=True)
                bar.console.print(f"[green]✓[/] {ep.label:<10} [{ui.DIM}]{len(segs)} clip(s)[/]")

            if strip_wav:
                strip_wav.close()
            if mode == "video" and output == "strip" and parts:
                status("strip", "joining clips into one video")
                cutter.concat_videos(parts, strip_path)
            bar.update(task, description="[bold green]Done[/]")
    finally:
        meta.close()
        shutil.rmtree(work, ignore_errors=True)
    return outdir


# --- entry point -----------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="waifu-extract",
        description="Cut one character's lines out of anime episodes, using subtitle tracks that name the speaker. "
                    "Run without options for the interactive mode.",
    )
    p.add_argument("path", type=Path, nargs="?", help="an episode file, or a folder (searched recursively)")
    p.add_argument("-V", "--version", action="version", version=f"waifu-extract {__version__}")
    p.add_argument("--list", action="store_true", help="only print the characters found and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="show which subtitle track each episode uses")
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
    p.add_argument("--remove-music", "--isolate-vocals", dest="remove_music", action=argparse.BooleanOptionalAction,
                   default=None, help="remove music and sound effects, keep voices (needs the [fx] extra)")
    p.add_argument("--dereverb", action=argparse.BooleanOptionalAction, default=None,
                   help="remove reverb/echo from the voices (needs the [fx] extra)")
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                   help="bring every clip to the same loudness (two-pass loudnorm)")
    p.add_argument("--loudness", type=float, default=fx.DEFAULT_LUFS,
                   help=f"loudness target for --normalize in LUFS (default {fx.DEFAULT_LUFS:g})")
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
    except (KeyboardInterrupt, ui.Cancelled):
        ui.console.print(f"\n[{ui.DIM}]Cancelled.[/]")
        sys.exit(130)
    except Abort as exc:
        ui.error(str(exc))
        sys.exit(1)


def run(args) -> None:
    if ui.interactive() and not args.list:
        ui.banner()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise Abort(f"{tool} not found on PATH")
    if args.path is None:
        need("a path")
        args.path = Path(ui.path("Episode file or folder:")).expanduser()
    try:
        files = collect_files(args.path)
    except FileNotFoundError:
        raise Abort(f"{args.path} does not exist") from None
    if not files:
        raise Abort(f"no video files in {args.path}")

    episodes = scan(files, args.jobs)
    report_scan(episodes, args.verbose)
    named = [e for e in episodes if e.track]
    if not named:
        raise Abort("no subtitle track in these files names its speakers, so there is nothing to cut by")
    catalog = name_catalog({e.label: e.lines for e in named})
    if args.list:
        ui.console.print()
        report_catalog(catalog)
        return
    ui.console.print()

    picked = pick_character(args, catalog)
    keys = {p.display.casefold() for p in picked}
    name = max(picked, key=lambda p: (len(p.display), p.solo)).display

    mode = args.mode
    if mode is None:
        need("--mode")
        mode = ui.select("What should come out?", [("audio", "Audio only   .wav, for voice clips and datasets"),
                                                   ("video", "Video        .mkv with picture and sound")], "audio")
    output = args.output
    if output is None:
        need("--output")
        output = ui.select("Cut how?", [("clips", "Clips        a folder with one file per line"),
                                        ("strip", "One strip    every line back to back in one file")], "clips")
    accurate = True
    if mode == "video":
        cut = args.cut
        if cut is None and ui.interactive():
            cut = ui.select("Video cuts?", [("accurate", "Accurate     re-encode, cuts land exactly on the line"),
                                            ("fast", "Fast         no re-encode, cuts snap to keyframes")],
                            "accurate")
        accurate = cut != "fast"

    effects = pick_effects(args)
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
        raise Abort(f"no usable lines for {name} (try --keep-overlap or --include-shared)")
    seconds = sum(seg.duration for _, segs in plan for seg in segs)
    lang, idx = audio_pref
    ui.console.print()
    ui.panel([
        ("Character", f"[bold {ui.PINK}]{name}[/]" + (f" [{ui.DIM}]({', '.join(p.display for p in picked)})[/]"
                                                     if len(picked) > 1 else "")),
        ("Lines", f"{count} clips · {fmt_dur(seconds)} · {len(plan)} episode(s)"),
        ("Output", ("audio .wav" if mode == "audio" else "video .mkv") + (", one strip" if output == "strip" else ", clips")
         + ("" if mode == "audio" else (", accurate cuts" if accurate else ", fast cuts"))),
        ("Effects", ", ".join(fx.EFFECTS[e] for e in fx.ORDER if e in effects) or f"[{ui.DIM}]none[/]"),
        ("Audio track", lang or f"track {idx}"),
        ("Save to", str(args.out.resolve() / f"{safe_name(name)}_{mode}_{output}")),
    ], "Ready")
    if not args.yes and ui.interactive() and not ui.confirm("Start?", True):
        raise ui.Cancelled

    outdir = extract(args, plan, name, mode, output, accurate, effects, audio_pref)
    ui.console.print()
    ui.panel([("Clips", f"{count} · {fmt_dur(seconds)}"), ("Folder", str(outdir.resolve())),
              ("Transcripts", "metadata.tsv")], "Done", color="green")
    if ui.interactive() and not args.yes and shutil.which("xdg-open") and ui.confirm("Open the folder?", False):
        subprocess.Popen(["xdg-open", str(outdir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
