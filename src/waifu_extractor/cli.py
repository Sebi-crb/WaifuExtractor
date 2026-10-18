"""waifu-extract: pull one character's lines out of anime episodes using speaker-named subtitles."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, cutter, fx, quotes, ui, voice
from .probe import MediaInfo, Stream, collect_files, probe
from .segments import (REF_CHECK, REF_SURE, VOICE_CHECK, VOICE_SURE, NameStats, Segment, build_segments, name_catalog,
                       pick_for_cloning, voice_segments)
from .subs import Line, Track, best_track, dialogue_track, extract_subs, load_tracks, spoken_tracks


@dataclass
class Episode:
    info: MediaInfo
    label: str
    track: Track | None  # the track that names its speakers
    dialogue: Track | None = None  # the fullest dialogue track, named or not (timing for voice matching)
    texts: list[Track] = field(default_factory=list)  # every dialogue track, to search for her lines

    @property
    def lines(self) -> list[Line]:
        return [line for line in self.track.lines if line.speakers] if self.track else []


MAX_HITS = 60  # search results shown at once when picking her lines
MIN_PICKED = 15  # seconds of picked lines below which matching gets unreliable
MIN_GROUP = 5  # voice groups smaller than this aren't shown
SAMPLES = 3  # lines shown and played per voice group
SAMPLE_MAX = 4.0  # seconds played per line
TYPICAL = 20  # a voice group's example lines come from this many of its most typical
EXAMPLE_MIN = 2.0  # seconds; shorter lines are only shown when a group has nothing longer
# Up to this many clips, seek to each one instead of decoding the episode's whole audio track.
SEEK_MAX_CLIPS = 30


class Abort(Exception):
    """Stop with an error message."""


# --- small helpers ---------------------------------------------------------

def fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def fmt_clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}m{seconds % 60:06.3f}s"


DURATION = re.compile(r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m(?:in)?)?(?:(?P<s>\d+(?:\.\d+)?)s?)?$")


def parse_duration(text: str) -> float | None:
    """'90', '90s', '1.5m', '10min', '1h30m', '1:30', '1:02:00' -> seconds. 'all' / '' -> None (everything)."""
    text = text.strip().lower().replace(" ", "")
    if text in ("", "all", "everything"):
        return None
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    m = DURATION.match(text)
    if not m or not any(m.groupdict().values()):
        raise ValueError(text)
    return sum(float(m[k] or 0) * f for k, f in (("h", 3600), ("m", 60), ("s", 1)))


def valid_duration(text: str) -> bool:
    try:
        value = parse_duration(text)
    except ValueError:
        return False
    return value is None or value > 0


# Presets for the amount question: (seconds, label, what it's good for)
AMOUNT_PRESETS = [(30 * 60, "30 min", "fine-tuning a voice model"), (90, "90 s", "zero-shot voice cloning")]


def pick_amount(args, available: float) -> float | None:
    """Seconds of audio wanted, or None for everything."""
    if args.amount is not None:
        try:
            return parse_duration(args.amount)
        except ValueError:
            raise Abort(f"can't read --amount {args.amount!r}; use e.g. 90s, 10m, 1h30m or all") from None
    if not ui.interactive():
        return None
    choices: list[tuple[object, str]] = [(None, f"Everything   {fmt_dur(available)}")]
    choices += [(secs, f"{label:<12} {use}") for secs, label, use in AMOUNT_PRESETS if secs < available]
    choices.append(("custom", "Custom…      type an amount"))
    picked = ui.select("How much?", choices, default=None)
    if picked != "custom":
        return picked
    return parse_duration(ui.text("Amount:", validate=valid_duration, invalid="Try 45s, 10m, 1h30m or 1:30",
                                  instruction="e.g. 45s, 10m, 1h30m, 1:30"))


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
        episodes.append(Episode(info, label, best_track(tracks), dialogue_track(tracks), spoken_tracks(tracks)))
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
    unnamed = [e.label for e in episodes if not e.track]
    if unnamed:
        shown = ", ".join(unnamed[:6]) + (f" and {len(unnamed) - 6} more" if len(unnamed) > 6 else "")
        ui.warn(f"{len(unnamed)} episode(s) have no speaker names: {shown}")


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


NONE = "\0none"  # "None of these" in the character list


def pick_character(args, catalog: list[NameStats]) -> list[NameStats]:
    """The names to cut by. Empty when she isn't in the list (or the names are wrong): find her by voice."""
    if args.character:
        picked = resolve_names(",".join(args.character), catalog)
        for st in similar_names({p.display.casefold() for p in picked}, catalog):
            ui.warn(f"Similar name {st.display!r} ({st.solo} lines) not included; add it with -c")
        return picked

    need("--character")
    width = min(24, max(len(s.display) for s in catalog))
    by_key = {st.display.casefold(): st for st in catalog}
    choices = [(key, character_title(st, width)) for key, st in by_key.items()]
    if not voice.missing():
        choices.insert(0, (NONE, "None of these  she isn't listed or the names are wrong: find her by voice"))
    keys = ui.search("Which character?", choices, answer=lambda titles: ", ".join(t[:width].strip() for t in titles))
    keys = [k for k in keys if k != NONE]
    if not keys:
        return []
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
    if not effects:
        args.every_step = False
    elif args.every_step is None:
        args.every_step = ui.interactive() and ui.confirm(
            "Also save a copy before the effects and after each one (to compare)?", False)
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


def pick_name(args) -> str:
    """The character's name when no subs name anyone, so it can't be picked from a list."""
    if args.character:
        return ", ".join(args.character).split(",")[0].strip()
    need("--character")
    return ui.text("Character name:", validate=lambda s: bool(s.strip()), invalid="Type a name").strip()


def pick_voice(args, named: list[Episode], unnamed: list[Episode], name: str) -> bool:
    """Search the episodes without speaker names by voice?"""
    if not unnamed:
        return False
    reason = voice.missing()
    if reason:
        ui.warn(f"{len(unnamed)} episode(s) without speaker names skipped; to search them by voice, {reason}")
        return False
    if args.voice is not None or not named:
        return args.voice is not False
    if not ui.interactive():
        return False
    return ui.confirm(f"{len(unnamed)} episode(s) have no speaker names. Find {name} there by her voice?", True)


def pick_reference(args, name: str, required: bool) -> list[Path]:
    """Clips of her voice. Required when no episode names her, optional extra material otherwise."""
    paths = list(args.reference or [])
    if not paths and required:
        need("--reference")
        paths = [Path(ui.path(f"Clips of {name}'s voice (file or folder):")).expanduser()]
    try:
        files = voice.reference_files(paths)
    except FileNotFoundError as exc:
        raise Abort(f"{exc} does not exist") from None
    if paths and not files:
        raise Abort(f"no audio or video files in {', '.join(map(str, paths))}")
    return files


def pick_lines_file(path: Path, texts: list[tuple[Episode, list[Line]]]) -> list[tuple[Episode, Line]]:
    """--lines: quotes of hers, one per row, found in the subtitles."""
    try:
        rows = path.expanduser().read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise Abort(f"can't read {path}: {exc.strerror}") from None
    found, missing, common = quotes.match_quotes(texts, rows)
    if missing:
        ui.warn(f"{len(missing)} quote(s) not in any subtitle: " + "; ".join(missing[:5])
                + (" ..." if len(missing) > 5 else ""))
    if common:
        ui.warn(f"{len(common)} short quote(s) turn up in several places, maybe said by others too, skipped: "
                + "; ".join(common[:5]) + (" ..." if len(common) > 5 else ""))
    if not found:
        raise Abort(f"none of the quotes in {path} are in the subtitles")
    return found


def search_lines(texts: list[tuple[Episode, list[Line]]], name: str) -> list[tuple[Episode, Line]]:
    """Search the subtitles and tick the lines she says, until Enter on an empty search."""
    picked: dict[tuple[str, Line], tuple[Episode, Line]] = {}
    ui.step(f"Search for things {name} says and tick her lines. Aim for 30 s or more; Enter on an empty search when done")
    while True:
        query = ui.text("Search her lines:", validate=lambda q: bool(q.strip() or picked),
                        invalid="Tick at least one line first", instruction="a few words, empty to finish")
        if not query.strip():
            return list(picked.values())
        hits = quotes.search(texts, query)
        if not hits:
            ui.warn("No subtitle line has that")
            continue
        if len(hits) > MAX_HITS:
            ui.warn(f"{len(hits)} lines have that, showing the first {MAX_HITS}; search for more of the line to narrow it")
            hits = hits[:MAX_HITS]
        width = max(40, ui.console.width - 30)
        keys = [(ep.label, line) for ep, line in hits]
        # Values are indexes: the prompt turns dataclass values into dicts
        ticked = set(ui.checkbox("Which are hers?", [
            (i, f"{ep.label:<8} {fmt_clock(line.start)}  {line.end - line.start:4.1f}s  {line.text[:width]}")
            for i, (ep, line) in enumerate(hits)], checked={i for i, k in enumerate(keys) if k in picked},
            answer=lambda titles: f"{len(titles)} ticked"))
        for i, (k, hit) in enumerate(zip(keys, hits)):
            if i in ticked:
                picked[k] = hit
            else:
                picked.pop(k, None)
        total = sum(line.end - line.start for _, line in picked.values())
        ui.ok(f"{len(picked)} line(s), {total:.0f} s of her voice")


def pick_teacher(args, episodes: list[Episode], name: str, required: bool
                 ) -> tuple[list[Path], list[tuple[Episode, Line]], bool]:
    """What teaches the app her voice when no subs name her: reference clips, her lines in the subs,
    or voice groups (picked later, once the audio track is known). The bool says groups."""
    texts = [(ep, [line for t in ep.texts for line in t.lines]) for ep in episodes if ep.texts]
    refs = pick_reference(args, name, required=False)
    picks = []
    if args.lines:
        if not texts:
            raise Abort("--lines needs subtitles to find the quotes in, and these files have none")
        picks = warn_little(pick_lines_file(args.lines, texts), name)
    if refs or picks or not required:
        return refs, picks, False
    need("--lines or --reference")
    ways = []
    if texts:
        ways.append(("lines", "Pick her lines   search the subtitles and tick lines she says"))
    if texts or shutil.which("ffplay"):
        ways.append(("groups", "Voice groups     the app groups the voices it hears, you pick hers"))
    ways.append(("clips", "Voice clips      audio/video files with only her voice"))
    how = ui.select(f"How should it learn {name}'s voice?", ways, ways[0][0]) if len(ways) > 1 else "clips"
    if how == "lines":
        return [], warn_little(search_lines(texts, name), name), False
    if how == "groups":
        return [], [], True
    return pick_reference(args, name, required=True), [], False


def span_text(ep: Episode, a: float, b: float) -> str:
    """What's said in a stretch of speech, from a non-Japanese subtitle track if there is one."""
    tracks = sorted(ep.texts, key=lambda t: t.stream.language == "jpn")
    span = Line(a, b, (), "")
    for t in tracks:
        hit = next((l.text for l in t.lines if quotes.same_moment(span, l)), None)
        if hit:
            return hit
    return ""


def cut_samples(clips: list[tuple[Episode, Stream, float, float]], out: Path) -> None:
    """Stretches of speech, one after another with short pauses, as one wav. Cut with ffmpeg, since
    ffplay's own seeking lands seconds early."""
    rate = 24000
    gap = b"\0" * 2 * (rate // 3)  # a third of a second between clips
    pcm = []
    for ep, audio, a, b in clips:
        pcm.append(subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{min(b - a, SAMPLE_MAX):.3f}",
                                   "-i", str(ep.info.path), "-map", f"0:a:{audio.kind_index}", "-f", "s16le",
                                   "-ac", "1", "-ar", str(rate), "-"], capture_output=True).stdout)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(gap.join(pcm))


PLAYER = ["ffplay", "-nodisp", "-autoexit", "-nostats", "-loglevel", "quiet"]


def play(clips: list[tuple[Episode, Stream, float, float]]) -> None:
    with tempfile.TemporaryDirectory(prefix="waifu-") as tmp:
        cut_samples(clips, Path(tmp) / "samples.wav")
        subprocess.run([*PLAYER, str(Path(tmp) / "samples.wav")], stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Preview:
    """Plays a clip in the background while a menu stays open; a new one stops the last."""

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="waifu-")
        self.proc: subprocess.Popen | None = None
        self.count = 0
        self.lock = threading.Lock()

    def start(self, clips: list[tuple[Episode, Stream, float, float]]) -> None:
        self.stop()
        self.count += 1
        threading.Thread(target=self._run, args=(clips, self.count), daemon=True).start()

    def _run(self, clips, n: int) -> None:
        out = Path(self.tmp.name) / f"preview{n}.wav"
        cut_samples(clips, out)
        with self.lock:
            if n == self.count:  # nothing newer was asked for while cutting
                self.proc = subprocess.Popen([*PLAYER, str(out)], stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = None

    def close(self) -> None:
        self.count += 1
        self.stop()
        self.tmp.cleanup()


def examples(spans: list, members, n: int) -> list[int]:
    """The group lines to show: its most typical ones, preferring lines long enough to say something
    ("Wow!" doesn't tell voices apart)."""
    pool = list(members[:TYPICAL])

    def short(j: int) -> bool:
        ep, _, a, b = spans[j]
        text = span_text(ep, a, b)
        return b - a < EXAMPLE_MIN or (bool(text) and len(text) < 15)

    return sorted(pool, key=short)[:n]  # stable sort: closeness order within each kind


def pick_groups(args, search: list[Episode], audio_pref: tuple[str | None, int], name: str):
    """Group every voice in the episodes, let the user pick the groups that are her (reading and listening),
    and return the most typical fingerprints of those groups."""
    import numpy as np

    spans, rows = [], []
    with ui.progress() as bar:
        task = bar.add_task("Loading the voice model", total=len(search))
        try:
            fp = voice.Fingerprinter()
        except OSError as exc:
            raise Abort(f"couldn't download the voice model: {exc}") from None

        def listen(ep: Episode):
            audio = resolve_audio(ep.info, *audio_pref)
            if audio is None:
                return ep, audio, None
            lines = ep.dialogue.lines if ep.dialogue else None
            return ep, audio, voice.episode_prints(fp, ep.info, audio,
                                                   None if lines is None else [(l.start, l.end) for l in lines])

        bar.update(task, description=f"[bold]Listening to {len(search)} episode(s)[/]")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for ep, audio, p in pool.map(listen, search):
                bar.advance(task)
                if p is None:
                    continue
                for a, b, e, ok in zip(p.starts, p.ends, p.emb, p.ok):
                    if ok:
                        spans.append((ep, audio, float(a), float(b)))
                        rows.append(e)
        if len(rows) < 20:
            raise Abort("too little speech in these episodes to group voices")
        bar.update(task, description="[bold]Grouping voices[/]")
        rows = np.array(rows)
        idx, labels = voice.voice_groups(rows)
        bar.update(task, description="[bold green]Grouped the voices[/]")

    groups = []
    for g in set(labels.tolist()):
        members = idx[labels == g]
        if len(members) >= MIN_GROUP:
            groups.append(members[voice.by_closeness(rows[members])])
    groups.sort(key=len, reverse=True)
    player = shutil.which("ffplay")
    shown = [examples(spans, m, SAMPLES) for m in groups]
    ui.step(f"Pick a group to {'hear and ' if player else ''}read its most typical lines, then say if it's {name}. "
            "One voice can be split over several groups: tick all of hers"
            + (". Tab plays a group's lines without opening it" if player else ""))

    def title(i: int) -> str:
        secs = sum(spans[j][3] - spans[j][2] for j in groups[i])
        return f"{'◉' if i in picked else '○'} Group {i + 1:<3} {len(groups[i])} lines, {fmt_dur(secs)}"

    width = max(20, ui.console.width - 12)
    lines = {i: [f"     “{text[:width]}”" if text else "     … (no subtitle)"
                 for text in (span_text(spans[j][0], spans[j][2], spans[j][3]) for j in shown[i])]
             for i in range(len(groups))}
    preview = Preview() if player else None
    keys = {"tab": lambda i: i != "done" and preview.start([spans[j] for j in shown[i]])} if preview else None
    picked: set[int] = set()
    try:
        while True:
            choice = ui.select(f"Which groups are {name}?", [(i, title(i)) for i in range(len(groups))]
                               + [("done", f"Done          use the {len(picked)} ticked group(s)")],
                               "done" if picked else 0, keys=keys, notes=lines,
                               instruction="↑↓ Enter · Tab plays its lines" if preview else "↑↓ Enter")
            if preview:
                preview.stop()
            if choice == "done":
                if picked:
                    break
                ui.warn(f"Tick at least one group first: pick one and answer yes if it's {name}")
                continue
            for j in shown[choice]:
                ep, audio, a, b = spans[j]
                ui.console.print(f"  [{ui.DIM}]{ep.label:<8} {fmt_clock(a)}[/]  {span_text(ep, a, b) or '…'}")
            if player:
                ui.console.print(f"  [{ui.DIM}]playing {len(shown[choice])} of them…[/]")
                play([spans[j] for j in shown[choice]])
            if ui.confirm(f"Is this {name}?", choice in picked or not picked):
                picked.add(choice)
            else:
                picked.discard(choice)
    finally:
        if preview:
            preview.close()
    return np.concatenate([voice.group_core(rows[groups[i]]) for i in sorted(picked)])


def warn_little(picks: list[tuple[Episode, Line]], name: str) -> list[tuple[Episode, Line]]:
    total = sum(line.end - line.start for _, line in picks)
    if total < MIN_PICKED:
        ui.warn(f"Only {total:.0f} s of {name}'s voice: matching works better with 30 s or more")
    return picks


def spread(items: list, n: int) -> list:
    """Up to n items, evenly spaced through the list."""
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def pick_ignore_names(args, named: list[Episode], name: str) -> bool:
    """Find her by voice in every episode, even where the subs name her?"""
    if not named:
        return False
    reason = voice.missing()
    if reason:
        if args.ignore_names:
            raise Abort(f"--ignore-names needs voice matching: {reason}")
        return False
    if args.ignore_names is not None:
        return args.ignore_names
    if not ui.interactive():
        return False
    return ui.confirm(f"Ignore the subtitle names and find {name} by her voice in every episode?", False)


def her_lines(picks: list[tuple[Episode, Line]]) -> dict[int, list[Line]]:
    """Picked lines per episode, one per moment (the same line can be picked from two subtitle tracks).
    Lines that run over someone else's are kept out: they'd teach that voice too."""
    out: dict[int, list[Line]] = {}
    for ep, line in sorted(picks, key=lambda x: x[1].start):
        mine = out.setdefault(id(ep), [])
        if not any(quotes.same_moment(line, l) for l in mine):
            mine.append(line)
    clean = {}
    for ep in {id(ep): ep for ep, _ in picks}.values():
        talked_over = {l for t in ep.texts for l, o in zip(t.lines, voice.overlapping(t.lines)) if o}
        clean[id(ep)] = [l for l in out[id(ep)] if l not in talked_over]
    return clean


def match_voice(args, named: list[Episode], search: list[Episode], keys: set[str], refs: list[Path],
                picks: list[tuple[Episode, Line]], grouped, name: str,
                audio_pref: tuple[str | None, int]) -> list[tuple[Episode, list[Segment]]]:
    """Learn her voice from named episodes, reference clips, lines picked as hers and/or fingerprints of
    the voice groups picked as hers (`grouped`), then find it in the `search` episodes."""
    import numpy as np

    teachers = spread(named, voice.MAX_TRAIN_EPISODES)
    with ui.progress() as bar:
        picked = her_lines(picks)
        picked_eps = [ep for ep in {id(ep): ep for ep, _ in picks}.values() if picked[id(ep)]]
        task = bar.add_task(f"Learning {name}'s voice",
                            total=len(teachers) + len(refs) + len(picked_eps) + len(search))

        def status(msg: str) -> None:
            bar.update(task, description=f"[bold]{msg}[/]")

        status("Loading the voice model")
        try:
            fp = voice.Fingerprinter()
        except OSError as exc:
            raise Abort(f"couldn't download the voice model: {exc}") from None

        def prints(ep: Episode, lines: list[Line] | None):
            audio = resolve_audio(ep.info, *audio_pref)
            if audio is None:
                return None
            return voice.episode_prints(fp, ep.info, audio, None if lines is None else [(l.start, l.end) for l in lines])

        # Her lines and everyone else's from episodes that name the speakers
        pos, neg = [] if grouped is None else list(grouped), []

        def learn(ep: Episode):
            overlap = dict(zip(map(id, ep.lines), voice.overlapping(ep.lines)))
            lines = [l for l in ep.lines if len(l.speakers) == 1 and l.end - l.start >= voice.MIN_LEN
                     and not overlap[id(l)]]
            return lines, prints(ep, lines)

        status(f"Learning {name}'s voice from {len(teachers)} named episode(s)" if teachers
               else f"Learning {name}'s voice from her lines" if picked_eps
               else f"Learning {name}'s voice from her groups" if grouped is not None else "Reading reference clips")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for lines, p in pool.map(learn, teachers):
                bar.advance(task)
                if p is None:
                    continue
                for line, e, ok in zip(lines, p.emb, p.ok):
                    if not ok:
                        continue
                    if line.speakers[0].casefold() in keys:
                        if 1.5 <= line.end - line.start <= 12:
                            pos.append(e)
                    else:
                        neg.append(e)
            for ref in refs:
                status(f"Reading reference clip {ref.name}")
                try:
                    pos += list(voice.reference_prints(fp, ref))
                except subprocess.CalledProcessError:
                    ui.warn(f"{ref.name}: couldn't read its audio, skipped")
                bar.advance(task)
            for p in pool.map(lambda ep: prints(ep, picked[id(ep)]), picked_eps):
                bar.advance(task)
                if p is not None:
                    pos += list(p.emb[p.ok])
            if not pos:
                raise Abort(f"nothing of {name}'s voice to learn from"
                            + (" (reference clips had no speech)" if refs else "")
                            + (" (her picked lines were too short or talked over)" if picks else ""))

            # Where people talk: dialogue sub timing, or voice detection without subs
            status(f"Listening to {len(search)} episode(s)")

            def listen(ep: Episode):
                lines = ep.dialogue.lines if ep.dialogue else None
                return ep, lines, prints(ep, lines)

            found = []
            for ep, lines, p in pool.map(listen, search):
                bar.advance(task)
                if p is None:
                    ui.warn(f"{ep.label}: no audio track, skipped")
                    continue
                found.append((ep, lines, p))

        if not found:
            return []
        status(f"Finding {name}")
        pool_rows = np.concatenate([p.emb[p.ok] for _, _, p in found])
        score = voice.train(np.array(pos), np.array(neg) if neg else None, pool_rows)
        # Groups teach from the episodes' own audio, like named lines, and score as high
        sure_at = args.sure if args.sure is not None else VOICE_SURE if neg or grouped is not None else REF_SURE
        check_at = args.check if args.check is not None else VOICE_CHECK if neg else REF_CHECK
        plan = []
        for ep, lines, p in found:
            scores = np.full(len(p.starts), np.nan)
            if p.ok.any():
                scores[p.ok] = score(p.emb[p.ok])
            # Lines picked as hers are hers, whatever the score
            for i, (a, b) in enumerate(zip(p.starts, p.ends)):
                if any(quotes.same_moment(Line(a, b, (), ""), l) for l in picked.get(id(ep), [])):
                    scores[i] = 1.0
            texts = [l.text for l in lines] if lines else [""] * len(p.starts)
            unsure = voice.overlapping(lines) if lines else [False] * len(p.starts)
            segs = voice_segments(list(zip(p.starts, p.ends, texts)), list(scores), unsure, pad=args.pad,
                                  merge_gap=args.merge_gap, sure=sure_at, low=check_at)
            segs = [s for s in segs if s.duration >= args.min_length]
            sure = sum(not s.check for s in segs)
            how = "sub timing" if lines else "voice detection"
            bar.console.print(f"[green]✓[/] {ep.label:<10} [{ui.DIM}]{sure} sure, {len(segs) - sure} to check · {how}[/]")
            if segs:
                plan.append((ep, segs))
        bar.update(task, description=f"[bold green]Learned {name}'s voice[/] [{ui.DIM}]"
                   f"{len(pos)} clip(s) of her, {len(neg)} of others[/]")
    return plan


def pick_by_amount(plan: list[tuple[Episode, list[Segment]]], amount: float) -> list[tuple[Episode, list[Segment]]]:
    """pick_for_cloning, taking sure clips first and topping up from the ones to check."""
    order = {id(ep): i for i, (ep, _) in enumerate(plan)}
    sure = [(ep, [s for s in segs if not s.check]) for ep, segs in plan]
    picked = pick_for_cloning([(ep, segs) for ep, segs in sure if segs], amount)
    got = sum(s.duration for _, segs in picked for s in segs)
    if got < amount:
        unsure = [(ep, [s for s in segs if s.check]) for ep, segs in plan]
        picked += pick_for_cloning([(ep, segs) for ep, segs in unsure if segs], amount - got)
    joined: dict[int, tuple[Episode, list[Segment]]] = {}
    for ep, segs in picked:
        joined.setdefault(id(ep), (ep, []))[1].extend(segs)
    return [(ep, sorted(segs, key=lambda s: s.start)) for ep, segs in sorted(joined.values(), key=lambda x: order[id(x[0])])]


def unique_dir(base: Path) -> Path:
    path, n = base, 2
    while path.exists() and any(path.iterdir()):
        path = base.with_name(f"{base.name}_{n}")
        n += 1
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- extraction ------------------------------------------------------------

@dataclass
class Version:
    """One copy of the output: the clips after some of the effects."""
    folder: str | None  # subfolder, or None for the only copy
    source: int  # decoded wav to cut from: 0 = unedited, then one per separation model; -1 = the last
    normalize: bool


def versions(effects: set[str], every_step: bool) -> list[Version]:
    picked = [e for e in fx.ORDER if e in effects]
    if not every_step or not picked:
        return [Version(None, -1, "normalize" in effects)]
    seps = [e for e in picked if e in fx.SEPARATIONS]
    out = [Version("1_unedited", 0, False)]
    for n, e in enumerate(picked, 2):
        folder = f"{n}_finished" if e == picked[-1] else f"{n}_{fx.STEP_NAMES[e]}"
        out.append(Version(folder, seps.index(e) + 1 if e in seps else len(seps), e == "normalize"))
    return out


class Output:
    """Where one version goes: its clips (or strip), check/ clips and metadata.tsv."""

    def __init__(self, root: Path, work: Path, name: str, ext: str, output: str, by_voice: bool, gap: float):
        root.mkdir(exist_ok=True)
        work.mkdir(exist_ok=True)
        self.root, self.output, self.gap = root, output, gap
        self.clips_dir = root if output == "clips" else work / "parts"
        self.clips_dir.mkdir(exist_ok=True)
        # Voice-matched clips that might be someone else: clips go to check/, strip lines are held back
        # and added at the end, most likely her first.
        self.check_dir = root / "check" if output == "clips" else work / "later"
        self.strip_path = root / f"{safe_name(name)}.{ext}"
        self.strip_wav: wave.Wave_write | None = None
        self.strip_pos = 0.0
        self.parts: list[Path] = []
        self.later: list[tuple[float, Path, list[str], list[str]]] = []
        self.meta = open(root / "metadata.tsv", "w", newline="")
        self.writer = csv.writer(self.meta, delimiter="\t")
        self.writer.writerow(["file", "episode", "source", "start", "end", "duration"]
                             + (["strip_start", "strip_end"] if output == "strip" else [])
                             + (["match", "score"] if by_voice else []) + ["text"])

    def dest(self, clip: str, check: bool) -> Path:
        path = (self.check_dir if check else self.clips_dir) / clip
        path.parent.mkdir(exist_ok=True)
        return path

    def add_to_strip(self, frames: bytes, params: wave._wave_params) -> None:
        if self.strip_wav is None:
            self.strip_wav = wave.open(str(self.strip_path), "wb")
            self.strip_wav.setparams(params)
        gap = int(self.gap * params.framerate) * params.nchannels * params.sampwidth
        self.strip_wav.writeframes(frames + b"\0" * gap)

    def row(self, dest: Path, head: list[str], tail: list[str], length: float | None) -> None:
        """`length` is the clip's length in the strip, None for clips."""
        row = [str(dest.relative_to(self.root)) if self.output == "clips" else self.strip_path.name, *head]
        if length is not None:
            row += [f"{self.strip_pos:.3f}", f"{self.strip_pos + length:.3f}"]
            self.strip_pos += length + (self.gap if self.strip_path.suffix == ".wav" else 0)
        self.writer.writerow(row + tail)

    def finish(self, status) -> None:
        """Unsure strip lines at the end, most likely first; then close the strip."""
        if self.later:
            status("strip", f"adding {len(self.later)} line(s) to check at the end")
        for _, path, head, tail in sorted(self.later, key=lambda x: -x[0]):
            if path.suffix == ".wav":
                with wave.open(str(path), "rb") as w:
                    self.add_to_strip(w.readframes(w.getnframes()), w.getparams())
                    length = w.getnframes() / w.getframerate()
            else:
                self.parts.append(path)
                length = cutter.duration(path)
            self.row(path, head, tail, length)
        if self.strip_wav:
            self.strip_wav.close()
        if self.strip_path.suffix == ".mkv" and self.output == "strip" and self.parts:
            status("strip", "joining clips into one video")
            cutter.concat_videos(self.parts, self.strip_path)

    def close(self) -> None:
        self.meta.close()
        if self.strip_wav:
            self.strip_wav.close()


def extract(args, plan: list[tuple[Episode, list[Segment]]], name: str, mode: str, output: str,
            accurate: bool, effects: set[str], audio_pref: tuple[str | None, int]) -> Path:
    outdir = unique_dir(args.out / f"{safe_name(name)}_{mode}_{output}")
    work = Path(tempfile.mkdtemp(prefix=".work-", dir=outdir))
    ext = "wav" if mode == "audio" else "mkv"
    by_voice = any(s.score is not None for _, segs in plan for s in segs)
    every = versions(effects, args.every_step)
    outs = [Output(outdir / v.folder if v.folder else outdir, work / f"v{i}", name, ext, output, by_voice, args.gap)
            for i, v in enumerate(every)]
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

                # Decoded episode audio: unedited, then after each separation model.
                wavs: list[Path] = []
                if mode == "audio" or effects:
                    status(ep.label, f"decoding {audio.language or 'audio'} track")
                    if len(segs) <= SEEK_MAX_CLIPS:
                        wavs.append(cutter.decode_spans(ep.info, audio, segs, work / f"{ep.label}.wav", args.mono))
                    else:
                        wavs.append(cutter.decode_audio(ep.info, audio, work / f"{ep.label}.wav", args.mono))
                    if effects & fx.SEPARATIONS:
                        wavs += fx.process_excerpts(wavs[0], segs, audio.start_time, effects, work, args.mono,
                                                    lambda m: status(ep.label, m), every_step=args.every_step)
                        if not args.every_step:
                            wavs.pop(0).unlink()

                status(ep.label, "cutting")
                srcs = {v.source: cutter.WavSource(wavs[v.source], audio.start_time) for v in every} if wavs else {}
                for n, seg in enumerate(segs, 1):
                    clip = f"{ep.label}_{n:04d}_{fmt_ts(seg.start)}.{ext}"
                    hold = output == "strip" and seg.check
                    head = [ep.label, ep.info.path.name, f"{seg.start:.3f}", f"{seg.end:.3f}", f"{seg.duration:.3f}"]
                    tail = [seg.text]
                    if by_voice:
                        verdict = "name" if seg.score is None else "check" if seg.check else "voice"
                        tail = [verdict, "" if seg.score is None else f"{seg.score:.2f}"] + tail
                    # The finished version first: in video mode the others reuse its picture.
                    cut = None
                    for v, out in reversed(list(zip(every, outs))):
                        dest = out.dest(clip, seg.check)
                        src = srcs.get(v.source)
                        frames = b""
                        if src:
                            frames = src.read(seg.start, seg.end)
                            if v.normalize:
                                frames = fx.normalize_frames(frames, src.params, work, args.loudness)
                        if mode == "audio":
                            if output == "clips" or hold:
                                cutter.write_wav(dest, src.params, frames)
                            else:
                                out.add_to_strip(frames, src.params)
                        else:
                            # With effects, the clip's processed audio goes in as its own wav starting at seg.start.
                            clip_audio = None
                            if src:
                                clip_audio = work / "clip-audio.wav"
                                cutter.write_wav(clip_audio, src.params, frames)
                            if cut is None:
                                cutter.cut_video(ep.info, seg, dest, video=video, audio=audio,
                                                 accurate=accurate, encoder=args.encoder,
                                                 audio_file=clip_audio, audio_offset=seg.start)
                                cut = dest
                            else:
                                cutter.swap_audio(cut, clip_audio, dest)
                            if not hold:
                                out.parts.append(dest)
                        if hold:
                            out.later.append((seg.score, dest, head, tail))
                        elif output == "strip":
                            # Fast video cuts start on a keyframe, so the clip can be longer than the line.
                            out.row(dest, head, tail, seg.duration if mode == "audio" else cutter.duration(dest))
                        else:
                            out.row(dest, head, tail, None)
                    bar.advance(task)
                for src in srcs.values():
                    src.close()
                for wav in wavs:
                    wav.unlink(missing_ok=True)
                bar.console.print(f"[green]✓[/] {ep.label:<10} [{ui.DIM}]{len(segs)} clip(s)[/]")

            for out in outs:
                out.finish(status)
            bar.update(task, description="[bold green]Done[/]")
    finally:
        for out in outs:
            out.close()
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
    p.add_argument("--every-step", action=argparse.BooleanOptionalAction, default=None,
                   help="also save the clips unedited and after each effect, each version in its own "
                        "numbered folder (1_unedited, 2_no-music, ... the last is the finished one)")
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
    p.add_argument("--voice", action=argparse.BooleanOptionalAction, default=None,
                   help="find the character by voice in episodes whose subs don't name the speakers "
                        "(needs the [voice] extra)")
    p.add_argument("--ignore-names", action=argparse.BooleanOptionalAction, default=None,
                   help="find the character by voice in every episode, also where the subs name her "
                        "(the names still teach the app her voice)")
    p.add_argument("-r", "--reference", type=Path, action="append",
                   help="audio/video clip, or folder of clips, of only the character's voice; repeatable. "
                        "Needed when no episode names her, unless --lines finds her in the subs")
    p.add_argument("-L", "--lines", type=Path,
                   help="text file of things the character says, one per row, in the words of one of the "
                        "subtitle tracks; the matching lines teach the app her voice (in place of --reference)")
    p.add_argument("--sure", type=float,
                   help=f"voice-match score that counts as her (default {VOICE_SURE:g} when learned from "
                        f"named episodes, {REF_SURE:g} from reference clips alone)")
    p.add_argument("--check", type=float,
                   help=f"lowest voice-match score kept, for listening (default {VOICE_CHECK:g} / {REF_CHECK:g})")
    p.add_argument("-a", "--amount", help="how much audio to take, e.g. 90s, 10m, 1h30m or all (default: all). "
                   "Picks clone-friendly 3-12 s clips spread across episodes")
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
    unnamed = [e for e in episodes if not e.track]
    if not named and (voice.missing() or args.voice is False):
        hint = f"; to find a character by voice instead, {voice.missing()}" if voice.missing() else ""
        raise Abort("no subtitle track in these files names its speakers, so there is nothing to cut by" + hint)
    if args.list:
        ui.console.print()
        if named:
            report_catalog(name_catalog({e.label: e.lines for e in named}))
        else:
            ui.step("No names to list. Find a character by voice: run without --list, or with -c NAME -L HER_QUOTES.txt")
        return
    ui.console.print()

    picked = pick_character(args, name_catalog({e.label: e.lines for e in named})) if named else []
    if picked:
        keys = {p.display.casefold() for p in picked}
        name = max(picked, key=lambda p: (len(p.display), p.solo)).display
    else:
        if named:
            ui.step("Leaving the subtitle names out: the character is found by her voice in every episode")
            named, unnamed = [], episodes
        else:
            ui.step("No subtitles name the speakers, so the character is found by her voice")
        keys = set()
        name = pick_name(args)

    ignore_names = pick_ignore_names(args, named, name)
    plan = []
    for ep in [] if ignore_names else named:
        segs = build_segments(ep.lines, keys, pad=args.pad, merge_gap=args.merge_gap,
                              include_shared=args.include_shared, keep_overlap=args.keep_overlap)
        segs = [s for s in segs if s.duration >= args.min_length]
        if segs:
            plan.append((ep, segs))

    audio_pref = None
    by_voice = ignore_names or pick_voice(args, named, unnamed, name)
    if by_voice:
        refs, picks, groups = pick_teacher(args, episodes, name, required=not named)
        # The voice is matched on the track that gets cut, so ask for it now rather than at the end.
        audio_pref = pick_audio(args, (named or unnamed)[0].info)
        search = episodes if ignore_names else unnamed
        grouped = pick_groups(args, search, audio_pref, name) if groups else None
        order = {id(ep): i for i, ep in enumerate(episodes)}
        plan += match_voice(args, named, search, keys, refs, picks, grouped, name,
                            audio_pref)
        plan.sort(key=lambda x: order[id(x[0])])

    available = sum(seg.duration for _, segs in plan for seg in segs)
    if not plan:
        raise Abort(f"no usable lines for {name}" + ("" if by_voice and not named
                                                    else " (try --keep-overlap or --include-shared)"))
    amount = pick_amount(args, available)
    if amount is not None and amount < available:
        plan = pick_by_amount(plan, amount)

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
    audio_pref = audio_pref or pick_audio(args, (named or unnamed)[0].info)

    if args.limit:
        left, trimmed = args.limit, []
        for ep, segs in plan:
            if left <= 0:
                break
            trimmed.append((ep, segs[:left]))
            left -= len(trimmed[-1][1])
        plan = trimmed

    count = sum(len(s) for _, s in plan)
    seconds = sum(seg.duration for _, segs in plan for seg in segs)
    matched = [s for _, segs in plan for s in segs if s.score is not None]
    unsure = sum(s.check for s in matched)
    where = "at the end of the strip, most likely first" if output == "strip" else "in check/"
    voice_row = [("By voice", f"{len(matched) - unsure} sure · {unsure} to listen to [{ui.DIM}]({where})[/]")] \
        if by_voice else []
    folders = [v.folder for v in versions(effects, args.every_step) if v.folder]
    steps_row = [("Versions", f"{len(folders)} [{ui.DIM}]({', '.join(folders)})[/]")] if folders else []
    lang, idx = audio_pref
    ui.console.print()
    ui.panel([
        ("Character", f"[bold {ui.PINK}]{name}[/]" + (f" [{ui.DIM}]({', '.join(p.display for p in picked)})[/]"
                                                     if len(picked) > 1 else "")),
        ("Lines", f"{count} clips · {fmt_dur(seconds)} · {len(plan)} episode(s)"
                  + (f" [{ui.DIM}]picked from {fmt_dur(available)}, 3-12 s clips first[/]" if seconds < available - 0.5 else "")),
        *voice_row,
        ("Output", ("audio .wav" if mode == "audio" else "video .mkv") + (", one strip" if output == "strip" else ", clips")
         + ("" if mode == "audio" else (", accurate cuts" if accurate else ", fast cuts"))),
        ("Effects", ", ".join(fx.EFFECTS[e] for e in fx.ORDER if e in effects) or f"[{ui.DIM}]none[/]"),
        *steps_row,
        ("Audio track", lang or f"track {idx}"),
        ("Save to", str(args.out.resolve() / f"{safe_name(name)}_{mode}_{output}")),
    ], "Ready")
    if not args.yes and ui.interactive() and not ui.confirm("Start?", True):
        raise ui.Cancelled

    outdir = extract(args, plan, name, mode, output, accurate, effects, audio_pref)
    ui.console.print()
    ui.panel([("Clips", f"{count} · {fmt_dur(seconds)}"), *voice_row, *steps_row, ("Folder", str(outdir.resolve())),
              ("Transcripts", "metadata.tsv")], "Done", color="green")
    if unsure:
        ui.step(f"Listen to the {unsure} unsure clip(s) {where} and delete the ones that aren't {name}")
    if ui.interactive() and not args.yes and shutil.which("xdg-open") and ui.confirm("Open the folder?", False):
        subprocess.Popen(["xdg-open", str(outdir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
