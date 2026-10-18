# WaifuExtractor

Cut one character's voice or scenes out of anime episodes.

Many fansub `.ass` tracks fill in the *Name* (actor) field on every dialogue line.
WaifuExtractor scans every text subtitle track in your files, finds the one that names
its speakers, lists the characters, and cuts out the lines of the one you pick — as a
folder of clips or one long strip, audio-only (`.wav`) or video (`.mkv`).

Episodes whose subtitles don't name anyone, or that have no subtitles at all, can be searched
by **voice** instead: the app learns her voice from episodes that do name her, or from a few
clips of her you give it, and marks every clip it isn't sure about for you to listen to.

## Install

Needs `ffmpeg`/`ffprobe` on `PATH` and Python 3.11+.

```sh
uv tool install git+https://github.com/Sebi-crb/WaifuExtractor
# or, from a clone:
uv sync && uv run waifu-extract --help
```

### Optional: music removal and de-reverb

These run AI separation models through [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
on each episode before cutting: BS-Roformer to remove music and sound effects, and a MelBand-Roformer
to remove reverb and echo. They need PyTorch, best installed for your GPU first:

```sh
uv tool install --editable . --with 'waifu-extractor[fx]'
# then put the GPU build of PyTorch into the same environment, e.g. AMD (ROCm):
uv pip install --python "$(uv tool dir)/waifu-extractor/bin/python" torch --index-url https://download.pytorch.org/whl/rocm7.2
# or NVIDIA: --index-url https://download.pytorch.org/whl/cu128
# Python 3.14 only: audio-separator pins beartype<0.19, which can't load the Roformer models there
uv pip install --python "$(uv tool dir)/waifu-extractor/bin/python" 'beartype>=0.22'
```

On CPU they work but take several minutes per episode. Models download to
`~/.cache/waifu-extractor/models` on first use. On RDNA2 cards other than the 6800/6900
(e.g. RX 6600/6650/6700) the app sets `HSA_OVERRIDE_GFX_VERSION=10.3.0` automatically.

### Optional: voice matching

Finds a character by her voice where no subtitles name the speakers (WeSpeaker voice
fingerprints and Silero speech detection, both on the CPU, no account or token needed):

```sh
uv tool install --editable . --with 'waifu-extractor[voice]'
# if the tool already has the GPU PyTorch from the fx steps, add just the missing piece instead,
# so torch isn't replaced:
uv pip install --python "$(uv tool dir)/waifu-extractor/bin/python" --no-deps silero-vad
```

The voice model (26 MB) downloads to `~/.cache/waifu-extractor/models` on first use.

## Usage

```sh
waifu-extract "/path/to/Season 1"          # interactive: asks for everything
waifu-extract "/path/to/show" --list       # just show who speaks how much
waifu-extract ep01.mkv -c Aqua -m audio -O clips -y
waifu-extract "/path/to/show" -c Kaz,Kazuma -m video -O strip --fast
waifu-extract "/path/to/show" -c Megumin,Megu -a 90s -m audio -O clips --normalize -y
```

Run it with no options (or just a path) for the interactive mode: arrow-key menus,
checkboxes, and a fuzzy-search list for characters. It asks, in order:

1. **Character**: type to search, Tab to tick several (spelling variants like *Kaz* +
   *Kazuma*). Close spellings are offered afterwards, already ticked.
2. **How much**: everything, *30 min* (fine-tuning), *90 s* (zero-shot cloning) or a custom
   amount (`45s`, `10m`, `1h30m`, `1:30`).
3. **Audio or video**, then **clips or one strip**.
4. **Accurate or fast** video cuts (video only).
5. **Audio effects** (checklist, see below).
6. **Audio track** (Japanese preselected).

A summary panel shows what will happen before anything starts; progress bars show each
episode's current step. Every question has a matching flag (`--help`), and with
`-y` plus flags it runs without asking anything, e.g. in scripts.

The effects checklist has three options; ones that aren't installed are listed greyed out:

- **Remove music and sound effects** (`--remove-music`)
- **Remove reverb and echo** (`--dereverb`)
- **Normalize loudness** (`--normalize`, target `--loudness -20` LUFS): two-pass loudnorm
  per clip, so every clip has the same volume.

Effects also apply to video output: the clip gets the processed soundtrack.

To hear what each effect does, answer yes to "Also save a copy before the effects and after
each one?" (`--every-step`). Each version gets its own numbered folder, with its own `check/`
and `metadata.tsv`. With all three effects:

```
Aqua_audio_clips/
  1_unedited/     straight from the episode
  2_no-music/     music and sound effects removed
  3_no-reverb/    2 + reverb and echo removed
  4_finished/     3 + normalized
```

The models still run only once per episode, so the extra copies cost disk space, not time.
In video mode the picture is cut once and reused for every version.

Output lands in `./waifu-extract/<Character>_<audio|video>_<clips|strip>/` together with
`metadata.tsv`: source episode, timestamps and the subtitle text of every clip, which
doubles as transcripts for voice-cloning datasets.

### How lines are chosen

- Per episode, the text subtitle track with the most speaker-named lines is used.
  Sign, song and karaoke events are ignored.
- Lines credited to several speakers (`Aqua & Kazuma`) are skipped unless `--include-shared`.
- Lines that overlap another speaker by more than 0.15 s are skipped unless `--keep-overlap`.
- Each line is padded by `--pad` (0.1 s) and lines closer than `--merge-gap` (0.3 s) are joined.

### Finding her by voice

Used for episodes where no subtitle track names the speakers (asked, or `--voice`):

```sh
waifu-extract "/path/to/show"   -c Aqua --voice -m audio -O clips -y     # some episodes name her
waifu-extract "/path/to/movie"  -c Aqua -L aqua_lines.txt -m audio -O clips # nothing names her
waifu-extract "/path/to/movie"  -c Aqua -r aqua_clips/ -m audio -O clips     # no subs at all
```

If she isn't in the character list, or the names are wrong, pick **None of these** at the top:
every episode is then searched by voice. A track only counts as naming its speakers when at least
30 % of its lines have a name; fansub notes in the name field (`TLC`, `edit`, `hon?`, `QC`...)
don't count as names.

After you pick her, it also asks whether to **ignore the names** and search every episode by
voice (`--ignore-names`); the names then only teach it her voice.

1. **Learning her voice.** From up to 6 episodes that do name her, her lines are compared
   with everyone else's. If nothing names her, it asks how to learn her voice:
   - **Pick her lines** (needs any subtitles): type a few words she says, it lists every subtitle
     line containing them (episode, time, length, text), you tick hers. Repeat until you have
     30 s or so, then Enter on an empty search. Any subtitle track can be searched, also Japanese.
     For scripts, `-L/--lines FILE` takes a text file of her quotes, one per row, e.g. from a wiki;
     they must use the wording of one of the subtitle tracks (punctuation and case don't matter,
     small differences are fine). Short quotes like "What?" that turn up in several places are
     skipped, as others probably say them too. Picked lines always count as hers in the output.
   - **Voice groups** (no input needed, interactive only): it sorts all the speech into about 15
     groups by voice. Each group shows 3 of its most typical lines (preferring ones of 2 s or more);
     Tab plays them without leaving the menu, Enter plays them and asks whether it's her. One voice is often split over two or three groups, so tick all of hers. Works with no
     subtitles at all (listening only). A bit less exact than picking lines for voices that sound
     alike: on KonoSuba, Megumin's sure clips were 3 in 19 someone else in one episode.
   - **Voice clips**: `-r/--reference` gives clips of only her voice (files or a folder, audio or
     video; 30–90 s is plenty, e.g. cut from another release with this tool).

   Then the episodes' own speech stands in for "everyone else".
2. **Where people talk.** The fullest dialogue subtitle track, named or not, gives the timing
   (any language; the text goes to `metadata.tsv`). Without one, speech detection finds it.
3. **Scoring.** Each line gets a 0–1 score. Clips at or above `--sure` go in the folder as usual;
   between `--check` and `--sure` they are **unsure** and go to `check/` (clips) or the end of
   the strip, most likely her first. Listen to those and delete the ones that aren't her.
   Lines overlapping another line also count as unsure. `metadata.tsv` gets `match` and `score`
   columns.

Defaults, measured on KonoSuba S1 with the names hidden (Aqua, Megumin, Darkness, Kazuma):

| Learned from | `--sure` | sure clips that were hers | `--check` | unsure clips that were hers |
|---|---|---|---|---|
| named episodes | 0.8 | 87–96 %, none clearly someone else | 0.5 | about 1 in 4 |
| reference clips only | 0.7 | 88–100 % | 0.3 | about 1 in 3 or more |
| voice groups (ticked where most shown lines were hers) | 0.8 | 68–97 %, 0–3 someone else per episode | 0.3 | about 1 in 3 |
| 12 picked lines only | 0.7 | 77–83 % in episodes not picked from, none clearly someone else | 0.3 | about 1 in 4 to 2 in 3 |

Most of the rest is clips where she talks together with someone else. Similar voices
(Aqua/Megumin) are the hardest; a male voice among female ones is easy. Fingerprints are cached
in `~/.cache/waifu-extractor/voice`, so trying another character on the same show is quick.
Music removal doesn't change the matching, so it runs only on the clips that get cut.

### Choosing an amount

With `--amount` (or the *How much?* question), clips are picked for voice cloning rather than
taken from the start: 3–12 s clips first (long enough to carry the voice, short enough to be
one utterance), then 1.5–20 s, then the rest. Within each group it takes turns across episodes
and spreads picks through each episode, for varied emotion and recording conditions. The pick
is deterministic, so the same settings always give the same clips. Sure voice matches are
picked before unsure ones. `--limit N` still exists
for a quick preview of the first N clips.

### Video cuts

- `--accurate` (default) re-encodes with x264, or `--encoder vaapi` for AMD/Intel GPUs.
  Cuts land exactly on the line.
- `--fast` stream-copies: instant, but each clip starts on the previous keyframe.

### Speed

The first scan of a file reads it completely (subtitles are interleaved with the video),
which is slow over network shares. Extracted subtitles are cached in
`~/.cache/waifu-extractor`, so later runs on the same files start instantly.

For audio, episodes with up to 30 clips are cut by seeking straight to each clip instead of
decoding the whole track (90 s of Megumin across 19 episodes: 15 s instead of ~8 min).

## Limits

- Without speaker names it relies on voice matching (above): good, not perfect, so listen to
  the unsure clips. Image subtitles (PGS/VobSub) give no timing; speech detection is used there.
- A singer who is also the voice actress can match in songs; song and karaoke subtitle lines are
  skipped, but with no subtitles at all an insert song could slip in.
- Subtitle timing is made for reading, so clips can include a little silence or a
  neighbouring sound; `--remove-music` and `--min-length` help for voice datasets.
- Timings match the audio the subtitles were made for (usually Japanese); the dub can drift.

## Known issues

- **beartype pin (Python 3.14).** `audio-separator` 0.47 requires `beartype<0.19`, but that version
  can't load the Roformer models on Python 3.14 (`type hint collections.abc.Callable | None either
  PEP-noncompliant or currently unsupported by @beartype`). Install `beartype>=0.22` over the pin
  (see the fx install steps). `uv tool upgrade` or a reinstall puts the old version back, so redo
  that step afterwards. Python 3.13 and older aren't affected.
- **Undeclared dependencies.** `audio-separator` imports `onnxruntime` and `audioread` without
  listing them; the `[fx]` extra adds both.
- **Separation speed.** The models run at roughly real time on a mid-range GPU (RX 6650 XT), so
  they only process the parts of each episode that get cut, plus 1 s on each side.

Only use it on media you own.
