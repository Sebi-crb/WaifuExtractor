# WaifuExtractor

Cut one character's voice or scenes out of anime episodes.

Many fansub `.ass` tracks fill in the *Name* (actor) field on every dialogue line.
WaifuExtractor scans every text subtitle track in your files, finds the one that names
its speakers, lists the characters, and cuts out the lines of the one you pick — as a
folder of clips or one long strip, audio-only (`.wav`) or video (`.mkv`).

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

## Usage

```sh
waifu-extract "/path/to/Season 1"          # interactive: asks for everything
waifu-extract "/path/to/show" --list       # just show who speaks how much
waifu-extract ep01.mkv -c Aqua -m audio -O clips -y
waifu-extract "/path/to/show" -c Kaz,Kazuma -m video -O strip --fast
```

Interactive mode asks, in order: character (several spelling variants can be picked at
once, and it offers similar names like *Kaz* / *Kazuma*), audio or video, clips or strip,
fast or accurate video cuts, audio effects and the audio track (Japanese by default).

The effects menu is a checklist of three; effects that aren't installed are marked:

- **Remove music and sound effects** (`--remove-music`)
- **Remove reverb and echo** (`--dereverb`)
- **Normalize loudness** (`--normalize`, target `--loudness -20` LUFS): two-pass loudnorm
  per clip, so every clip has the same volume.

Effects also apply to video output: the clip gets the processed soundtrack. Each question has a matching flag, see `--help`.

Output lands in `./waifu-extract/<Character>_<audio|video>_<clips|strip>/` together with
`metadata.tsv`: source episode, timestamps and the subtitle text of every clip, which
doubles as transcripts for voice-cloning datasets.

### How lines are chosen

- Per episode, the text subtitle track with the most speaker-named lines is used.
  Sign, song and karaoke events are ignored.
- Lines credited to several speakers (`Aqua & Kazuma`) are skipped unless `--include-shared`.
- Lines that overlap another speaker by more than 0.15 s are skipped unless `--keep-overlap`.
- Each line is padded by `--pad` (0.1 s) and lines closer than `--merge-gap` (0.3 s) are joined.

### Video cuts

- `--accurate` (default) re-encodes with x264, or `--encoder vaapi` for AMD/Intel GPUs.
  Cuts land exactly on the line.
- `--fast` stream-copies: instant, but each clip starts on the previous keyframe.

### Speed

The first scan of a file reads it completely (subtitles are interleaved with the video),
which is slow over network shares. Extracted subtitles are cached in
`~/.cache/waifu-extractor`, so later runs on the same files start instantly.

## Limits

- Only works when a subtitle track names its speakers. Image subtitles (PGS/VobSub) and
  most official/streaming subtitles don't.
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
