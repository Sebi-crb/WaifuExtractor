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

### Optional: vocal isolation

`--isolate-vocals` runs [Demucs](https://github.com/adefossez/demucs) on each episode to
strip music and sound effects, leaving just voices. It pulls in PyTorch:

```sh
uv tool install 'waifu-extractor[vocals] @ git+https://github.com/Sebi-crb/WaifuExtractor'
```

For GPU speed, install the PyTorch build for your card (CUDA or ROCm) into the same
environment first; on CPU it works but takes several minutes per episode.

## Usage

```sh
waifu-extract "/path/to/Season 1"          # interactive: asks for everything
waifu-extract "/path/to/show" --list       # just show who speaks how much
waifu-extract ep01.mkv -c Aqua -m audio -O clips -y
waifu-extract "/path/to/show" -c Kaz,Kazuma -m video -O strip --fast
```

Interactive mode asks, in order: character (several spelling variants can be picked at
once, and it offers similar names like *Kaz* / *Kazuma*), audio or video, clips or strip,
fast or accurate video cuts, vocal isolation (if Demucs is installed) and the audio track
(Japanese by default). Each question has a matching flag, see `--help`.

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
  neighbouring sound; `--isolate-vocals` and `--min-length` help for voice datasets.
- Timings match the audio the subtitles were made for (usually Japanese); the dub can drift.

Only use it on media you own.
