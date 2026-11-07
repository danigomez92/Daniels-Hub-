# beatcut

Cuts a library of video clips to the beat of a song. Point it at folders of
footage and an audio file; it detects the beat grid, builds an edit plan, and
renders a finished video. Source audio is always discarded — the only audio in
the output is the song you name.

## Requirements

- `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg`)
- Python 3.9+
- `pip install numpy librosa` — librosa is only needed for automatic beat
  detection; you can skip it and pass `--bpm` instead
- `pip install moviepy` — only for `--backend moviepy`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy librosa moviepy

# print the edit plan without rendering
python beatcut.py -c ./footage -s ./track.mp3 -o ./out/cut.mp4 --dry-run

# render it
python beatcut.py -c ./footage -s ./track.mp3 -o ./out/cut.mp4
```

Always `--dry-run` first. It prints the clip scan, the detected tempo, the cut
list, and the usage spread, and costs nothing.

## How it works

1. **Beat detection.** librosa finds the tempo and beat positions. `--fixed-grid`
   locks those to a metronomic grid seeded by the first detected beat, which is
   usually cleaner for quantised music. `--bpm` skips detection entirely.
2. **Cut grid.** The beat list is walked in `--pattern` steps and snapped to
   whole frames. Snapping matters: without it, per-segment rounding error
   random-walks and the cuts drift off the beat over a long edit. Snapped, total
   drift stays under half a frame.
3. **Clip scan.** Every video under the source folders is probed with ffprobe,
   in parallel, and cached by path + mtime + size.
4. **Edit plan.** For each cut the picker chooses a clip and an in-point.
5. **Render.** Segments are extracted to a temp directory, concatenated, and
   muxed with the song.

## Avoiding repetition

Four mechanisms, all tunable:

| Mechanism | Flag | What it does |
|---|---|---|
| Cooldown | `--cooldown` | Cuts a clip must sit out before reuse. `-1` auto-sets to ~60% of the library |
| Time gap | `--min-gap` | Seconds of finished video before a clip may return (default 6) |
| Reuse bias | `--reuse-bias` | Weights selection by `1/(1+uses)^bias`, pulling under-used clips forward |
| In-point spread | — | When a clip does return, the in-point is picked from 16 candidates, keeping whichever lands farthest from moments already shown |

**Duplicate grouping.** Folders accumulate `Clip.mp4` and `Clip copy.mp4`. Those
are the same footage under different names, so without grouping the picker cuts
straight from one into the other. Files are merged into a family when their
normalised filenames match, or when byte size *and* duration are both identical.
A family shares one cooldown.

Note that duration plus resolution is deliberately **not** used as a signal.
Batches of AI-generated clips often share an exact length and frame size while
being completely different footage, and matching on that collapses unrelated
material into one family.

## Patterns

`--pattern` is a repeating cycle of beats-per-cut. Fractions are allowed; the
beat grid subdivides to match.

| Pattern | Feel |
|---|---|
| `4` | Slow, one cut per bar |
| `2` | Steady two-beat cut (default) |
| `4,4,2,2` | Breathes, then tightens |
| `2,2,1,1,0.5` | Accelerating into half-beat cuts |

## Options

**Sources and output**

| Flag | Default | Notes |
|---|---|---|
| `-c, --clips` | required | One or more folders, space separated |
| `-s, --song` | required | Audio file for the final track |
| `-o, --out` | required | Output path; parent folders are created |
| `--recursive` | off | Scan subfolders |
| `--exclude` | none | Glob patterns to skip, e.g. `'*proxy*'` |
| `--size` | `1920x1080` | Output resolution; clips are scaled and centre-cropped to fill |
| `--fps` | `30` | Output frame rate |

**Music and timing**

| Flag | Default | Notes |
|---|---|---|
| `--pattern` | `2` | Beats per cut, cycled |
| `--bpm` | auto | Fixed grid at this tempo; skips librosa |
| `--first-beat` | `0.0` | Downbeat offset, used with `--bpm` |
| `--fixed-grid` | off | Lock detected beats to a metronome |
| `--song-start` | `0.0` | Begin the edit this far into the song |
| `--max-duration` | none | Cap the finished video length |
| `--analyze-seconds` | all | Analyse only this much audio |
| `--fade-in` / `--fade-out` | `0.0` / `1.5` | Audio fades |

**Clip selection**

| Flag | Default | Notes |
|---|---|---|
| `--min-cut` / `--max-cut` | `0.15` / `4.0` | Duration bounds; shorter cuts merge, longer ones split |
| `--cooldown` | auto | Cuts before a clip may repeat |
| `--min-gap` | `6.0` | Seconds before a clip may repeat |
| `--reuse-bias` | `1.5` | Higher pushes harder toward unused clips |
| `--margin` | `0.25` | Seconds trimmed from each clip's head and tail |
| `--seed` | random | Same seed plus same footage reproduces an edit exactly |

**Performance and output control**

| Flag | Default | Notes |
|---|---|---|
| `--backend` | `ffmpeg` | `moviepy` also available; see below |
| `--encoder` | `libx264` | `videotoolbox` for Apple Silicon hardware encoding |
| `--crf` / `--preset` | `18` / `veryfast` | x264 quality and speed |
| `--jobs` | `4` | Parallel segment extractions |
| `--scan-jobs` | `8` | Parallel ffprobe workers; lower for spinning disks |
| `--cache` / `--no-cache` | `.beatcut_cache.json` | Probe cache |
| `--temp-dir` | system temp | Where segments are staged |
| `--keep-temp` | off | Keep temp segments after rendering |
| `--dry-run` | off | Print the plan, render nothing |
| `--report` | none | Write the edit plan to JSON |

## Backends

`ffmpeg` (default) extracts each cut to a normalised temp segment in parallel,
joins them with the concat demuxer, and muxes the song. `moviepy` builds the
same edit through MoviePy instead.

Use the default. MoviePy holds an open reader per segment for the entire render,
so a 139-cut edit means 139 live subprocesses. On a test edit the ffmpeg backend
ran roughly 2.7x faster, and the gap widens with cut count. Both produce
frame-identical output from the same seed.

## Example

```bash
python beatcut.py \
  -c "/Volumes/LaCie/Footage" "/Volumes/Archive/B-Roll" \
  -s "/Volumes/LaCie/Music/track.mp3" \
  -o ~/Desktop/Exports/cut.mp4 \
  --recursive \
  --pattern 4,4,2,2 \
  --size 1920x1080 --fps 30 \
  --encoder videotoolbox \
  --seed 1
```

## Troubleshooting

**`command not found: python`** — macOS ships `python3` only. Create a venv;
inside it, plain `python` works.

**`these source folders are not readable`** — an external drive is asleep or
unmounted. The check runs before beat analysis so you find out immediately.

**Clip count lower than expected** — the scan reports what it skipped and why.
The usual cause is `--margin`: clips shorter than your longest cut plus head and
tail trim are rejected. macOS `._` resource forks are skipped silently.

**`no clip is long enough for a Ns cut`** — lower `--max-cut`, use a denser
`--pattern`, or reduce `--margin`.

**Tempo detected at half or double** — pass `--bpm` explicitly, or try
`--fixed-grid`.

**Dead air or black frames in the cut** — long unedited timelines get sampled at
arbitrary points, including slates and gaps between takes. Pull selects out of
those files first, or exclude them.

**`h264_videotoolbox not available`** — the encoder is missing from your ffmpeg
build; it falls back to libx264 automatically.

## Notes

- Clip audio is dropped at two points: `-an -sn -dn` on every extraction, and
  `audio=False` when MoviePy opens a source. There is no path by which source
  audio reaches the output.
- Clips are scaled to fill and centre-cropped, so mixed aspect ratios produce no
  letterboxing. Rotation metadata is honoured.
- The video ends when the plan ends. If the song is shorter than the planned
  edit, the render is trimmed to the song and a warning is printed.
