#!/usr/bin/env python3
"""
beatcut.py — cut a folder of clips to the beat of a song.

Pipeline
--------
1. Detect the beat grid of the song (librosa, or a manual --bpm grid).
2. Turn that grid into a list of cut durations using a rhythmic pattern
   (e.g. "4,4,2,2" = two 4-beat shots, then two 2-beat shots, repeating).
3. Scan a folder of clips with ffprobe (duration / dimensions / fps).
4. For each cut, pick a source clip and an in-point using a picker that
   enforces a cooldown, weights toward under-used clips, and avoids
   returning to a part of a clip it has already shown.
5. Render. Clip audio is discarded everywhere; the only audio track is the
   song you name.

Usage
-----
    python beatcut.py --clips ./footage --song ./track.wav --out cut.mp4

    # 2-beat cuts, 9:16, start 32s into the song, hard cap at 60s
    python beatcut.py -c ./footage -s track.wav -o reel.mp4 \
        --pattern 2 --size 1080x1920 --song-start 32 --max-duration 60

    # no librosa installed / you know the tempo
    python beatcut.py -c ./footage -s track.wav -o cut.mp4 --bpm 128 --first-beat 0.31

    # see the edit decision list without rendering
    python beatcut.py -c ./footage -s track.wav -o cut.mp4 --dry-run

Requirements
------------
    ffmpeg + ffprobe on PATH
    pip install numpy librosa          (librosa optional if you pass --bpm)
    pip install moviepy                (only for --backend moviepy)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg",
    ".wmv", ".mts", ".m2ts", ".flv", ".3gp", ".ts",
}


# --------------------------------------------------------------------------
# ffprobe helpers
# --------------------------------------------------------------------------

def require_binaries() -> None:
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            sys.exit(f"error: {exe} not found on PATH")


def has_encoder(name: str) -> bool:
    """Check an ffmpeg encoder actually exists before relying on it."""
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20)
        return f" {name} " in res.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def probe(path: Path) -> Optional[dict]:
    """Return {duration, width, height, fps} for a media file, or None."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None

    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    fmt = data.get("format", {})

    duration = 0.0
    for candidate in (fmt.get("duration"), (vstream or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    if vstream is None:
        return {"duration": duration, "width": 0, "height": 0, "fps": 0.0}

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)

    # Honour rotation metadata so aspect logic matches what ffmpeg will output.
    rotation = 0
    for sd in vstream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(abs(float(sd["rotation"]))) % 180
            except (TypeError, ValueError):
                rotation = 0
    if rotation == 90:
        width, height = height, width

    fps = 0.0
    rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/0"
    try:
        num, _, den = rate.partition("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    return {"duration": duration, "width": width, "height": height, "fps": fps}


# --------------------------------------------------------------------------
# Beat detection
# --------------------------------------------------------------------------

def detect_beats(song: Path, song_start: float, analyze_len: Optional[float],
                 bpm_hint: Optional[float], fixed_grid: bool) -> Tuple[float, np.ndarray]:
    """Return (tempo, absolute beat times in seconds) using librosa."""
    try:
        import librosa
    except ImportError:
        sys.exit(
            "librosa is not installed. Either `pip install librosa` or supply a "
            "manual grid with --bpm (and optionally --first-beat)."
        )

    y, sr = librosa.load(str(song), sr=None, mono=True,
                         offset=song_start, duration=analyze_len)
    if y.size == 0:
        sys.exit("error: no audio decoded — check --song-start against the song length")

    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, units="time",
        start_bpm=bpm_hint or 120.0,
        tightness=100,
    )
    beats = np.asarray(beats, dtype=float)
    tempo = float(np.atleast_1d(tempo)[0])

    if beats.size < 2:
        sys.exit("error: beat tracking found no usable beats — try --bpm instead")

    if fixed_grid:
        # Lock to a metronomic grid seeded by the first detected beat. Cleaner
        # than raw detection for electronic / quantised music.
        period = float(np.median(np.diff(beats)))
        span = len(y) / sr
        count = int((span - beats[0]) / period)
        beats = beats[0] + period * np.arange(max(count, 1) + 1)
        tempo = 60.0 / period

    return tempo, beats + song_start


def manual_beats(bpm: float, first_beat: float, song_duration: float) -> np.ndarray:
    period = 60.0 / bpm
    count = int(max(song_duration - first_beat, 0) / period)
    return first_beat + period * np.arange(count + 1)


def subdivide(beats: np.ndarray, factor: int) -> np.ndarray:
    """Linearly interpolate `factor` points between every pair of beats."""
    if factor <= 1:
        return beats
    out = []
    for a, b in zip(beats[:-1], beats[1:]):
        for k in range(factor):
            out.append(a + (b - a) * k / factor)
    out.append(beats[-1])
    return np.asarray(out, dtype=float)


def parse_pattern(text: str) -> List[float]:
    values = []
    for token in text.replace(" ", "").split(","):
        if not token:
            continue
        values.append(float(Fraction(token)))
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("--pattern must be positive numbers, e.g. '4,4,2,2' or '1,0.5'")
    return values


def snap_to_frames(times: Sequence[float], fps: int) -> List[float]:
    """
    Quantise cut boundaries to whole frames.

    Without this, each backend rounds each segment's length independently and
    the error random-walks: by the end of a long edit the cuts sit several
    frames off the beat. Snapping the boundaries once, up front, keeps every
    segment an exact frame count and caps total drift at half a frame.
    """
    snapped: List[float] = []
    for t in times:
        q = round(t * fps) / fps
        if snapped and q <= snapped[-1]:
            q = snapped[-1] + 1.0 / fps
        snapped.append(q)
    return snapped


def build_cut_times(beats: np.ndarray, pattern: Sequence[float],
                    min_dur: float, max_dur: float) -> List[float]:
    """Walk the beat grid in `pattern` steps and return cut boundary times."""
    denominators = [Fraction(p).limit_denominator(16).denominator for p in pattern]
    factor = 1
    for d in denominators:
        factor = factor * d // math.gcd(factor, d)
    grid = subdivide(beats, factor)
    steps = [max(1, int(round(p * factor))) for p in pattern]

    times: List[float] = []
    i, p = 0, 0
    while i < len(grid) - 1:
        times.append(float(grid[i]))
        i += steps[p % len(steps)]
        p += 1
    times.append(float(grid[-1]))

    # Merge anything shorter than min_dur into its neighbour; split anything
    # longer than max_dur down the middle.
    cleaned = [times[0]]
    for t in times[1:]:
        if t - cleaned[-1] < min_dur:
            continue
        while t - cleaned[-1] > max_dur:
            cleaned.append(cleaned[-1] + (t - cleaned[-1]) / 2.0)
        cleaned.append(t)
    return cleaned


# --------------------------------------------------------------------------
# Source clips + the anti-repetition picker
# --------------------------------------------------------------------------

def family_key(path: Path) -> str:
    """
    Group near-duplicate assets so they share one cooldown.

    A folder that has accumulated "Clip.mp4" and "Clip copy.mp4" will otherwise
    cut from one straight into the other: different filenames, same footage.
    Matching is deliberately conservative — only unambiguous duplicate markers
    are stripped, so "ON TV 2" and "ON TV 3" stay separate.
    """
    stem = path.stem.lower()
    stem = re.sub(r"[*]+", " ", stem)                       # Clip **.mp4
    stem = re.sub(r"\s*\((?:\d+)\)\s*$", " ", stem)          # Clip (1).mp4
    stem = re.sub(r"\b(copy|duplicate|dup|kopie)\b", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or path.stem.lower()


@dataclass
class Source:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    family: str = ""
    uses: int = 0
    last_cut_index: int = -10_000
    last_timeline_pos: float = -1e9
    used_spans: List[Tuple[float, float]] = field(default_factory=list)


def load_cache(path: Optional[Path]) -> dict:
    if path and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_cache(path: Optional[Path], cache: dict) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache), encoding="utf-8")
    except OSError as exc:
        print(f"warning: could not write cache ({exc})")


def scan_clips(folders: Sequence[Path], min_clip_len: float, recursive: bool,
               jobs: int = 8, cache_path: Optional[Path] = None,
               exclude: Sequence[str] = ()) -> List[Source]:
    """
    Probe every video under one or more folders.

    Results are cached by path + mtime + size, so rescanning a drive full of
    footage is near-instant after the first pass. Probing runs in parallel
    because one ffprobe per file over thousands of files on external media is
    otherwise the slowest part of the whole run.
    """
    missing = [f for f in folders if not f.is_dir()]
    if missing:
        listing = "\n".join(f"  {m}" for m in missing)
        sys.exit(
            f"error: these source folders are not readable — is the drive mounted?\n{listing}"
        )

    files: List[Path] = []
    seen: set = set()
    for folder in folders:
        walker = folder.rglob("*") if recursive else folder.glob("*")
        found = 0
        for p in sorted(walker):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            if p.name.startswith("._"):          # macOS resource forks
                continue
            if any(p.match(pat) for pat in exclude):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            files.append(p)
            found += 1
        print(f"  {folder}: {found} video files")

    if not files:
        sys.exit("error: no video files found in any source folder")

    cache = load_cache(cache_path)
    fresh = 0

    def info_for(path: Path) -> Tuple[Path, Optional[dict]]:
        key = str(path.resolve())
        try:
            st = path.stat()
            stamp = [int(st.st_mtime), st.st_size]
        except OSError:
            return path, None
        hit = cache.get(key)
        if hit and hit.get("stamp") == stamp:
            return path, hit
        info = probe(path)
        if info is not None:
            info = dict(info, stamp=stamp)
        return path, info

    print(f"  probing {len(files)} files with {jobs} worker(s)...")
    results: List[Tuple[Path, Optional[dict]]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for path, info in pool.map(info_for, files):
            results.append((path, info))
            if info is not None and str(path.resolve()) not in cache:
                fresh += 1

    sources: List[Source] = []
    skipped = {"unreadable": 0, "no video": 0, "too short": 0}
    for path, info in results:
        if info is None:
            skipped["unreadable"] += 1
            continue
        cache[str(path.resolve())] = info
        if info["width"] == 0 or info["height"] == 0:
            skipped["no video"] += 1
            continue
        if info["duration"] < min_clip_len:
            skipped["too short"] += 1
            continue
        sources.append(Source(path=path, duration=info["duration"],
                              width=info["width"], height=info["height"],
                              fps=info["fps"]))

    # Merge duplicate families using two STRONG signals only:
    #   1. normalised filename ("Clip.mp4" / "Clip copy.mp4")
    #   2. identical byte size AND identical duration
    # Duration+resolution alone is NOT usable: batches of AI-generated clips
    # share an exact length and frame size while being completely different
    # footage, which collapses unrelated material into one family.
    parent: dict = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_name: dict = {}
    by_bytes: dict = {}
    for src in sources:
        node = str(src.path.resolve())
        find(node)
        name = family_key(src.path)
        if name in by_name:
            union(node, by_name[name])
        else:
            by_name[name] = node
        try:
            size = src.path.stat().st_size
        except OSError:
            continue
        shape = (size, round(src.duration, 2))
        if shape in by_bytes:
            union(node, by_bytes[shape])
        else:
            by_bytes[shape] = node

    for src in sources:
        src.family = find(str(src.path.resolve()))

    groups: dict = {}
    for src in sources:
        groups.setdefault(src.family, []).append(src.path.name)
    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    if dupes:
        print(f"  grouped {sum(len(v) for v in dupes.values())} duplicate files into "
              f"{len(dupes)} set(s); each set shares one cooldown:")
        for names in list(dupes.values())[:8]:
            print(f"    {', '.join(n[:44] for n in names)}")

    save_cache(cache_path, cache)
    detail = ", ".join(f"{v} {k}" for k, v in skipped.items() if v)
    if detail:
        print(f"  skipped: {detail}")
    if fresh < len(files):
        print(f"  ({len(files) - fresh} read from cache)")
    if not sources:
        sys.exit("error: no usable clips after scanning")
    return sources


class ClipPicker:
    """
    Chooses a source clip and an in-point for each cut.

    Three mechanisms keep it from feeling repetitive:
      * cooldown   — a clip cannot come back for N cuts after it is used
      * min-gap    — a clip cannot come back within N seconds of screen time
      * reuse bias — clips are weighted by 1 / (1 + uses) ** bias, so the
                     under-used tail of the folder gets pulled forward
    Within a clip, the in-point is chosen from several random candidates,
    keeping whichever lands farthest from moments already used on screen.
    """

    def __init__(self, sources: List[Source], cooldown: int, min_gap: float,
                 reuse_bias: float, margin: float, rng: random.Random,
                 start_candidates: int = 16):
        self.sources = sources
        self.min_gap = min_gap
        self.reuse_bias = reuse_bias
        self.margin = margin
        self.rng = rng
        self.start_candidates = start_candidates
        self.fam_cut: dict = {}
        self.fam_pos: dict = {}
        if cooldown < 0:
            cooldown = max(1, int(len(sources) * 0.6))
        self.cooldown = min(cooldown, max(0, len(sources) - 1))

    def _eligible(self, need: float, cut_index: int, pos: float, relax: int) -> List[Source]:
        cooldown = [self.cooldown, self.cooldown // 2, 0, 0][relax]
        min_gap = [self.min_gap, self.min_gap / 2.0, 0.0, 0.0][relax]
        margin = [self.margin, self.margin, self.margin / 2.0, 0.0][relax]
        out = []
        for s in self.sources:
            if s.duration < need + 2 * margin:
                continue
            # A zero threshold means "no constraint" — this is what makes the
            # last relax level a true fallback rather than another filter.
            # Check the clip's own history and its duplicate-family history.
            last_cut = max(s.last_cut_index, self.fam_cut.get(s.family, -10_000))
            last_pos = max(s.last_timeline_pos, self.fam_pos.get(s.family, -1e9))
            if cooldown > 0 and cut_index - last_cut <= cooldown:
                continue
            if min_gap > 0 and pos - last_pos < min_gap:
                continue
            out.append(s)
        return out

    def reset(self) -> None:
        """Clear usage history so the same Source list can drive a second plan."""
        for s in self.sources:
            s.uses = 0
            s.last_cut_index = -10_000
            s.last_timeline_pos = -1e9
            s.used_spans.clear()

    def _choose_start(self, src: Source, need: float, margin: float) -> float:
        lo, hi = margin, src.duration - need - margin
        if hi <= lo:
            lo, hi = 0.0, max(0.0, src.duration - need)
        if hi <= lo:
            return 0.0
        if not src.used_spans:
            return self.rng.uniform(lo, hi)
        best, best_score = lo, -1.0
        for _ in range(self.start_candidates):
            cand = self.rng.uniform(lo, hi)
            score = min(abs(cand - s) for s, _ in src.used_spans)
            if score > best_score:
                best, best_score = cand, score
        return best

    def pick(self, need: float, cut_index: int, pos: float) -> Tuple[Source, float]:
        candidates: List[Source] = []
        relax = 0
        for relax in range(4):
            candidates = self._eligible(need, cut_index, pos, relax)
            if candidates:
                break
        if not candidates:
            raise RuntimeError(
                f"no clip is long enough for a {need:.2f}s cut (longest source is "
                f"{max(s.duration for s in self.sources):.2f}s) — lower --max-cut, "
                f"use a denser --pattern, or reduce --margin"
            )
        margin = [self.margin, self.margin, self.margin / 2.0, 0.0][relax]
        weights = [1.0 / (1.0 + s.uses) ** self.reuse_bias for s in candidates]
        src = self.rng.choices(candidates, weights=weights, k=1)[0]
        start = self._choose_start(src, need, margin)

        src.uses += 1
        src.last_cut_index = cut_index
        src.last_timeline_pos = pos
        src.used_spans.append((start, start + need))
        self.fam_cut[src.family] = cut_index
        self.fam_pos[src.family] = pos
        return src, start


# --------------------------------------------------------------------------
# Edit plan
# --------------------------------------------------------------------------

@dataclass
class Cut:
    index: int
    timeline_start: float
    duration: float
    source: Path
    source_in: float

    @property
    def source_out(self) -> float:
        return self.source_in + self.duration


def build_plan(cut_times: Sequence[float], picker: ClipPicker,
               max_total: Optional[float]) -> List[Cut]:
    plan: List[Cut] = []
    pos = 0.0
    for i in range(len(cut_times) - 1):
        duration = cut_times[i + 1] - cut_times[i]
        if max_total is not None and pos + duration > max_total:
            duration = max_total - pos
            if duration < 0.08:
                break
        src, start = picker.pick(duration, i, pos)
        plan.append(Cut(index=i, timeline_start=pos, duration=duration,
                        source=src.path, source_in=start))
        pos += duration
        if max_total is not None and pos >= max_total - 1e-6:
            break
    if not plan:
        sys.exit("error: the edit plan came out empty")
    return plan


# --------------------------------------------------------------------------
# Rendering — ffmpeg backend (default)
# --------------------------------------------------------------------------

def vf_chain(width: int, height: int, fps: int) -> str:
    return (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={width}:{height},setsar=1,format=yuv420p"
    )


def extract_segment(cut: Cut, dest: Path, width: int, height: int, fps: int,
                    crf: int, preset: str, encoder: str = "libx264") -> Path:
    # -frames:v rather than -t: asking for a duration lets ffmpeg return an
    # extra frame depending on where the seek lands, and a single extra frame
    # shifts every cut after it. An explicit frame count is exact.
    nframes = max(1, round(cut.duration * fps))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{cut.source_in:.6f}",
        "-i", str(cut.source),
        "-frames:v", str(nframes),
        "-an", "-sn", "-dn",                      # never carry source audio
        "-vf", vf_chain(width, height, fps),
        "-r", str(fps),
        *(["-c:v", "h264_videotoolbox", "-q:v", str(max(1, 100 - crf * 2))]
          if encoder == "videotoolbox" else
          ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]),
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", "90000",
        str(dest),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg failed on {cut.source.name}\n{res.stderr.strip()}")
    return dest


def render_ffmpeg(plan: List[Cut], song: Path, out: Path, width: int, height: int,
                  fps: int, song_start: float, fade_in: float, fade_out: float,
                  crf: int, preset: str, jobs: int, keep_temp: bool,
                  temp_dir: Optional[Path] = None, encoder: str = "libx264") -> None:
    # Frame-exact total: the segments are whole frame counts, so derive the
    # timeline length the same way rather than from float durations.
    if encoder == "videotoolbox" and not has_encoder("h264_videotoolbox"):
        print("warning: h264_videotoolbox not available in this ffmpeg build — using libx264")
        encoder = "libx264"

    total_frames = sum(max(1, round(c.duration * fps)) for c in plan)
    total = total_frames / fps
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="beatcut_", dir=str(temp_dir) if temp_dir else None))
    try:
        print(f"Extracting {len(plan)} segments with {jobs} worker(s)...")
        targets = [tmpdir / f"seg_{c.index:05d}.mp4" for c in plan]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(extract_segment, c, t, width, height, fps, crf, preset, encoder)
                for c, t in zip(plan, targets)
            ]
            for n, f in enumerate(futures, 1):
                f.result()
                if n % 20 == 0 or n == len(futures):
                    print(f"  {n}/{len(futures)}")

        listfile = tmpdir / "concat.txt"
        listfile.write_text(
            "".join(f"file '{str(t.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
                    for t in targets),
            encoding="utf-8",
        )

        filters = []
        if fade_in > 0:
            filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
        if fade_out > 0:
            filters.append(f"afade=t=out:st={max(total - fade_out, 0):.3f}:d={fade_out:.3f}")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-stats",
            "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ss", f"{song_start:.6f}", "-i", str(song),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
            "-shortest",
            "-movflags", "+faststart",
        ]
        if filters:
            cmd += ["-af", ",".join(filters)]
        cmd.append(str(out))

        print("Muxing final video...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"final mux failed:\n{res.stderr.strip()}")
    finally:
        if keep_temp:
            print(f"Temp segments kept in {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# Rendering — MoviePy backend
# --------------------------------------------------------------------------

def render_moviepy(plan: List[Cut], song: Path, out: Path, width: int, height: int,
                   fps: int, song_start: float, fade_in: float, fade_out: float,
                   crf: int, preset: str) -> None:
    try:
        from moviepy import (AudioFileClip, VideoFileClip,  # MoviePy 2.x
                             concatenate_videoclips)
        mp2 = True
    except ImportError:
        try:
            from moviepy.editor import (AudioFileClip, VideoFileClip,  # MoviePy 1.x
                                        concatenate_videoclips)
            mp2 = False
        except ImportError:
            sys.exit("moviepy is not installed — `pip install moviepy` or use --backend ffmpeg")

    def _sub(clip, a, b):
        return clip.subclipped(a, b) if mp2 else clip.subclip(a, b)

    def _resize(clip, size):
        return clip.resized(new_size=size) if mp2 else clip.resize(newsize=size)

    def _crop(clip, x1, y1, x2, y2):
        if mp2:
            return clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
        from moviepy.video.fx.all import crop
        return crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)

    def _with_audio(clip, audio):
        return clip.with_audio(audio) if mp2 else clip.set_audio(audio)

    def _fade(aclip, fin, fout):
        if mp2:
            from moviepy import afx
            fx = []
            if fin > 0:
                fx.append(afx.AudioFadeIn(fin))
            if fout > 0:
                fx.append(afx.AudioFadeOut(fout))
            return aclip.with_effects(fx) if fx else aclip
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout
        if fin > 0:
            aclip = audio_fadein(aclip, fin)
        if fout > 0:
            aclip = audio_fadeout(aclip, fout)
        return aclip

    if len(plan) > 80:
        print(f"warning: {len(plan)} cuts on the moviepy backend means {len(plan)} open "
              f"ffmpeg readers. Use --backend ffmpeg for edits this long.")

    def fit(clip):
        scale = max(width / clip.w, height / clip.h)
        clip = _resize(clip, (math.ceil(clip.w * scale), math.ceil(clip.h * scale)))
        x1 = int((clip.w - width) / 2)
        y1 = int((clip.h - height) / 2)
        return _crop(clip, x1=x1, y1=y1, x2=x1 + width, y2=y1 + height)

    opened, segments = [], []
    total = sum(c.duration for c in plan)
    try:
        for c in plan:
            base = VideoFileClip(str(c.source), audio=False)   # audio dropped at load
            opened.append(base)
            segments.append(fit(_sub(base, c.source_in, c.source_out)))

        video = concatenate_videoclips(segments, method="chain")
        audio = _fade(_sub(AudioFileClip(str(song)), song_start, song_start + total),
                      fade_in, fade_out)
        final = _with_audio(video, audio)
        final.write_videofile(
            str(out), fps=fps, codec="libx264", audio_codec="aac",
            preset=preset, audio_bitrate="320k",
            ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart"],
        )
    finally:
        for clip in opened:
            try:
                clip.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_size(text: str) -> Tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("--size must look like 1920x1080")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cut a folder of clips to the beat of a song.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-c", "--clips", required=True, type=Path, nargs="+",
                    help="one or more folders of source clips (space separated)")
    ap.add_argument("-s", "--song", required=True, type=Path, help="audio file for the final track")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output video path")

    ap.add_argument("--recursive", action="store_true", help="scan subfolders too")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="glob patterns to skip, e.g. '*proxy*' '*_LOG.mov'")
    ap.add_argument("--cache", type=Path, default=Path(".beatcut_cache.json"),
                    help="probe cache; makes rescans near-instant")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--scan-jobs", type=int, default=8, help="parallel ffprobe workers")
    ap.add_argument("--temp-dir", type=Path, default=None,
                    help="where to stage segments (default: system temp, usually fastest)")
    ap.add_argument("--size", type=parse_size, default="1920x1080", help="output resolution")
    ap.add_argument("--fps", type=int, default=30)

    ap.add_argument("--pattern", type=parse_pattern, default="2",
                    help="beats per cut, cycled: '2' | '4,4,2,2' | '2,2,1,1,0.5'")
    ap.add_argument("--bpm", type=float, default=None,
                    help="skip beat detection and use a fixed grid at this tempo")
    ap.add_argument("--first-beat", type=float, default=0.0,
                    help="downbeat offset in seconds, used with --bpm")
    ap.add_argument("--fixed-grid", action="store_true",
                    help="lock detected beats to a metronomic grid (good for electronic music)")
    ap.add_argument("--song-start", type=float, default=0.0,
                    help="start the edit this far into the song")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="cap the finished video at this many seconds")
    ap.add_argument("--analyze-seconds", type=float, default=None,
                    help="only analyse this much audio (speeds up long tracks)")

    ap.add_argument("--min-cut", type=float, default=0.15, help="shortest allowed cut")
    ap.add_argument("--max-cut", type=float, default=4.0, help="longest allowed cut")
    ap.add_argument("--cooldown", type=int, default=-1,
                    help="cuts a clip must sit out before reuse (-1 = auto, 60%% of library)")
    ap.add_argument("--min-gap", type=float, default=6.0,
                    help="seconds of finished video before a clip may return")
    ap.add_argument("--reuse-bias", type=float, default=1.5,
                    help="higher = harder push toward the least-used clips")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="seconds trimmed off each clip's head and tail")
    ap.add_argument("--seed", type=int, default=None, help="seed for a reproducible edit")

    ap.add_argument("--fade-in", type=float, default=0.0, help="audio fade in")
    ap.add_argument("--fade-out", type=float, default=1.5, help="audio fade out")
    ap.add_argument("--backend", choices=("ffmpeg", "moviepy"), default="ffmpeg")
    ap.add_argument("--encoder", choices=("libx264", "videotoolbox"), default="libx264",
                    help="videotoolbox uses Apple Silicon hardware encoding — much faster")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--jobs", type=int, default=4, help="parallel segment extractions")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the edit plan, render nothing")
    ap.add_argument("--report", type=Path, default=None, help="write the edit plan to JSON")

    args = ap.parse_args()
    require_binaries()

    if isinstance(args.size, str):
        args.size = parse_size(args.size)
    if isinstance(args.pattern, str):
        args.pattern = parse_pattern(args.pattern)
    width, height = args.size

    if not args.song.is_file():
        sys.exit(f"error: {args.song} not found")

    # Fail fast on sleeping or unplugged drives, before beat analysis.
    missing = [f for f in args.clips if not f.is_dir()]
    if missing:
        listing = "\n".join(f"  {m}" for m in missing)
        sys.exit(f"error: these source folders are not readable — is the drive mounted?\n{listing}")

    rng = random.Random(args.seed)

    # 1. Beats -------------------------------------------------------------
    if args.bpm:
        song_info = probe(args.song) or {}
        song_len = song_info.get("duration") or 600.0
        beats = manual_beats(args.bpm, args.first_beat, song_len)
        tempo = args.bpm
        print(f"Beat grid: manual, {tempo:.1f} BPM, first beat {args.first_beat:.3f}s")
    else:
        tempo, beats = detect_beats(args.song, args.song_start,
                                    args.analyze_seconds, None, args.fixed_grid)
        print(f"Beat grid: detected, {tempo:.1f} BPM, {len(beats)} beats")

    beats = beats[beats >= args.song_start] - args.song_start
    if beats.size < 2:
        sys.exit("error: not enough beats after --song-start")

    cut_times = build_cut_times(beats, args.pattern, args.min_cut, args.max_cut)
    cut_times = snap_to_frames(cut_times, args.fps)
    max_drift = 1000.0 / (2 * args.fps)
    print(f"Pattern {args.pattern} -> {len(cut_times) - 1} cuts "
          f"over {cut_times[-1] - cut_times[0]:.1f}s of song "
          f"(snapped to {args.fps}fps grid, max drift {max_drift:.1f}ms)")

    # 2. Clips -------------------------------------------------------------
    print(f"Scanning {len(args.clips)} source folder(s)...")
    sources = scan_clips(args.clips, args.min_cut + 2 * args.margin, args.recursive,
                         jobs=max(1, args.scan_jobs),
                         cache_path=None if args.no_cache else args.cache,
                         exclude=args.exclude)
    print(f"Found {len(sources)} usable clips "
          f"({sum(s.duration for s in sources) / 60:.1f} min of footage)")

    # 3. Plan --------------------------------------------------------------
    picker = ClipPicker(sources, args.cooldown, args.min_gap, args.reuse_bias,
                        args.margin, rng)
    plan = build_plan(cut_times, picker, args.max_duration)
    total = sum(c.duration for c in plan)
    print(f"Edit plan: {len(plan)} cuts, {total:.2f}s, "
          f"cooldown {picker.cooldown} cuts / {args.min_gap:.1f}s")

    song_len = (probe(args.song) or {}).get("duration") or 0.0
    if song_len and song_len - args.song_start < total - 0.05:
        print(f"warning: only {song_len - args.song_start:.1f}s of song remains after "
              f"--song-start, so the video will be trimmed to match")

    counts = sorted(((s.uses, s.path.name) for s in sources), reverse=True)
    print(f"  most used:  {counts[0][1]} ({counts[0][0]}x)")
    print(f"  least used: {counts[-1][1]} ({counts[-1][0]}x)")

    if args.report or args.dry_run:
        report = {
            "song": str(args.song), "sources": [str(c) for c in args.clips],
            "tempo": tempo, "song_start": args.song_start,
            "pattern": args.pattern, "seed": args.seed, "total_duration": total,
            "cuts": [
                {"i": c.index, "at": round(c.timeline_start, 3),
                 "dur": round(c.duration, 3), "clip": c.source.name,
                 "in": round(c.source_in, 3), "out": round(c.source_out, 3),
                 "path": str(c.source)}
                for c in plan
            ],
        }
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Wrote plan to {args.report}")
        if args.dry_run:
            for c in plan:
                print(f"  {c.timeline_start:7.2f}s  {c.duration:5.2f}s  "
                      f"{c.source.name}  [{c.source_in:.2f} -> {c.source_out:.2f}]")
            return

    # 4. Render ------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "ffmpeg":
        render_ffmpeg(plan, args.song, args.out, width, height, args.fps,
                      args.song_start, args.fade_in, args.fade_out,
                      args.crf, args.preset, max(1, args.jobs), args.keep_temp,
                      args.temp_dir, args.encoder)
    else:
        render_moviepy(plan, args.song, args.out, width, height, args.fps,
                       args.song_start, args.fade_in, args.fade_out,
                       args.crf, args.preset)

    print(f"Done: {args.out} ({total:.2f}s, {width}x{height} @ {args.fps}fps)")


if __name__ == "__main__":
    main()