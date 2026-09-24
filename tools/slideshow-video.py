"""Make a photo-slideshow lyric video for a finished song.

    uv run python tools/slideshow-video.py <song> <storyboard.json> [--out name.mp4] [--fade 1.2]

Reads out/<song>/lyrics.json (word-timed, from `songcomposer chart`) and out/<song>/<song>.mp3, and a storyboard:

    {"title": "Ten Years On",                       # optional; title card over the first slide
     "slides": [
       {"photo": "photos/a.jpg"},                   # first slide starts at 0
       {"photo": "photos/b.jpg", "text": "Five a.m."},                      # starts when this lyric line starts
       {"photo": "photos/c.jpg", "text": "candles", "section": "Chorus"}   # ... first line matching, in that section
       {"photo": "photos/d.jpg", "at": 123.4}                               # or an explicit time
     ]}

Photo paths are relative to the storyboard file. Each slide gets a slow Ken Burns move (alternating in/out), sits on a
blurred copy of itself when it does not fill 16:9, and cross-fades into the next; the lyric lines are burned in with
the same timing as the SRT. Writes out/<song>/<name>.mp4 (default <song>.mp4), plus <name>.srt and <name>.ass.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1920, 1080, 25
LEAD_S = 0.35                     # a slide lands just before its line is sung
HOLD_S, GAP_S = 1.2, 0.08         # keep a line up a little after it ends, but never into the next line
ZOOM = 0.10                       # 10 % Ken Burns travel per slide


def stamp(t: float, ass: bool) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    if ass:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def cue_lines(lyrics: dict) -> list[tuple[float, float, str]]:
    lines = sorted((ln for ln in lyrics["lines"] if ln.get("start") is not None), key=lambda ln: ln["start"])
    cues = []
    for ln, nxt in zip(lines, lines[1:] + [None]):
        end = ln["end"] + HOLD_S
        if nxt is not None:
            end = min(end, nxt["start"] - GAP_S)
        cues.append((ln["start"], max(end, ln["start"] + 0.5), ln["text"]))
    return cues


def slide_times(slides: list[dict], lyrics: dict, duration: float) -> list[float]:
    """Resolve each slide's start time; anchors are consumed in order so a repeated phrase lands on its next occurrence."""
    lines = sorted((ln for ln in lyrics["lines"] if ln.get("start") is not None), key=lambda ln: ln["start"])
    starts, cursor = [0.0], 0.0
    for i, s in enumerate(slides):
        if i == 0:
            continue
        if "at" in s:
            t = float(s["at"])
        elif "text" in s:
            want = s["text"].lower()
            hits = [ln for ln in lines if ln["start"] >= cursor and want in ln["text"].lower()
                    and (not s.get("section") or ln.get("section") == s["section"])]
            k = int(s.get("occurrence", 1)) - 1          # "occurrence": 3 -> the third matching line after the previous slide
            hit = hits[k] if len(hits) > k else None
            if hit is None:
                raise SystemExit(f"storyboard slide {i}: no lyric line after {cursor:.1f}s contains {s['text']!r}"
                                 + (f" in section {s['section']!r}" if s.get("section") else ""))
            t = max(0.0, hit["start"] - LEAD_S)
        else:
            raise SystemExit(f"storyboard slide {i}: needs 'text' or 'at'")
        if i == 1 and t < 1.0:
            # the singing starts straight away: there is no room for the opening slide, so slide 1 opens the video
            print(f"  slide 0 dropped: the first line is sung at {t + LEAD_S:.1f}s")
            starts, cursor = [None], t
            continue
        if t <= cursor + 0.5:
            raise SystemExit(f"storyboard slide {i}: starts at {t:.1f}s, not after the previous slide ({cursor:.1f}s)")
        if t >= duration - 1.0:
            raise SystemExit(f"storyboard slide {i}: starts at {t:.1f}s, past the end of the song ({duration:.1f}s)")
        starts.append(t)
        cursor = t
    return starts


def ken_burns(idx: int, frames: int, zoom_in: bool) -> str:
    """One still -> `frames` frames of slow zoom with a slight drift. The still is fitted onto a blurred copy of itself so
    portrait and 4:3 photos fill 16:9 without cropping faces; zoompan works on a 2x frame to avoid its integer jitter."""
    z0, z1 = (1.0, 1.0 + ZOOM) if zoom_in else (1.0 + ZOOM, 1.0)
    z = f"{z0}+({z1}-{z0})*on/{max(frames - 1, 1)}"
    # drift the window a little sideways over the slide, alternating direction
    dx = "(iw-iw/zoom)/2" + ("+0.06*iw*on/%d" % max(frames - 1, 1) if idx % 2 == 0 else "-0.06*iw*on/%d" % max(frames - 1, 1))
    return (f"[{idx}:v]split[bg{idx}][fg{idx}];"
            f"[bg{idx}]scale={2*W}:{2*H}:force_original_aspect_ratio=increase,crop={2*W}:{2*H},gblur=sigma=40,eq=brightness=-0.18:saturation=0.85[bgs{idx}];"
            f"[fg{idx}]scale={2*W}:{2*H}:force_original_aspect_ratio=decrease[fgs{idx}];"
            f"[bgs{idx}][fgs{idx}]overlay=(W-w)/2:(H-h)/2,setsar=1,"
            f"zoompan=z='{z}':x='{dx}':y='(ih-ih/zoom)/2':d={frames}:s={W}x{H}:fps={FPS},format=yuv420p[v{idx}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("storyboard")
    ap.add_argument("--out", help="output name without extension (default: the song)")
    ap.add_argument("--fade", type=float, default=1.2, help="cross-fade seconds between slides")
    args = ap.parse_args()

    out = ROOT / "out" / args.song
    lyrics_path, mp3 = out / "lyrics.json", out / f"{args.song}.mp3"
    for p in (lyrics_path, mp3):
        if not p.exists():
            print(f"error: {p} is missing — run `songcomposer pick` and `songcomposer chart` first", file=sys.stderr)
            return 1
    sb_path = Path(args.storyboard).resolve()
    sb = json.loads(sb_path.read_text(encoding="utf-8"))
    photos = [(sb_path.parent / s["photo"]).resolve() for s in sb["slides"]]
    missing = [p for p in photos if not p.exists()]
    if missing:
        print("error: missing photos:\n  " + "\n  ".join(map(str, missing)), file=sys.stderr)
        return 1
    lyrics = json.loads(lyrics_path.read_text(encoding="utf-8"))
    duration = float(lyrics.get("duration_s") or 0) or float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
        capture_output=True, text=True, check=True).stdout.strip())
    name = args.out or args.song
    F = args.fade

    # ---- subtitles (same cues as the srt)
    cues = cue_lines(lyrics)
    (out / f"{name}.srt").write_text("\n".join(f"{i}\n{stamp(a, False)} --> {stamp(b, False)}\n{t}\n" for i, (a, b, t) in enumerate(cues, start=1)), encoding="utf-8")
    title = sb.get("title") or lyrics.get("title") or args.song
    first = cues[0][0] if cues else 3.0
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
            "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Lyric,Georgia,64,&H00FFFFFF,&H00FFFFFF,&H00101010,&HA0000000,0,0,0,0,100,100,0,0,1,3,2,2,140,140,120,1\n"
            "Style: Title,Georgia,120,&H00FFFFFF,&H00FFFFFF,&H00101010,&HA0000000,0,1,0,0,100,100,3,0,1,3,3,5,140,140,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    if first > 1.2:
        events.append(f"Dialogue: 0,{stamp(0.4, True)},{stamp(first - 0.3, True)},Title,,0,0,0,,{{\\fad(900,700)}}{title}")
    events += [f"Dialogue: 0,{stamp(a, True)},{stamp(b, True)},Lyric,,0,0,0,,{{\\fad(180,220)}}{t}" for a, b, t in cues]
    ass = out / f"{name}.ass"
    ass.write_text(head + "\n".join(events) + "\n", encoding="utf-8")

    # ---- slides: clip i runs from s_i - F/2 to s_{i+1} + F/2 so the picture changes at s_{i+1}, mid-fade
    starts = slide_times(sb["slides"], lyrics, duration)
    if starts[0] is None:
        starts, photos = [0.0] + starts[1:], photos[1:]
    n = len(starts)
    lengths = []
    for i in range(n):
        a = 0.0 if i == 0 else starts[i] - F / 2
        b = duration if i == n - 1 else starts[i + 1] + F / 2
        lengths.append(b - a)
    filters, inputs = [], []
    for i, (photo, L) in enumerate(zip(photos, lengths)):
        inputs += ["-i", str(photo)]
        filters.append(ken_burns(i, int(round(L * FPS)), zoom_in=(i % 2 == 0)))
    chain = "[v0]"
    for i in range(1, n):
        filters.append(f"{chain}[v{i}]xfade=transition=fade:duration={F}:offset={starts[i] - F / 2:.3f}[x{i}]")
        chain = f"[x{i}]"
    filters.append(f"{chain}subtitles={ass.name}[v]")      # relative name: the filter chokes on Windows drive-letter colons
    graph = out / f"_{name}.filters.txt"
    graph.write_text(";\n".join(filters), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats", *inputs, "-i", mp3.name, "-filter_complex_script", graph.name,
           "-map", "[v]", "-map", f"{n}:a", "-t", f"{duration:.2f}", "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", f"{name}.mp4"]
    for i, (photo, t) in enumerate(zip(photos, starts)):
        print(f"  slide {i:2d} at {t:6.1f}s  {photo.name}")
    subprocess.run(cmd, check=True, cwd=out)
    print(f"{len(cues)} subtitle lines -> {out / (name + '.srt')}")
    print(f"video -> {out / (name + '.mp4')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
