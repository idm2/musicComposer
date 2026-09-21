"""Make subtitles and a subtitled video for a finished song.

    uv run python tools/subtitle-video.py <song> [--image path/to/background.jpg]

Reads out/<song>/lyrics.json (word-timed, written by `songcomposer chart`) and out/<song>/<song>.mp3, and writes beside them:
  <song>.srt   plain subtitles (upload alongside a video anywhere)
  <song>.ass   styled subtitles
  <song>.mp4   1920x1080 video: the background image (blurred, darkened), title card, lyrics burned in

Timings come from transcribing the actual recording, so the words appear when they are sung.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLD_S, GAP_S = 1.2, 0.08            # keep a line up a little after it ends, but never into the next line


def stamp(t: float, ass: bool) -> str:
    h, rem = divmod(max(t, 0.0), 3600)
    m, s = divmod(rem, 60)
    if ass:
        return f"{int(h)}:{int(m):02d}:{s:05.2f}"
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("--image", help="background image (default: a plain dark gradient)")
    args = ap.parse_args()

    out = ROOT / "out" / args.song
    lyrics_path, mp3 = out / "lyrics.json", out / f"{args.song}.mp3"
    for p in (lyrics_path, mp3):
        if not p.exists():
            print(f"error: {p} is missing — run `songcomposer pick` and `songcomposer chart` first", file=sys.stderr)
            return 1
    data = json.loads(lyrics_path.read_text(encoding="utf-8"))
    lines = sorted((ln for ln in data["lines"] if ln.get("start") is not None), key=lambda ln: ln["start"])
    cues = []
    for ln, nxt in zip(lines, lines[1:] + [None]):
        end = ln["end"] + HOLD_S
        if nxt is not None:
            end = min(end, nxt["start"] - GAP_S)
        cues.append((ln["start"], max(end, ln["start"] + 0.5), ln["text"]))

    srt = "\n".join(f"{i}\n{stamp(a, False)} --> {stamp(b, False)}\n{text}\n" for i, (a, b, text) in enumerate(cues, start=1))
    (out / f"{args.song}.srt").write_text(srt, encoding="utf-8")

    title = data.get("title") or args.song
    first = cues[0][0] if cues else 3.0
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
            "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Lyric,Georgia,68,&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,0,0,0,0,100,100,0,0,1,3,2,2,140,140,150,1\n"
            "Style: Title,Georgia,112,&H00FFFFFF,&H00FFFFFF,&H00101010,&H96000000,0,1,0,0,100,100,2,0,1,3,2,5,140,140,0,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    if first > 1.2:
        events.append(f"Dialogue: 0,{stamp(0.3, True)},{stamp(first - 0.2, True)},Title,,0,0,0,,{{\\fad(600,500)}}{title}")
    for a, b, text in cues:
        events.append(f"Dialogue: 0,{stamp(a, True)},{stamp(b, True)},Lyric,,0,0,0,,{{\\fad(180,220)}}{text}")
    ass = out / f"{args.song}.ass"
    ass.write_text(head + "\n".join(events) + "\n", encoding="utf-8")

    duration = float(data.get("duration_s") or 0) or float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
        capture_output=True, text=True, check=True).stdout.strip())
    if args.image:
        bg_in = ["-loop", "1", "-i", str(Path(args.image).resolve())]
        bg = ("[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,gblur=sigma=14,"
              "eq=brightness=-0.22:saturation=0.9,format=yuv420p[bg]")
    else:
        # dusk over a calm sea: night blue -> violet -> rose -> amber at the horizon -> dark water
        # rendered once to a still: the gradients source slowly rotates, which tilts the horizon over a whole song
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "gradients=s=1920x1080:nb_colors=6:c0=0x081430:c1=0x2b2a5e:c2=0xb5567a:c3=0xf2a35e:c4=0x27334f:c5=0x0a1224"
                        ":x0=960:y0=0:x1=960:y1=1080", "-frames:v", "1", "-vf", "gblur=sigma=6", "_background.png"], check=True, cwd=out)
        bg_in = ["-loop", "1", "-i", "_background.png"]
        bg = "[0:v]format=yuv420p[bg]"
    # run inside out/<song> with a relative .ass name: the subtitles filter chokes on Windows drive-letter colons
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *bg_in, "-i", mp3.name, "-filter_complex", f"{bg};[bg]subtitles={ass.name}[v]",
           "-map", "[v]", "-map", "1:a", "-t", f"{duration:.2f}", "-r", "25", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k", "-shortest", f"{args.song}.mp4"]
    subprocess.run(cmd, check=True, cwd=out)
    print(f"{len(cues)} subtitle lines -> {out / (args.song + '.srt')}")
    print(f"video -> {out / (args.song + '.mp4')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
