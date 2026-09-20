"""Copy a song's small, permanent artefacts from work/ and out/ into the versioned songs/ folder.

    uv run python tools/archive-song.py <song> [--force]

work/ and out/ are git-ignored (big, regenerable). songs/<song>/ is versioned and is what survives.
Audio is never copied; takes.json records enough to trace or regenerate a take.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source path relative to work/<song>, destination name in songs/<song>)
FROM_WORK = [
    ("brief.md", "brief.source.md"),
    ("00-source.json", "00-source.json"),
    ("01-analysis.json", "01-analysis.json"),
    ("02-brief.json", "02-brief.json"),
    ("03-spec.json", "03-spec.json"),
    ("06-transcription.json", "06-transcription.json"),
    ("05-chosen.json", "05-chosen.json"),
    ("adhoc-objective.json", "adhoc-objective.json"),
    ("04-takes/takes.json", "takes.json"),
]
FROM_OUT = ["chords.txt", "tab.txt", "lyrics.json", "chart.ly"]   # plus <song>.md and <song>.musicxml


def scrub(path: Path) -> None:
    """takes.json holds the exact provider request; drop anything that could carry a secret."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for run in data.get("runs", []):
        payload = run.get("payload") or {}
        for holder in (payload, payload.get("body") or {}, (payload.get("body") or {}).get("input") or {}):
            for key in ("callBackUrl", "callback_url", "apiKey", "api_key", "authorization"):
                holder.pop(key, None)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("--force", action="store_true", help="overwrite files already archived")
    args = ap.parse_args()

    work, out = ROOT / "work" / args.song, ROOT / "out" / args.song
    if not work.exists():
        print(f"error: {work} does not exist — nothing to archive", file=sys.stderr)
        return 1
    dest = ROOT / "songs" / args.song
    dest.mkdir(parents=True, exist_ok=True)

    copied, skipped = [], []
    pairs = [(work / src, dest / name) for src, name in FROM_WORK]
    pairs += [(out / n, dest / n) for n in FROM_OUT]
    pairs += [(out / f"{args.song}.md", dest / f"{args.song}.md"),
              (out / f"{args.song}.musicxml", dest / f"{args.song}.musicxml")]
    for src, target in pairs:
        if not src.exists():
            continue
        if target.exists() and not args.force:
            skipped.append(target.name)
            continue
        shutil.copyfile(src, target)
        copied.append(target.name)
        if target.name == "takes.json":
            scrub(target)

    print(f"archived {len(copied)} file(s) -> {dest}")
    for name in copied:
        print(f"  {name}")
    if skipped:
        print(f"  ({len(skipped)} already present, use --force to overwrite: {', '.join(skipped)})")
    if not (dest / f"{args.song}.md").exists():
        print("  note: no song sheet yet — write songs/<song>/<song>.md (see songs/in-the-flow/ for the shape)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
