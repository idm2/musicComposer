"""Gather every take of every variation of a song into ONE listening folder.

    uv run python tools/collect-variations.py <prefix>

Each variation is its own pipeline project (work/<prefix>, work/<prefix>-v2, work/<prefix>-v6-a-rnb ...), because
each has its own spec and takes. This copies all their takes into out/<prefix>-variations/ under descriptive names
and writes index.md beside them: title, style prompt and lyrics for each variation. Copies, never moves — `pick`
and `chart` still run against the work/ projects. Takes deleted from the listening folder are not copied back.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    args = ap.parse_args()

    projects = sorted(p for p in (ROOT / "work").glob(f"{args.prefix}*") if (p / "04-takes").is_dir())
    if not projects:
        print(f"error: no work/{args.prefix}* project has takes", file=sys.stderr)
        return 1
    dest = ROOT / "out" / f"{args.prefix}-variations"
    dest.mkdir(parents=True, exist_ok=True)
    seen_path = dest / ".collected.json"                     # so a take you deleted here stays deleted
    seen = set(json.loads(seen_path.read_text(encoding="utf-8"))) if seen_path.exists() else set()

    index, copied = [f"# {args.prefix} — all variations\n"], 0
    for proj in projects:
        variation = proj.name[len(args.prefix):].lstrip("-") or "v1"
        spec = json.loads((proj / "03-spec.json").read_text(encoding="utf-8"))
        slug = "".join(c if c.isalnum() else "-" for c in spec["title"]).strip("-")
        takes = sorted((proj / "04-takes").glob("take-*.mp3"))
        names = []
        for take in takes:
            name = f"{variation}_{take.stem}_{slug}.mp3"
            names.append(name)
            if name not in seen:
                shutil.copyfile(take, dest / name)
                seen.add(name)
                copied += 1
        index += [f"\n## {variation} — {spec['title']}\n", f"Project: `work/{proj.name}`  \nTakes: " + ", ".join(f"`{n}`" for n in names),
                  f"\n**Style:** {spec['style_prompt']}\n"]
        for s in spec["sections"]:
            if s["lines"]:
                index += [f"\n*{s['name']}*  "] + [f"{ln}  " for ln in s["lines"]]
    (dest / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    seen_path.write_text(json.dumps(sorted(seen), indent=1), encoding="utf-8")
    print(f"{copied} new take(s) copied -> {dest}  ({len(projects)} variations, index.md written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
