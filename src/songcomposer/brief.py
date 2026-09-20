"""Stage: brief. Inline text, a markdown file, or a one-liner → work/<song>/02-brief.json."""
from pathlib import Path

from .jsonio import write_model
from .models import Brief
from .paths import SongPaths


def run_brief(song: str, from_file: str | None = None, text: str | None = None) -> Brief:
    if (from_file is None) == (text is None):
        raise ValueError("give exactly one of --from <file> or --text \"...\"")
    if from_file is not None:
        body = Path(from_file).read_text(encoding="utf-8-sig")     # utf-8-sig tolerates a BOM
        origin = str(from_file)
    else:
        body, origin = text, "inline"
    body = body.strip()
    if len(body) < 10:
        raise ValueError(f"brief is too short ({len(body)} chars) — say what the song is about")
    brief = Brief(text=body, origin=origin)
    paths = SongPaths(song)
    write_model(paths.brief, brief)
    print(f"  brief ({len(body)} chars) → {paths.brief}")
    return brief
