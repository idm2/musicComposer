import pytest

from songcomposer.brief import run_brief
from songcomposer.jsonio import read_json
from songcomposer.paths import SongPaths


def test_inline_text(root):
    b = run_brief("demo", text="A song about leaving a coastal town at dawn.")
    assert b.origin == "inline"
    assert read_json(SongPaths("demo").brief)["text"].startswith("A song about")


def test_markdown_file_with_bom_and_smart_quotes(root):
    f = root / "brief.md"
    f.write_bytes("﻿# Brief\n\nShe said “don’t go” — café lights.\n".encode("utf-8"))
    b = run_brief("demo", from_file=str(f))
    assert b.text.startswith("# Brief") and "“don’t go” — café" in b.text
    assert b.origin == str(f)


@pytest.mark.parametrize("kw", [{}, {"text": "x", "from_file": "y.md"}])
def test_exactly_one_source_required(root, kw):
    with pytest.raises(ValueError, match="exactly one"):
        run_brief("demo", **kw)


def test_too_short_rejected(root):
    with pytest.raises(ValueError, match="too short"):
        run_brief("demo", text="sad")
