import shutil

import pytest

from songcomposer.render import songsheet
from test_musicxml import ch, t  # noqa: F401 — reuse the fixture and chord helper


def test_songsheet_html_has_diagrams_sections_and_chord_rows(t):  # noqa: F811
    page = songsheet.build_html("Glass Hour", t)
    assert "<h1>Glass Hour</h1>" in page and "standard tuning, no capo" in page
    assert page.count("<svg") == len({c.symbol for c in t.analysis.chords})
    assert '<div class="ch">' in page and '<div class="ly">' in page and "<h3>" in page


def test_diagram_svg_draws_muted_open_and_fretted_strings():
    svg = songsheet.diagram_svg("D", ["x", "x", "0", "2", "3", "2"])
    assert svg.count("×") == 2 and svg.count('class="open"') == 1 and svg.count("<circle") == 4


@pytest.mark.skipif(songsheet.find_browser() is None, reason="no Chrome/Edge")
def test_songsheet_pdf_is_printed(t, tmp_path):  # noqa: F811
    assert songsheet.write_songsheet("Glass Hour", t, tmp_path / "s.html", tmp_path / "s.pdf") is True
    assert (tmp_path / "s.pdf").read_bytes().startswith(b"%PDF")
