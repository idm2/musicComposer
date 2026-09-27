import shutil

import pytest

from songcomposer.render import lilypond
from test_musicxml import ch, t  # noqa: F401 — reuse the fixture and chord helper


@pytest.mark.parametrize("midi,flats,expected", [(60, False, "c'"), (48, False, "c"), (72, False, "c''"), (40, False, "e,"),
                                                 (61, False, "cis'"), (61, True, "des'"), (70, True, "bes'"), (63, True, "ees'")])
def test_ly_pitch(midi, flats, expected):
    assert lilypond.ly_pitch(midi, flats) == expected


@pytest.mark.parametrize("harte,symbol,beats,expected", [
    ("C:maj", "C", 4, "c1"), ("A:min", "Am", 2, "a2:m"), ("Bb:min7", "Bbm7", 3, "bes2.:m7"), ("F#:dim", "F#dim", 1, "fis4:dim"),
    ("D:sus4", "Dsus4", 4, "d1:sus4"), ("B:hdim7", "Bm7b5", 4, "b1:m7.5-"), ("C:maj7", "Cmaj7", 4, "c1:maj7")])
def test_ly_chord(harte, symbol, beats, expected):
    assert lilypond.ly_chord(ch(harte, symbol, 0, 1), beats) == expected


def test_ly_chord_with_bass():
    c = ch("C:maj", "C/E", 0, 1).model_copy(update={"bass": "E"})
    assert lilypond.ly_chord(c, 4) == "c1/e"


def test_build_ly_has_every_layer_and_marks_doubt(t):  # noqa: F811
    ly = lilypond.build_ly('Glass "Hour"', t)
    assert 'title = "Glass \\"Hour\\""' in ly
    for needle in ("\\new ChordNames", "\\fret-diagram-terse", "\\new TabStaff", "\\lyricsto", "\\key c \\major",
                   "\\time 4/4", "\\tempo 4 = 100", "chordChanges = ##t"):
        assert needle in ly, needle
    assert "r4 e'4 \\parenthesize g'2" in ly                        # rest, sure note, low-confidence note
    assert '"Glass" "hour"' in ly
    assert "c1 s1 \\once \\override ChordName.color = #grey bes1:m7" in ly


@pytest.mark.skipif(shutil.which("lilypond") is None, reason="lilypond not installed")
def test_pdf_is_produced(t, tmp_path):  # noqa: F811
    assert lilypond.write_pdf("Glass Hour", t, tmp_path / "chart.ly", tmp_path / "chart.pdf") is True
    assert (tmp_path / "chart.pdf").read_bytes().startswith(b"%PDF")


def test_missing_lilypond_is_reported_not_fatal(t, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(lilypond.shutil, "which", lambda name: None)
    assert lilypond.write_pdf("T", t, tmp_path / "chart.ly", tmp_path / "chart.pdf") is False
    assert (tmp_path / "chart.ly").exists() and "scoop install lilypond" in capsys.readouterr().out


def test_importing_lilypond_does_not_import_heavy_deps():
    """Architectural pin: render/ is pure Python — no torch/librosa/numpy/music21 at module scope,
    anywhere in the import chain. Run in a subprocess so the current session's already-imported
    modules cannot mask a regression (pattern: tests/test_tab.py)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.render.lilypond; "
         "heavy = {'torch', 'librosa', 'numpy', 'music21'}; "
         "leaked = heavy & set(sys.modules); "
         "assert not leaked, f'render.lilypond imported {leaked} eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_fret_diagrams_use_the_tab_shapes_once_above_the_music(t):  # noqa: F811
    """LilyPond's own fret calculation picked odd voicings, and a box over every change cluttered the page:
    each chord's diagram appears once, above the score, with the same shape the tab shows."""
    ly = lilypond.build_ly("T", t)
    assert '\\bold "C" \\fret-diagram-terse #"x;3;2;o;1;o;"' in ly
    assert ly.count('\\bold "C" ') == 1
    assert ly.index("fret-diagram-terse") < ly.index("\\score")
    assert "FretBoards" not in ly


@pytest.mark.parametrize("harte,symbol,bass,expected", [
    ("D:maj", "D", None, "x;x;0;2;3;2;"), ("A:maj", "A/C#", "C#", "x;4;2;2;2;0;"),
    ("D:maj", "D/F#", "F#", "2;x;0;2;3;2;"), ("B:min7", "Bm7", None, "x;2;4;2;3;2;"), ("F#:min7", "F#m7", None, "2;4;2;2;2;2;")])
def test_diagram_frets(harte, symbol, bass, expected):
    c = ch(harte, symbol, 0, 1).model_copy(update={"bass": bass})
    assert lilypond.diagram_frets(c) == expected
