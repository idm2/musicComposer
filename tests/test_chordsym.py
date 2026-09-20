import pytest

from songcomposer.chordsym import parse_harte, pitch_class, transpose_name


@pytest.mark.parametrize("label,symbol,root,quality,bass", [
    ("C:maj", "C", "C", "maj", None),
    ("A:min", "Am", "A", "min", None),
    ("A:min7", "Am7", "A", "min7", None),
    ("C:maj7", "Cmaj7", "C", "maj7", None),
    ("G:7", "G7", "G", "7", None),
    ("D:sus4", "Dsus4", "D", "sus4", None),
    ("D:sus2", "Dsus2", "D", "sus2", None),
    ("B:hdim7", "Bm7b5", "B", "hdim7", None),
    ("F#:dim", "F#dim", "F#", "dim", None),
    ("Bb:maj", "Bb", "Bb", "maj", None),
    ("C:maj/3", "C/E", "C", "maj", "E"),
    ("A:min/b7", "Am/G", "A", "min", "G"),
    ("Bb:maj/5", "Bb/F", "Bb", "maj", "F"),
    ("G", "G", "G", "maj", None),              # bare root = major triad
])
def test_parse_harte(label, symbol, root, quality, bass):
    c = parse_harte(label)
    assert (c.symbol, c.root, c.quality, c.bass) == (symbol, root, quality, bass)


def test_cmaj7_and_am_are_distinct():
    """The librosa failure mode named in CLAUDE.md must be representable as two different chords."""
    assert parse_harte("C:maj7").symbol != parse_harte("A:min").symbol


def test_parenthesised_extensions_kept():
    c = parse_harte("C:maj(9)")
    assert c.quality == "maj" and c.extensions == ["9"] and c.symbol == "Cadd9"


@pytest.mark.parametrize("label", ["N", "X", ""])
def test_no_chord_is_none(label):
    assert parse_harte(label) is None


def test_unknown_quality_is_preserved_not_guessed():
    c = parse_harte("C:weird")
    assert c.quality == "weird" and c.symbol == "C(weird)"


def test_pitch_class_and_transpose():
    assert pitch_class("C") == 0 and pitch_class("F#") == 6 and pitch_class("Bb") == 10
    assert transpose_name("A", 3, prefer_flats=False) == "C"
    assert transpose_name("A", 1, prefer_flats=True) == "Bb"
    assert transpose_name("A", 1, prefer_flats=False) == "A#"
