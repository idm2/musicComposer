import pytest

from songcomposer.accuracy import parse_lab, score_chords
from songcomposer.models import Chord


def ch(harte, onset, dur, conf):
    root, _, q = harte.partition(":")
    return Chord(symbol=harte, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


def test_parse_lab_skips_blanks_and_no_chord():
    assert parse_lab("0.0 2.0 C:maj\n\n2.0 4.0 N\n4.0 6.0 A:min7\n") == [(0.0, 2.0, "C:maj"), (4.0, 6.0, "A:min7")]


def test_scores_are_duration_weighted():
    truth = [(0.0, 4.0, "C:maj"), (4.0, 8.0, "A:min7")]
    found = [ch("C:maj", 0, 4, 0.9), ch("A:min", 4, 2, 0.9), ch("F:maj", 6, 2, 0.2)]
    s = score_chords(found, truth)
    assert s["root"] == pytest.approx(0.75)                  # C right (4s) + A root right (2s) of 8s
    assert s["full"] == pytest.approx(0.5)                   # A:min ≠ A:min7
    assert s["high_confidence_wrong"] == pytest.approx(0.0)  # the wrong-root F was honest about its doubt
