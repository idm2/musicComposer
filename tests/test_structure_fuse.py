import subprocess
import sys

import pytest

from songcomposer.analysis import fuse, structure
from songcomposer.models import Chord, SectionGuess


def ch(harte, onset, dur, conf=0.8):
    return Chord(symbol=harte.split(":")[0], harte=harte, root=harte.split(":")[0], quality="maj", extensions=[],
                 bass=None, onset=onset, duration=dur, confidence=conf)


def test_snap_moves_near_onsets_onto_the_beat_and_keeps_far_ones():
    got = fuse.snap_to_beats([ch("C:maj", 0.05, 2.35), ch("F:maj", 2.4, 1.0), ch("G:maj", 3.7, 1.1)], [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2])
    assert [c.onset for c in got] == [0.0, 2.4, 3.6]
    assert got[0].duration == pytest.approx(2.4)                     # end re-joined to the next onset


def test_snap_leaves_a_chord_between_beats_alone():
    assert fuse.snap_to_beats([ch("C:maj", 0.3, 1.0)], [0.0, 0.6, 1.2])[0].onset == 0.3


def test_merge_adjacent_weights_confidence_by_duration():
    got = fuse.merge_adjacent([ch("C:maj", 0, 3, 0.9), ch("C:maj", 3, 1, 0.5), ch("G:maj", 4, 2, 0.7)])
    assert [(c.harte, c.onset, c.duration) for c in got] == [("C:maj", 0, 4), ("G:maj", 4, 2)]
    assert got[0].confidence == pytest.approx(0.8)


def test_labels_come_from_the_heard_form_with_overlap_as_confidence():
    guesses = [SectionGuess(label="verse", start=0, end=20), SectionGuess(label="chorus", start=20, end=40)]
    got = structure.label_sections([0.0, 9.6, 19.2, 38.4], guesses)
    assert [(s.label, s.start, s.end) for s in got] == [("verse", 0.0, 19.2), ("chorus", 19.2, 38.4)]   # two verse bars merged
    assert got[0].confidence == pytest.approx(1.0) and 0.9 < got[1].confidence <= 1.0


def test_without_a_subjective_ear_sections_are_honestly_anonymous():
    got = structure.label_sections([0.0, 10.0, 20.0], [])
    assert [s.label for s in got] == ["part 1", "part 2"] and all(s.confidence == 0.0 for s in got)


def test_boundaries_are_bar_aligned_on_the_synth(synth_song):
    pytest.importorskip("librosa")
    downbeats = [i * 2.4 for i in range(16)]
    got = structure.boundaries(synth_song.harmonic, downbeats, 38.4)
    assert got[0] == 0.0 and got[-1] == 38.4 and got == sorted(got)
    assert all(min(abs(b - d) for d in downbeats + [38.4]) < 0.1 for b in got)


def test_too_few_downbeats_gives_one_segment(synth_song):
    assert structure.boundaries(synth_song.harmonic, [0.0], 38.4) == [0.0, 38.4]


def test_importing_structure_does_not_import_librosa():
    """Architectural pin, same pattern as test_beats.py: analyze() imports every component
    module, including structure, even for an --ears subjective run that never touches DSP."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.structure; "
         "assert 'librosa' not in sys.modules, 'structure imported librosa eagerly'; "
         "assert 'numpy' not in sys.modules, 'structure imported numpy eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_importing_fuse_does_not_import_librosa():
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.fuse; "
         "assert 'librosa' not in sys.modules, 'fuse imported librosa eagerly'; "
         "assert 'numpy' not in sys.modules, 'fuse imported numpy eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
