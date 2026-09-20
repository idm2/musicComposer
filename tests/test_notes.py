import pytest

from songcomposer.analysis import notes
from songcomposer.models import Note


def n(pitch, onset, dur, conf):
    return Note(pitch=pitch, onset=onset, duration=dur, confidence=conf, stem="vocals")


def test_monophonic_keeps_the_more_confident_of_two_overlapping_notes():
    got = notes.to_monophonic([n(60, 0.0, 1.0, 0.9), n(72, 0.1, 0.9, 0.3), n(62, 1.0, 0.5, 0.8)])
    assert [(x.pitch, x.onset) for x in got] == [(60, 0.0), (62, 1.0)]


def test_monophonic_trims_a_tail_that_runs_into_the_next_note():
    got = notes.to_monophonic([n(60, 0.0, 1.0, 0.9), n(62, 0.8, 0.6, 0.9)])
    assert got[0].duration == pytest.approx(0.8) and got[1].onset == 0.8


def test_basic_pitch_recovers_the_synth_melody(tmp_path, synth_song):
    pytest.importorskip("basic_pitch")
    got = notes.to_monophonic(notes.transcribe_stem(synth_song.melody, "vocals", tmp_path))
    truth = synth_song.truth["melody"]
    hits = sum(1 for onset, pitch in truth
               if any(abs(g.onset - onset) < 0.08 and g.pitch == pitch for g in got))
    assert hits / len(truth) >= 0.85, f"{hits}/{len(truth)} melody notes recovered"
    assert all(0.0 <= g.confidence <= 1.0 for g in got)


def test_importing_notes_does_not_import_heavy_deps():
    """Architectural pin: analyze() will import every component module, including notes,
    even for an LLM-only --ears subjective run that has no business touching these. Run in
    a subprocess so the current session's already-imported modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.notes; "
         "assert 'basic_pitch' not in sys.modules, 'notes imported basic_pitch eagerly'; "
         "assert 'tensorflow' not in sys.modules, 'notes imported tensorflow eagerly'; "
         "assert 'torch' not in sys.modules, 'notes imported torch eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
