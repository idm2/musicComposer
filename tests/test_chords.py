import pytest

from songcomposer.chordsym import chord_pitch_classes, parse_harte

np = pytest.importorskip("numpy")
from songcomposer.analysis import chords  # noqa: E402


@pytest.mark.parametrize("label,pcs", [
    ("C:maj", {0, 4, 7}), ("A:min", {9, 0, 4}), ("C:maj7", {0, 4, 7, 11}), ("A:min7", {9, 0, 4, 7}),
    ("G:7", {7, 11, 2, 5}), ("D:sus4", {2, 7, 9}), ("D:sus2", {2, 4, 9}), ("B:hdim7", {11, 2, 5, 9}),
    ("C:maj(9)", {0, 4, 7, 2}), ("C:weird", {0, 4, 7}),
])
def test_chord_pitch_classes(label, pcs):
    assert chord_pitch_classes(parse_harte(label)) == pcs


def chroma(*pcs, floor=0.05):
    v = np.full(12, floor)
    v[list(pcs)] = 1.0
    return v


def test_support_is_high_when_the_spectrum_matches_and_low_when_it_does_not():
    c_major = chroma(0, 4, 7)
    assert chords.spectral_support(c_major, {0, 4, 7}) > 0.9
    assert chords.spectral_support(c_major, {6, 10, 1}) < 0.2          # F# major claimed over C major audio


def test_support_separates_cmaj7_from_am():
    """The exact confusion CLAUDE.md warns about: the evidence must prefer the right one."""
    heard = chroma(0, 4, 7, 11)                                        # C E G B
    assert chords.spectral_support(heard, {0, 4, 7, 11}) > chords.spectral_support(heard, {9, 0, 4})


def test_support_of_silence_is_zero():
    assert chords.spectral_support(np.zeros(12), {0, 4, 7}) == 0.0


def test_agreement_is_the_overlap_fraction_with_same_root_and_quality():
    other = [{"start_time": 0.0, "end_time": 3.0, "chord": "C:maj"}, {"start_time": 3.0, "end_time": 8.0, "chord": "A:min"}]
    assert chords.agreement(0.0, 4.0, "C", "maj", other) == pytest.approx(0.75)
    assert chords.agreement(0.0, 4.0, "G", "maj", other) == 0.0
    assert chords.agreement(0.0, 4.0, "C", "maj", []) == 0.0


def test_score_bounds_and_weights():
    assert chords.score(1.0, 1.0) == 1.0 and chords.score(0.0, 0.0) == 0.0
    assert chords.score(0.8, 0.0) < 0.6 < chords.score(0.8, 1.0)
    assert chords.score(0.4, 1.0) == pytest.approx(0.4)               # support at the floor contributes nothing


def test_importing_chords_does_not_import_heavy_deps():
    """Architectural pin: analyze() will import every component module, including chords,
    even for an LLM-only --ears subjective run that has no business touching these. Run in
    a subprocess so the current session's already-imported modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.chords; "
         "assert 'torch' not in sys.modules, 'chords imported torch eagerly'; "
         "assert 'lv_chordia' not in sys.modules, 'chords imported lv_chordia eagerly'; "
         "assert 'librosa' not in sys.modules, 'chords imported librosa eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.gpu
def test_lv_chordia_recovers_the_synth_progression_with_confidence(tmp_path, synth_song):
    got = chords.detect_chords(synth_song.harmonic, synth_song.mix, tmp_path)
    truth = synth_song.truth["chords"]

    def label_at(t):
        return next((c.harte for c in got if c.onset <= t < c.onset + c.duration), None)

    probes = [(a + b) / 2 for a, b, _ in truth]
    assert [label_at(t) for t in probes] == [lab for _, _, lab in truth]
    assert all(c.confidence >= 0.5 for c in got), [(c.symbol, c.confidence) for c in got]
    assert all(0.0 <= c.confidence <= 1.0 for c in got)
