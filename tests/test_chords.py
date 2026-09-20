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
    assert chords.score(1.0, 1.0, rival=0.0) == 1.0 and chords.score(0.0, 0.0, rival=0.0) == 0.0
    assert chords.score(0.8, 0.0, rival=0.0) < 0.6 < chords.score(0.8, 1.0, rival=0.0)
    assert chords.score(0.4, 1.0, rival=0.0) == pytest.approx(0.4)    # support at the floor contributes nothing


def test_score_pins_the_disagreement_edge():
    """I2: perfect support that also clearly beats every other root outranks one dissenting pass —
    strong direct spectral evidence is allowed to outweigh a single disagreeing cross-pass."""
    assert chords.score(1.0, 0.0, rival=0.0) == 0.6
    # Same perfect support and total disagreement, but now out-fitted by a rival on another root:
    # the margin evidence and the agreement evidence agree the claim is shaky, so it is halved again.
    assert chords.score(1.0, 0.0, rival=1.05) == 0.3


# --- C1: the competing-hypothesis margin --------------------------------------------------------

def test_relative_major_minor_confusion_is_not_confident_even_when_both_passes_agree():
    """The exact blind spot the review found: relative major/minor triads share 2 of 3 pitch
    classes, so a claim that loses to the true chord on a different root must not clear
    LOW_CONFIDENCE just because both lv-chordia passes made the same mistake (agreement 1.0)."""
    c_major = chroma(0, 4, 7)
    support = chords.spectral_support(c_major, chord_pitch_classes(parse_harte("A:min")))
    rival = chords.best_rival_support(c_major, "A")
    assert chords.score(support, agree=1.0, rival=rival) < 0.5

    a_minor = chroma(9, 0, 4)
    support = chords.spectral_support(a_minor, chord_pitch_classes(parse_harte("C:maj")))
    rival = chords.best_rival_support(a_minor, "C")
    assert chords.score(support, agree=1.0, rival=rival) < 0.5


def test_the_right_answer_keeps_its_confidence_despite_a_rival_search():
    c_major = chroma(0, 4, 7)
    support = chords.spectral_support(c_major, chord_pitch_classes(parse_harte("C:maj")))
    rival = chords.best_rival_support(c_major, "C")
    assert chords.score(support, agree=1.0, rival=rival) >= 0.9

    cmaj7 = chroma(0, 4, 7, 11)
    support = chords.spectral_support(cmaj7, chord_pitch_classes(parse_harte("C:maj7")))
    rival = chords.best_rival_support(cmaj7, "C")
    assert chords.score(support, agree=1.0, rival=rival) >= 0.8

    am7 = chroma(9, 0, 4, 7)
    support = chords.spectral_support(am7, chord_pitch_classes(parse_harte("A:min7")))
    rival = chords.best_rival_support(am7, "A")   # C major must not out-fit it
    assert chords.score(support, agree=1.0, rival=rival) >= 0.8


def test_same_root_colour_is_not_a_rival():
    """Cmaj7 audio "claimed" as plain C:maj: the best-fitting DIFFERENT root (E minor, which is
    exactly Cmaj7's upper structure) ties the claim's own support rather than crushing it the way
    the same-root C:maj7 reading would (0.9975 vs 0.8639) — proving the same-root exclusion is
    doing its job. The tie still leaves the chord at or above LOW_CONFIDENCE."""
    cmaj7 = chroma(0, 4, 7, 11)
    claim_support = chords.spectral_support(cmaj7, chord_pitch_classes(parse_harte("C:maj")))
    same_root_support = chords.spectral_support(cmaj7, chord_pitch_classes(parse_harte("C:maj7")))
    rival = chords.best_rival_support(cmaj7, "C")
    assert rival < same_root_support           # excluding root C really did drop the best same-root fit
    assert chords.score(claim_support, agree=1.0, rival=rival) >= 0.5


def test_best_rival_support_excludes_the_claimed_root_regardless_of_spelling():
    heard = chroma(0, 4, 7, 11)
    assert chords.best_rival_support(heard, "Db") == chords.best_rival_support(heard, "C#")


def test_margin_factor_boundaries():
    assert chords.margin_factor(0.5, 0.55) == 0.0     # margin == MARGIN_LOW
    assert chords.margin_factor(0.55, 0.5) == 1.0      # margin == MARGIN_HIGH
    assert chords.margin_factor(0.5, 0.5) == 0.5       # tied → midpoint


# --- I3: the short-segment path --------------------------------------------------------------

def test_segment_chroma_on_a_too_short_segment_is_zero_not_nan():
    frame_chroma = np.ones((12, 3))
    times = np.array([1.0, 1.01, 1.02])
    mean = chords.segment_chroma(frame_chroma, times, onset=1.0, end=1.02)   # narrower than the 0.05s guard bands
    assert not np.isnan(mean).any()
    assert (mean == 0.0).all()


def test_a_chord_from_a_zero_chroma_vector_is_not_confident():
    zero = np.zeros(12)
    support = chords.spectral_support(zero, chord_pitch_classes(parse_harte("C:maj")))
    rival = chords.best_rival_support(zero, "C")
    assert chords.score(support, agree=1.0, rival=rival) < 0.5


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
