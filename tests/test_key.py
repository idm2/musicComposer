import pytest

np = pytest.importorskip("numpy")
from songcomposer.analysis import key  # noqa: E402


def test_major_and_minor_profiles_are_recovered():
    assert key.estimate_key(key.MAJOR)[0] == "C major"
    assert key.estimate_key(np.roll(key.MINOR, 9))[0] == "A minor"
    assert key.estimate_key(np.roll(key.MAJOR, 10))[0] == "Bb major"


def test_flat_chroma_is_unknown_not_a_guess():
    assert key.estimate_key(np.ones(12)) == ("unknown", 0.0)
    assert 0.5 < key.estimate_key(key.MAJOR)[1] <= 1.0


def test_measure_on_the_synth(tmp_path, synth_song):
    pytest.importorskip("librosa")
    pytest.importorskip("pyloudnorm")
    got = key.measure(synth_song.harmonic, synth_song.mix, tmp_path)
    assert got["key"] == "C major"
    assert abs(got["duration_s"] - 38.4) < 0.05
    assert -40 < got["loudness_lufs"] < 0


def test_importing_key_does_not_import_heavy_deps():
    """Architectural pin: analyze() will import every component module, including key,
    even for an LLM-only --ears subjective run that has no business touching these. Run in
    a subprocess so the current session's already-imported modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.key; "
         "assert 'numpy' not in sys.modules, 'key imported numpy eagerly'; "
         "assert 'librosa' not in sys.modules, 'key imported librosa eagerly'; "
         "assert 'pyloudnorm' not in sys.modules, 'key imported pyloudnorm eagerly'; "
         "assert 'soundfile' not in sys.modules, 'key imported soundfile eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
