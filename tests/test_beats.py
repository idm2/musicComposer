import subprocess
import sys

import pytest

from songcomposer.analysis.beats import summarise_beats, track_beats


def test_tempo_and_meter_from_a_clean_grid():
    beats = [i * 0.6 for i in range(32)]
    g = summarise_beats(beats, beats[::4])
    assert g.tempo_bpm == pytest.approx(100.0) and g.time_signature == "4/4"


def test_waltz():
    beats = [i * 0.5 for i in range(30)]
    assert summarise_beats(beats, beats[::3]).time_signature == "3/4"


def test_median_ignores_a_dropped_beat():
    beats = [i * 0.6 for i in range(32)]
    del beats[10]
    assert summarise_beats(beats, beats[::4]).tempo_bpm == pytest.approx(100.0)


def test_no_beats_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no beat"):
        summarise_beats([0.5], [])


def test_importing_beats_does_not_import_torch():
    """Architectural pin: analyze() will import every component module, including beats,
    even for an LLM-only --ears subjective run that has no business touching torch. Run in
    a subprocess so the current session's already-imported torch cannot mask a regression."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.beats; "
         "assert 'torch' not in sys.modules, 'beats imported torch eagerly'; "
         "assert 'beat_this' not in sys.modules, 'beats imported beat_this eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.gpu
def test_beat_this_recovers_the_synth_tempo(tmp_path, synth_song):
    g = track_beats(synth_song.mix, tmp_path)
    assert abs(g.tempo_bpm - synth_song.truth["tempo_bpm"]) < 2.0
    assert g.time_signature == "4/4"
    assert abs(g.downbeats[1] - g.downbeats[0] - 2.4) < 0.1
