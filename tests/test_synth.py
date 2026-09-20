import pytest

sf = pytest.importorskip("soundfile")


def test_synth_matches_its_own_truth(synth_song):
    info = sf.info(str(synth_song.mix))
    assert info.samplerate == 44100 and abs(info.duration - 38.4) < 0.01
    t = synth_song.truth
    assert (t["tempo_bpm"], t["key"], t["time_signature"]) == (100, "C major", "4/4")
    assert [c[2] for c in t["chords"][:4]] == ["C:maj", "A:min", "F:maj", "G:maj"]
    assert t["chords"][1][0] == pytest.approx(4.8)                 # two bars per chord
    assert len(t["melody"]) == 64 and t["melody"][0] == (0.0, 72)
    assert synth_song.harmonic.exists() and synth_song.melody.exists()


def test_render_is_deterministic():
    import numpy as np
    import synth
    assert np.array_equal(synth.render(synth.ALL_PARTS), synth.render(synth.ALL_PARTS))
