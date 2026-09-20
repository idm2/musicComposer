import subprocess

import pytest

sf = pytest.importorskip("soundfile")
from songcomposer.analysis import stems  # noqa: E402


def test_separate_is_cached_and_mixes_harmonic(tmp_path, synth_song, monkeypatch):
    """No GPU: fake the demucs subprocess by writing four stems where demucs would."""
    import numpy as np
    runs = []

    def fake_run(args, **kw):
        runs.append(args)
        assert args[1:3] == ["-m", "demucs"] and "-n" in args and str(synth_song.mix) == args[-1]
        out = tmp_path_out(args) / stems.MODEL / synth_song.mix.stem
        out.mkdir(parents=True)
        for name, level in (("vocals", 0.0), ("drums", 0.1), ("bass", 0.2), ("other", 0.3)):
            sf.write(str(out / f"{name}.wav"), np.full((1000, 2), level, dtype="float32"), 44100)
        return subprocess.CompletedProcess(args, 0, "", "")

    def tmp_path_out(args):
        from pathlib import Path
        return Path(args[args.index("-o") + 1])

    monkeypatch.setattr(stems.subprocess, "run", fake_run)
    got = stems.separate(synth_song.mix, tmp_path)
    data, sr = sf.read(str(got.harmonic))
    assert sr == 44100 and data[0, 0] == pytest.approx(0.5, abs=1e-3)       # bass 0.2 + other 0.3
    assert all(p.exists() for p in (got.vocals, got.drums, got.bass, got.other))
    stems.separate(synth_song.mix, tmp_path)
    assert len(runs) == 1                                                    # second call: cache hit


@pytest.mark.gpu
def test_real_demucs_separates_the_synth(tmp_path, synth_song):
    got = stems.separate(synth_song.mix, tmp_path)
    assert abs(sf.info(str(got.harmonic)).duration - 38.4) < 0.1
