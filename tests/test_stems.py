import subprocess
import sys

import pytest

sf = pytest.importorskip("soundfile")
from songcomposer.analysis import stems  # noqa: E402


def test_separate_is_cached_and_mixes_harmonic(tmp_path, synth_song, monkeypatch):
    """No GPU: fake the demucs subprocess by writing four stems where demucs would.

    `stems.subprocess` IS the global `subprocess` module (not a copy), so patching
    `.run` intercepts every subprocess call made anywhere in the process for the
    lifetime of the monkeypatch — not just the Demucs invocation `separate()` makes.
    In particular, `_device()`'s lazy `import torch` can itself shell out on Windows
    (DLL-loading calls `platform.machine()` -> `ver`). The fake must therefore only
    handle the Demucs call and delegate everything else to the real `subprocess.run`.
    """
    import numpy as np
    runs = []
    real_run = subprocess.run

    def fake_run(args, **kw):
        if args[1:3] != ["-m", "demucs"]:
            return real_run(args, **kw)
        runs.append(args)
        assert "-n" in args and str(synth_song.mix) == args[-1]
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


def test_importing_stems_does_not_import_torch():
    """Architectural pin: analyze() will import every component module, including stems,
    even for an LLM-only --ears subjective run that has no business touching torch. Run in
    a subprocess so the current session's already-imported torch cannot mask a regression."""
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.stems; "
         "assert 'torch' not in sys.modules, 'stems imported torch eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.gpu
def test_real_demucs_separates_the_synth(tmp_path, synth_song):
    got = stems.separate(synth_song.mix, tmp_path)
    assert abs(sf.info(str(got.harmonic)).duration - 38.4) < 0.1
