import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """An empty project root; cwd is moved there so SongPaths defaults resolve inside it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sine_wav(tmp_path) -> Path:
    out = tmp_path / "sine.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ar", "22050", "-ac", "1", str(out)],
        check=True,
    )
    return out
