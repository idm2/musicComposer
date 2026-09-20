"""ffmpeg / ffprobe helpers. ffmpeg is installed via scoop and on PATH."""
import subprocess
from pathlib import Path


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, encoding="utf-8", errors="replace").stdout
    except FileNotFoundError as e:
        raise RuntimeError(f"{args[0]} not found on PATH — install with `scoop install ffmpeg`") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{args[0]} failed ({e.returncode}): {e.stderr[-400:]}") from e


def to_wav(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
          "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)])


def to_mp3(src: Path, dst: Path, bitrate: str = "128k", mono: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn"]
    if mono:
        args += ["-ac", "1"]
    _run(args + ["-b:a", bitrate, str(dst)])


def probe_duration(path: Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.strip())
