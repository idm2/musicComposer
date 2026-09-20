"""Demucs stem separation — run FIRST. Chord detection on a full mix is fighting percussion;
splitting out drums and vocals is the single biggest accuracy gain available (BUILD-SPEC §4)."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..hashing import content_key, file_sha1

MODEL = "htdemucs"
VERSION = "1"
NAMES = ("vocals", "drums", "bass", "other")


class Stems(BaseModel):
    vocals: Path
    drums: Path
    bass: Path
    other: Path
    harmonic: Path


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def separate(audio: Path, cache_dir: Path) -> Stems:
    audio = Path(audio).resolve()
    out = Path(cache_dir) / f"stems-{content_key(file_sha1(audio), MODEL, VERSION)}"
    result = Stems(**{n: out / f"{n}.wav" for n in NAMES}, harmonic=out / "harmonic.wav")
    if all(p.exists() for p in result.model_dump().values()):
        return result
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-m", "demucs", "-n", MODEL, "-d", _device(), "-o", tmp, str(audio)],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"demucs failed: {proc.stderr[-800:]}")
        produced = Path(tmp) / MODEL / audio.stem
        for n in NAMES:
            shutil.move(str(produced / f"{n}.wav"), str(out / f"{n}.wav"))
    import soundfile as sf
    bass, sr = sf.read(str(result.bass))
    other, _ = sf.read(str(result.other))
    n = min(len(bass), len(other))
    sf.write(str(result.harmonic), bass[:n] + other[:n], sr, subtype="PCM_16")
    return result
