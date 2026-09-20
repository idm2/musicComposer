"""Key (Krumhansl–Schmuckler over the harmonic stem) and integrated loudness. This is librosa's proper job."""
from pathlib import Path
from typing import Sequence

from ..hashing import file_sha1
from .cache import cached

VERSION = "1"
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def estimate_key(chroma_mean: Sequence[float]) -> tuple[str, float]:
    import numpy as np
    v = np.asarray(chroma_mean, dtype=float)
    scored = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for tonic in range(12):
            scored.append((float(np.corrcoef(v, np.roll(MAJOR, tonic))[0, 1]), f"{MAJOR_NAMES[tonic]} major"))
            scored.append((float(np.corrcoef(v, np.roll(MINOR, tonic))[0, 1]), f"{MINOR_NAMES[tonic]} minor"))
    scored.sort(reverse=True)
    (best, name), (runner_up, _) = scored[0], scored[1]
    if np.isnan(best):
        return "unknown", 0.0
    confidence = max(0.0, min(1.0, best)) * max(0.2, min(1.0, (best - runner_up) / 0.15))
    return name, round(confidence, 3)


def measure(harmonic: Path, mix: Path, cache_dir: Path) -> dict:
    def work() -> dict:
        import librosa
        import pyloudnorm
        import soundfile as sf
        y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
        name, conf = estimate_key(librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1))
        data, rate = sf.read(str(mix))
        lufs = float(pyloudnorm.Meter(rate).integrated_loudness(data))
        return {"key": name, "key_confidence": conf, "loudness_lufs": round(lufs, 2),
                "duration_s": round(len(data) / rate, 3)}

    return cached(cache_dir, file_sha1(mix), "key", VERSION, work)
