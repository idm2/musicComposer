"""Beat and downbeat tracking with beat_this (CPJKU, 2024). dbn=False: the DBN post-processor needs madmom,
which cannot be built on this machine."""
from collections import Counter
from pathlib import Path
from statistics import median

from pydantic import BaseModel

from ..hashing import file_sha1
from .cache import cached

VERSION = "1"


class BeatGrid(BaseModel):
    beats: list[float]
    downbeats: list[float]
    tempo_bpm: float
    time_signature: str


def summarise_beats(beats: list[float], downbeats: list[float]) -> BeatGrid:
    if len(beats) < 2:
        raise ValueError("no beat found — cannot derive tempo")
    tempo = 60.0 / median(b - a for a, b in zip(beats, beats[1:]))
    counts = [sum(1 for b in beats if lo - 0.02 <= b < hi - 0.02) for lo, hi in zip(downbeats, downbeats[1:])]
    per_bar = Counter(counts).most_common(1)[0][0] if counts else 4
    per_bar = max(2, min(12, per_bar))          # a degenerate run must never propagate a "1/4" or "40/4" time signature
    return BeatGrid(beats=[round(b, 4) for b in beats], downbeats=[round(d, 4) for d in downbeats],
                    tempo_bpm=round(tempo, 2), time_signature=f"{per_bar}/4")


def track_beats(audio: Path, cache_dir: Path) -> BeatGrid:
    def work() -> dict:
        import torch
        from beat_this.inference import File2Beats
        device = "cuda" if torch.cuda.is_available() else "cpu"
        beats, downbeats = File2Beats(checkpoint_path="final0", device=device, dbn=False)(str(Path(audio).resolve()))
        return summarise_beats([float(b) for b in beats], [float(d) for d in downbeats]).model_dump()

    return BeatGrid(**cached(cache_dir, file_sha1(audio), "beats", VERSION, work))
