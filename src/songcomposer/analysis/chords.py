"""Chord recognition: lv-chordia names the chords; we measure how far the spectrum backs each one up.

librosa appears here ONLY as a measuring instrument for confidence. It never names a chord.
"""
from pathlib import Path
from typing import Sequence

from ..chordsym import chord_pitch_classes, parse_harte
from ..hashing import file_sha1
from ..models import Chord
from .cache import cached

VERSION = "1"
SUPPORT_FLOOR, SUPPORT_CEIL = 0.4, 0.9       # cosine similarity range mapped onto 0..1
W_SUPPORT, W_AGREE = 0.6, 0.4


def spectral_support(chroma_mean: Sequence[float], pitch_classes: set[int]) -> float:
    import numpy as np
    v = np.asarray(chroma_mean, dtype=float)
    template = np.zeros(12)
    template[list(pitch_classes)] = 1.0
    denom = np.linalg.norm(v) * np.linalg.norm(template)
    return float(v @ template / denom) if denom > 0 else 0.0


def agreement(onset: float, end: float, root: str, quality: str, other_pass: list[dict]) -> float:
    if end <= onset:
        return 0.0
    shared = 0.0
    for seg in other_pass:
        parsed = parse_harte(seg["chord"])
        if parsed and (parsed.root, parsed.quality) == (root, quality):
            shared += max(0.0, min(end, seg["end_time"]) - max(onset, seg["start_time"]))
    return min(1.0, shared / (end - onset))


def score(support: float, agree: float) -> float:
    scaled = min(1.0, max(0.0, (support - SUPPORT_FLOOR) / (SUPPORT_CEIL - SUPPORT_FLOOR)))
    return round(W_SUPPORT * scaled + W_AGREE * agree, 3)


def recognise(audio: Path, cache_dir: Path) -> list[dict]:
    def work() -> list[dict]:
        from lv_chordia.chord_recognition import chord_recognition
        # ABSOLUTE path: lv-chordia resolves relative paths against its own package directory.
        return chord_recognition(str(Path(audio).resolve()), chord_dict_name="submission")

    return cached(cache_dir, file_sha1(audio), "lvchordia", VERSION, work)


def detect_chords(harmonic: Path, mix: Path, cache_dir: Path) -> list[Chord]:
    import librosa
    import numpy as np
    primary, second = recognise(harmonic, cache_dir), recognise(mix, cache_dir)
    y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    out: list[Chord] = []
    for seg in primary:
        parsed = parse_harte(seg["chord"])
        if parsed is None:                                   # "N" — no chord
            continue
        onset, end = float(seg["start_time"]), float(seg["end_time"])
        frames = (times >= onset + 0.05) & (times < end - 0.05)
        mean = chroma[:, frames].mean(axis=1) if frames.any() else np.zeros(12)
        conf = score(spectral_support(mean, chord_pitch_classes(parsed)),
                     agreement(onset, end, parsed.root, parsed.quality, second))
        out.append(Chord(symbol=parsed.symbol, harte=seg["chord"], root=parsed.root, quality=parsed.quality,
                         extensions=parsed.extensions, bass=parsed.bass, onset=round(onset, 3),
                         duration=round(end - onset, 3), confidence=conf))
    return out
