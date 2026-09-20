"""Chord recognition: lv-chordia names the chords; we measure how far the spectrum backs each one up.

librosa appears here ONLY as a measuring instrument for confidence. It never names a chord.

Confidence is built from three independent signals, none of which is the recogniser's own opinion:
  1. spectral support   — does the claimed chord's pitch-class template fit the measured chroma?
  2. cross-pass agreement — do the harmonic-stem and full-mix passes name the same root+quality?
  3. competing-hypothesis margin — does some chord on a DIFFERENT root fit the spectrum even better?
     Without (3), a relative major/minor confusion (they share 2 of 3 pitch classes) can look
     well-supported and be agreed on by both passes (the same algorithm run twice on correlated
     audio), yet still be the wrong chord entirely.
"""
from pathlib import Path
from typing import Sequence

from ..chordsym import chord_pitch_classes, parse_harte, pitch_class
from ..hashing import file_sha1
from ..models import Chord
from .cache import cached

VERSION = "1"
SUPPORT_FLOOR, SUPPORT_CEIL = 0.4, 0.9       # cosine similarity range mapped onto 0..1
W_SUPPORT, W_AGREE = 0.6, 0.4
MARGIN_LOW, MARGIN_HIGH = -0.05, 0.05        # support-minus-rival range mapped onto the margin factor

_TRIAD_QUALITIES = ((0, 4, 7), (0, 3, 7))    # major, minor — the only qualities a rival is judged as


def spectral_support(chroma_mean: Sequence[float], pitch_classes: set[int]) -> float:
    import numpy as np
    v = np.asarray(chroma_mean, dtype=float)
    template = np.zeros(12)
    template[list(pitch_classes)] = 1.0
    denom = np.linalg.norm(v) * np.linalg.norm(template)
    return float(v @ template / denom) if denom > 0 else 0.0


def best_rival_support(chroma_mean: Sequence[float], claimed_root: str) -> float:
    """The best-fitting major/minor triad on a root OTHER than the claimed one. Same-root
    alternatives (C vs Cmaj7 vs Csus4) are deliberately excluded: a root error breaks a chart,
    a colour error does not, and the recogniser's quality vocabulary is richer than any triad
    contest we could stage."""
    claimed_pc = pitch_class(claimed_root)
    candidates = [
        spectral_support(chroma_mean, {(root_pc + i) % 12 for i in intervals})
        for root_pc in range(12) if root_pc != claimed_pc
        for intervals in _TRIAD_QUALITIES
    ]
    return max(candidates) if candidates else 0.0


def margin_factor(support: float, rival: float) -> float:
    margin = support - rival
    if margin <= MARGIN_LOW:
        return 0.0
    if margin >= MARGIN_HIGH:
        return 1.0
    return (margin - MARGIN_LOW) / (MARGIN_HIGH - MARGIN_LOW)


def agreement(onset: float, end: float, root: str, quality: str, other_pass: list[dict]) -> float:
    if end <= onset:
        return 0.0
    shared = 0.0
    for seg in other_pass:
        parsed = parse_harte(seg["chord"])
        if parsed and (parsed.root, parsed.quality) == (root, quality):
            shared += max(0.0, min(end, seg["end_time"]) - max(onset, seg["start_time"]))
    return min(1.0, shared / (end - onset))


def score(support: float, agree: float, rival: float) -> float:
    scaled = min(1.0, max(0.0, (support - SUPPORT_FLOOR) / (SUPPORT_CEIL - SUPPORT_FLOOR)))
    base = W_SUPPORT * scaled + W_AGREE * agree
    # A chord out-fitted by a different root loses half its confidence; one that clearly beats
    # every other root keeps all of it. Silence (support 0, rival 0) gives factor 0.5, but base
    # is already <= W_AGREE there, so it never gets pulled above LOW_CONFIDENCE by this alone.
    confidence = base * (0.5 + 0.5 * margin_factor(support, rival))
    return round(confidence, 3)


def segment_chroma(chroma, times, onset: float, end: float):
    """Mean chroma vector for the frames strictly inside [onset, end], with a 0.05 s guard band
    on each side (segment boundaries from the recogniser are approximate). A zero vector — never
    NaN — comes back when the segment is too short to leave any frames inside the guard bands."""
    import numpy as np
    frames = (times >= onset + 0.05) & (times < end - 0.05)
    return chroma[:, frames].mean(axis=1) if frames.any() else np.zeros(chroma.shape[0])


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
        mean = segment_chroma(chroma, times, onset, end)
        support = spectral_support(mean, chord_pitch_classes(parsed))
        rival = best_rival_support(mean, parsed.root)
        conf = score(support,
                     agreement(onset, end, parsed.root, parsed.quality, second),
                     rival)
        out.append(Chord(symbol=parsed.symbol, harte=seg["chord"], root=parsed.root, quality=parsed.quality,
                         extensions=parsed.extensions, bass=parsed.bass, onset=round(onset, 3),
                         duration=round(end - onset, 3), confidence=conf))
    return out
