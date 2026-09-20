"""Song form. Boundaries are MEASURED (bar-synchronous features, agglomerative segmentation);
labels are HEARD (the subjective ear's sections). Confidence = how well the two agree."""
from pathlib import Path

from ..models import Section, SectionGuess

SECONDS_PER_SEGMENT = 25


def boundaries(harmonic: Path, downbeats: list[float], duration_s: float) -> list[float]:
    if len(downbeats) < 4:
        return [0.0, duration_s]
    import librosa
    import numpy as np
    y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.util.normalize(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop), axis=1)
    bars = librosa.util.fix_frames(librosa.time_to_frames(downbeats, sr=sr, hop_length=hop), x_min=0, x_max=chroma.shape[1])
    feats = np.vstack([librosa.util.sync(chroma, bars, aggregate=np.median), librosa.util.sync(mfcc, bars)])
    k = int(min(feats.shape[1], max(2, min(12, round(duration_s / SECONDS_PER_SEGMENT)))))
    starts = librosa.segment.agglomerative(feats, k)
    times = librosa.frames_to_time(bars[starts], sr=sr, hop_length=hop)
    inner = sorted({round(float(t), 3) for t in times if 0.5 < t < duration_s - 0.5})
    return [0.0] + inner + [duration_s]


def label_sections(bounds: list[float], guesses: list[SectionGuess]) -> list[Section]:
    out: list[Section] = []
    for i, (start, end) in enumerate(zip(bounds, bounds[1:]), start=1):
        best, cover = None, 0.0
        for g in guesses:
            shared = max(0.0, min(end, g.end) - max(start, g.start))
            if shared > cover:
                best, cover = g, shared
        label = best.label if best else f"part {i}"
        conf = round(cover / (end - start), 3) if best and end > start else 0.0
        if out and best and out[-1].label == label:
            prev = out[-1]
            total = end - prev.start
            merged = (prev.confidence * (prev.end - prev.start) + conf * (end - start)) / total
            out[-1] = Section(label=label, start=prev.start, end=end, confidence=round(merged, 3))
        else:
            out.append(Section(label=label, start=start, end=end, confidence=conf))
    return out
