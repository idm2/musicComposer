"""Small, pure fusion steps between the raw components and the final Analysis."""
from bisect import bisect_left

from ..models import Chord


def _nearest(beats: list[float], t: float) -> float:
    i = bisect_left(beats, t)
    return min(beats[max(0, i - 1):i + 1], key=lambda b: abs(b - t))


def snap_to_beats(chords: list[Chord], beats: list[float], tolerance: float = 0.12) -> list[Chord]:
    if not beats or not chords:
        return chords
    ends = [c.onset + c.duration for c in chords]
    onsets = [b if abs((b := _nearest(beats, c.onset)) - c.onset) <= tolerance else c.onset for c in chords]
    out = []
    for i, c in enumerate(chords):
        end = onsets[i + 1] if i + 1 < len(chords) and abs(chords[i + 1].onset - ends[i]) < 0.05 else ends[i]
        out.append(c.model_copy(update={"onset": round(onsets[i], 3), "duration": round(max(0.0, end - onsets[i]), 3)}))
    return out


def merge_adjacent(chords: list[Chord]) -> list[Chord]:
    out: list[Chord] = []
    for c in chords:
        if out and out[-1].harte == c.harte and abs(out[-1].onset + out[-1].duration - c.onset) < 0.05:
            p = out[-1]
            total = p.duration + c.duration
            conf = (p.confidence * p.duration + c.confidence * c.duration) / total if total else p.confidence
            out[-1] = p.model_copy(update={"duration": round(total, 3), "confidence": round(conf, 3)})
        else:
            out.append(c)
    return out
