"""Seconds → musical position, using the measured beat grid (never an assumed constant tempo)."""
from bisect import bisect_right
from statistics import median

from ..models import Analysis

NOTATABLE = (16, 12, 8, 6, 4, 3, 2, 1)       # in sixteenths: whole, dotted half, half, dotted quarter, quarter, …


class BeatMap:
    def __init__(self, beats: list[float], downbeats: list[float], beats_per_bar: int):
        if len(beats) < 2:
            raise ValueError("need at least two beats to build a grid")
        self.beats, self.bpb = list(beats), beats_per_bar
        self.interval = median(b - a for a, b in zip(beats, beats[1:]))
        first = downbeats[0] if downbeats else beats[0]
        self.offset = min(range(len(beats)), key=lambda i: abs(beats[i] - first))     # beat index where bar 0 starts

    @classmethod
    def from_analysis(cls, a: Analysis) -> "BeatMap":
        bpb = int(a.global_info.time_signature.split("/")[0]) if a.global_info else 4
        return cls(a.beats, a.downbeats, bpb)

    @property
    def bar16(self) -> int:
        return self.bpb * 4

    def _beat_float(self, t: float) -> float:
        b = self.beats
        if t <= b[0]:
            return (t - b[0]) / self.interval
        if t >= b[-1]:
            return len(b) - 1 + (t - b[-1]) / self.interval
        i = bisect_right(b, t) - 1
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def sixteenth(self, t: float) -> int:
        return max(0, round((self._beat_float(t) - self.offset) * 4))

    def beat_index(self, t: float) -> int:
        return max(0, round(self._beat_float(t) - self.offset))


def split_sixteenths(n: int) -> list[int]:
    out = []
    while n > 0:
        piece = next(p for p in NOTATABLE if p <= n)
        out.append(piece)
        n -= piece
    return out


def split_at_bars(start16: int, len16: int, bar16: int) -> list[tuple[int, int]]:
    out = []
    while len16 > 0:
        room = bar16 - start16 % bar16
        take = min(room, len16)
        out.append((start16, take))
        start16, len16 = start16 + take, len16 - take
    return out
