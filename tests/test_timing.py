import pytest

from songcomposer.render.timing import BeatMap, split_at_bars, split_sixteenths

BEATS = [0.5 + i * 0.6 for i in range(16)]                 # first downbeat at 0.5 s


def test_sixteenth_positions():
    m = BeatMap(BEATS, BEATS[::4], 4)
    assert m.bar16 == 16
    assert m.sixteenth(0.5) == 0 and m.sixteenth(1.1) == 4 and m.sixteenth(0.5 + 0.15) == 1
    assert m.sixteenth(0.5 + 2.4) == 16                     # bar 2
    assert m.sixteenth(0.0) == 0                            # before the grid clamps to 0
    assert m.sixteenth(BEATS[-1] + 0.6) == 64               # extrapolates past the last beat
    assert m.beat_index(1.75) == 2


def test_pickup_beats_before_the_first_downbeat():
    m = BeatMap(BEATS, [BEATS[2]], 4)
    assert m.sixteenth(BEATS[2]) == 0 and m.sixteenth(BEATS[3]) == 4


@pytest.mark.parametrize("n,parts", [(16, [16]), (5, [4, 1]), (7, [6, 1]), (11, [8, 3]), (1, [1]), (13, [12, 1])])
def test_split_sixteenths(n, parts):
    assert split_sixteenths(n) == parts


def test_split_at_bars():
    assert split_at_bars(14, 6, 16) == [(14, 2), (16, 4)]
    assert split_at_bars(0, 40, 16) == [(0, 16), (16, 16), (32, 8)]
