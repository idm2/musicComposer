"""Guitar-specific knowledge. This is the ONLY package allowed to know what a fret or a capo is."""
from collections import Counter

from ..chordsym import parse_harte, pitch_class, transpose_name
from ..models import Chord, Note
from .timing import BeatMap

OPEN_FRIENDLY = {"C", "D", "E", "G", "A", "Am", "Dm", "Em", "A7", "B7", "C7", "D7", "E7", "G7", "Cmaj7", "Fmaj7",
                 "Am7", "Dm7", "Em7", "Dsus2", "Dsus4", "Asus2", "Asus4", "Esus4", "Cadd9"}
MAX_CAPO = 7
CAPO_PENALTY = 0.02

TUNING = (40, 45, 50, 55, 59, 64)            # E2 A2 D3 G3 B3 E4
MAX_FRET = 15
SHAPES = {
    "C": "x32010", "D": "xx0232", "E": "022100", "F": "133211", "G": "320003", "A": "x02220", "B": "x24442",
    "Am": "x02210", "Dm": "xx0231", "Em": "022000", "Bm": "x24432",
    "A7": "x02020", "B7": "x21202", "C7": "x32310", "D7": "xx0212", "E7": "020100", "G7": "320001",
    "Cmaj7": "x32000", "Dmaj7": "xx0222", "Fmaj7": "xx3210", "Gmaj7": "320002", "Amaj7": "x02120",
    "Am7": "x02010", "Dm7": "xx0211", "Em7": "022030",
    "Dsus2": "xx0230", "Dsus4": "xx0233", "Asus2": "x02200", "Asus4": "x02230", "Esus4": "022200", "Cadd9": "x32030",
}
_BARRE = {"maj": (0, 2, 2, 1, 0, 0), "min": (0, 2, 2, 0, 0, 0), "7": (0, 2, 0, 1, 0, 0), "min7": (0, 2, 0, 0, 0, 0),
          "maj7": (0, 2, 1, 1, 0, 0), "sus4": (0, 2, 2, 2, 0, 0)}


def transpose_harte(harte: str, semitones: int) -> str:
    root, sep, rest = harte.partition(":")
    bare_root, slash, bass = root.partition("/")
    new_root = transpose_name(bare_root, semitones, prefer_flats="b" in bare_root)
    return f"{new_root}{slash}{bass}{sep}{rest}"


def _shape_symbol(chord: Chord, capo: int) -> str:
    parsed = parse_harte(transpose_harte(chord.harte, -capo))
    return parsed.symbol if parsed else chord.symbol


def suggest_capo(chords: list[Chord]) -> tuple[int, dict[str, str]]:
    total = sum(c.duration for c in chords)
    if not total:
        return 0, {}

    def friendliness(capo: int) -> float:
        good = sum(c.duration for c in chords if _shape_symbol(c, capo).split("/")[0] in OPEN_FRIENDLY)
        return good / total - CAPO_PENALTY * capo

    best = max(range(MAX_CAPO + 1), key=friendliness)
    if best == 0 or friendliness(best) <= friendliness(0) + 0.1:        # only suggest a capo when it clearly helps
        return 0, {}
    return best, {c.symbol: _shape_symbol(c, best) for c in chords}


def shape_for(chord: Chord) -> str | None:
    name = chord.symbol.split("/")[0]
    if name in SHAPES:
        return SHAPES[name]
    if chord.quality not in _BARRE:
        return None
    fret = (pitch_class(chord.root) - 4) % 12                       # root on the low E string
    frets = [fret + d for d in _BARRE[chord.quality]]
    return ".".join(map(str, frets)) if max(frets) >= 10 else "".join(map(str, frets))


def into_range(pitch: int) -> int:
    while pitch < TUNING[0]:
        pitch += 12
    while pitch > TUNING[-1] + MAX_FRET:
        pitch -= 12
    return pitch


def fret_positions(pitches: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    hand = 0
    for pitch in pitches:
        pitch = into_range(pitch)
        options = [(s, pitch - open_) for s, open_ in enumerate(TUNING) if 0 <= pitch - open_ <= MAX_FRET]
        best = min(options, key=lambda sf: (abs(sf[1] - hand) if sf[1] else 0.5, sf[1]))
        out.append(best)
        if best[1]:
            hand = best[1]
    return out


def strum_onsets(notes: list[Note]) -> list[float]:
    starts = sorted(n.onset for n in notes if n.stem == "other")
    out, i = [], 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] - starts[i] <= 0.04:
            j += 1
        if j - i + 1 >= 3:
            out.append(starts[i])
        i = j + 1
    return out


def bar_pattern(onsets: list[float], bar_start16: int, beatmap: BeatMap) -> str:
    hit = {(beatmap.sixteenth(o) - bar_start16) // 2 for o in onsets
           if bar_start16 <= beatmap.sixteenth(o) < bar_start16 + beatmap.bar16 and (beatmap.sixteenth(o) - bar_start16) % 2 == 0}
    return "".join(("D" if slot % 2 == 0 else "U") if slot in hit else "-" for slot in range(beatmap.bar16 // 2))


MIN_PATTERN_HITS = 3           # a pattern must place at least this many strokes in its 8-slot bar to count as "strummed"
MIN_PATTERN_COVERAGE = 0.5     # ...and be the detected pattern in at least this fraction of the section's played bars


def common_pattern(patterns: list[str], min_hits: int = 0, min_coverage: float = 0.0) -> str | None:
    """The most common non-empty bar pattern, or None if there isn't one — or, when thresholds are given,
    None unless that pattern also clears a minimum stroke count and a minimum share of the played bars.
    A single stray onset (one stroke in one bar out of many) must never be reported as "strummed"."""
    played = [p for p in patterns if p.strip("-")]
    if not played:
        return None
    pattern, count = Counter(played).most_common(1)[0]
    hits = len(pattern) - pattern.count("-")
    if hits < min_hits or count / len(played) < min_coverage:
        return None
    return pattern
