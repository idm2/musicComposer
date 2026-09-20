"""Guitar-specific knowledge. This is the ONLY package allowed to know what a fret or a capo is."""
from ..chordsym import parse_harte, transpose_name
from ..models import Chord

OPEN_FRIENDLY = {"C", "D", "E", "G", "A", "Am", "Dm", "Em", "A7", "B7", "C7", "D7", "E7", "G7", "Cmaj7", "Fmaj7",
                 "Am7", "Dm7", "Em7", "Dsus2", "Dsus4", "Asus2", "Asus4", "Esus4", "Cadd9"}
MAX_CAPO = 7
CAPO_PENALTY = 0.02


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
