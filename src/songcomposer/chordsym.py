"""Harte chord-label parsing (the 'C:min7/b7' syntax used by MIR chord recognisers).

Instrument-neutral: knows pitch names and chord qualities, nothing about guitars.
"""
import re
from dataclasses import dataclass, field

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Harte quality → display suffix
_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "7": "7", "maj7": "maj7", "min7": "m7", "minmaj7": "m(maj7)",
    "dim7": "dim7", "hdim7": "m7b5",
    "6": "6", "maj6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9", "11": "11", "min11": "m11", "13": "13", "maj13": "maj13", "min13": "m13",
    "sus2": "sus2", "sus4": "sus4", "5": "5", "1": "5",
}
# Harte interval → semitones above root
_INTERVAL = {"1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7,
             "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3,
             "11": 5, "#11": 6, "b13": 8, "13": 9}
_LABEL_RE = re.compile(r"^([A-G][#b]?)(?::([^/()]*)(?:\(([^)]*)\))?)?(?:/(.+))?$")


@dataclass
class ParsedChord:
    root: str
    quality: str
    extensions: list[str] = field(default_factory=list)
    bass: str | None = None
    symbol: str = ""


def pitch_class(name: str) -> int:
    pc = _NATURAL[name[0]]
    for acc in name[1:]:
        pc += 1 if acc == "#" else -1
    return pc % 12


def transpose_name(name: str, semitones: int, prefer_flats: bool) -> str:
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    return names[(pitch_class(name) + semitones) % 12]


def _prefers_flats(root: str) -> bool:
    return "b" in root[1:] or root == "F"


def parse_harte(label: str) -> ParsedChord | None:
    label = (label or "").strip()
    if label in ("", "N", "X"):
        return None
    m = _LABEL_RE.match(label)
    if not m:
        return None
    root, quality, ext, bass_iv = m.group(1), m.group(2), m.group(3), m.group(4)
    quality = quality or "maj"
    extensions = [e.strip() for e in ext.split(",")] if ext else []
    bass = None
    if bass_iv and bass_iv in _INTERVAL and _INTERVAL[bass_iv] != 0:
        bass = transpose_name(root, _INTERVAL[bass_iv], _prefers_flats(root))
    if quality in _SUFFIX:
        suffix = _SUFFIX[quality]
    else:
        suffix = f"({quality})"          # unknown quality: show it verbatim, never guess
    adds = "".join(f"add{e}" for e in extensions if not e.startswith("*"))
    symbol = f"{root}{suffix}{adds}" + (f"/{bass}" if bass else "")
    return ParsedChord(root=root, quality=quality, extensions=extensions, bass=bass, symbol=symbol)


_QUALITY_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "5": (0, 7), "1": (0,),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "6": (0, 4, 7, 9), "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 2), "maj9": (0, 4, 7, 11, 2), "min9": (0, 3, 7, 10, 2),
    "11": (0, 4, 7, 10, 2, 5), "min11": (0, 3, 7, 10, 2, 5),
    "13": (0, 4, 7, 10, 2, 9), "maj13": (0, 4, 7, 11, 2, 9), "min13": (0, 3, 7, 10, 2, 9),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}


def chord_pitch_classes(parsed: ParsedChord) -> set[int]:
    """Pitch classes sounding in the chord. Unknown qualities fall back to a major triad on the root
    (in which case any confidence computed against this template is unreliable by construction —
    it is being checked against a guess, not the recogniser's actual claim)."""
    root = pitch_class(parsed.root)
    intervals = set(_QUALITY_INTERVALS.get(parsed.quality, (0, 4, 7)))
    for ext in parsed.extensions:
        if ext.startswith("*"):
            intervals.discard(_INTERVAL.get(ext[1:], -1))
        elif ext in _INTERVAL:
            intervals.add(_INTERVAL[ext])
    return {(root + i) % 12 for i in intervals}
