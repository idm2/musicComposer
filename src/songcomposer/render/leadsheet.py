"""Quantise the transcription onto the beat grid once, for both engraved outputs (MusicXML and LilyPond)
AND for tab.py's plain-text melody — one selection, shared, so every renderer describes the same music."""
from dataclasses import dataclass

from ..models import LOW_CONFIDENCE, Chord, Note, Transcription
from .timing import BeatMap

LYRIC_SNAP_S = 0.2
MELODY_MIN_CONFIDENCE = 0.3                 # below this, a "vocal" note is more likely a transcription artefact
MELODY_SPAN_PAD_S = 0.5                     # widen each sung line's span by this much on each side


@dataclass
class MelodyEvent:
    start16: int
    len16: int
    pitch: int | None
    low: bool
    lyric: str | None


@dataclass
class ChordEvent:
    start_beat: int
    beats: int
    chord: Chord | None


def selected_melody(t: Transcription) -> tuple[list[Note], int, int]:
    """The vocal notes worth engraving: above the confidence floor, and — when we know where lines
    were sung — inside a sung line's span padded on each side. Below the floor a "vocal" note is more
    likely a transcription artefact than something to notate; outside every sung span it is almost
    always basic-pitch picking up an instrumental section. Returns (kept notes, kept count, total count).
    Shared by every renderer (tab.txt, MusicXML, LilyPond) so they can never disagree about the melody."""
    all_vocals = sorted((n for n in t.analysis.notes if n.stem == "vocals"), key=lambda n: n.onset)
    total = len(all_vocals)
    conf_ok = [n for n in all_vocals if n.confidence >= MELODY_MIN_CONFIDENCE]
    timed = [line for line in t.lines if line.start is not None]
    if timed:
        spans = [(line.start - MELODY_SPAN_PAD_S, line.end + MELODY_SPAN_PAD_S) for line in timed]
        melody = [n for n in conf_ok if any(start <= n.onset <= end for start, end in spans)]
    else:
        # no lyric line has timing at all: we cannot clip to sung spans, so keep every note that
        # clears the confidence floor rather than silently dropping everything.
        melody = conf_ok
    return melody, len(melody), total


def melody_events(t: Transcription, beatmap: BeatMap) -> list[MelodyEvent]:
    notes, _, _ = selected_melody(t)
    words = [w for line in t.lines for w in line.words]
    placed: list[MelodyEvent] = []
    for n in notes:
        start = beatmap.sixteenth(n.onset)
        end = max(start + 1, beatmap.sixteenth(n.onset + n.duration))
        if placed and start < placed[-1].start16 + placed[-1].len16:
            if start <= placed[-1].start16:
                continue                                           # two notes on one slot: keep the first
            placed[-1].len16 = start - placed[-1].start16
        near = [w for w in words if abs(w.start - n.onset) <= LYRIC_SNAP_S]
        lyric = min(near, key=lambda w: abs(w.start - n.onset)).word if near else None
        if lyric and any(e.lyric == lyric and abs(e.start16 - start) <= 2 for e in placed):
            lyric = None
        placed.append(MelodyEvent(start, end - start, n.pitch, n.confidence < LOW_CONFIDENCE, lyric))
    out: list[MelodyEvent] = []
    cursor = 0
    for e in placed:
        if e.start16 > cursor:
            out.append(MelodyEvent(cursor, e.start16 - cursor, None, False, None))
        out.append(e)
        cursor = e.start16 + e.len16
    pad = -cursor % beatmap.bar16
    if pad:
        out.append(MelodyEvent(cursor, pad, None, False, None))
    return out


def chord_events(t: Transcription, beatmap: BeatMap) -> list[ChordEvent]:
    out: list[ChordEvent] = []
    cursor = 0
    for c in sorted(t.analysis.chords, key=lambda c: c.onset):
        start = max(cursor, beatmap.beat_index(c.onset))
        end = max(start + 1, beatmap.beat_index(c.onset + c.duration))
        if start > cursor:
            out.append(ChordEvent(cursor, start - cursor, None))
        out.append(ChordEvent(start, end - start, c))
        cursor = end
    return out
