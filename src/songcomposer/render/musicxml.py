"""<song>.musicxml — melody, lyrics and chord symbols. Opens in MuseScore and Guitar Pro."""
from pathlib import Path

from ..models import LOW_CONFIDENCE, Transcription
from .leadsheet import chord_events, melody_events, selected_melody
from .timing import BeatMap

KIND = {"maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented", "5": "power", "1": "power",
        "7": "dominant", "maj7": "major-seventh", "min7": "minor-seventh", "minmaj7": "major-minor",
        "dim7": "diminished-seventh", "hdim7": "half-diminished", "6": "major-sixth", "maj6": "major-sixth",
        "min6": "minor-sixth", "9": "dominant-ninth", "maj9": "major-ninth", "min9": "minor-ninth",
        "11": "dominant-11th", "min11": "minor-11th", "13": "dominant-13th", "maj13": "major-13th", "min13": "minor-13th",
        "sus2": "suspended-second", "sus4": "suspended-fourth"}


def _m21_name(name: str) -> str:
    return name[0] + name[1:].replace("b", "-")                    # music21 spells flats with '-'


def write_musicxml(title: str, t: Transcription, dest: Path) -> None:
    from music21 import expressions, harmony, key, metadata, meter, note, stream, tempo
    a = t.analysis
    beatmap = BeatMap.from_analysis(a)
    part = stream.Part()
    g = a.global_info
    if g:
        part.insert(0, meter.TimeSignature(g.time_signature))
        part.insert(0, tempo.MetronomeMark(number=round(g.tempo_bpm)))
        tonic, _, mode = g.key.partition(" ")
        if mode in ("major", "minor"):
            part.insert(0, key.Key(_m21_name(tonic), mode))
    melody = melody_events(t, beatmap)
    for e in melody:
        if e.pitch is None:
            el = note.Rest(quarterLength=e.len16 / 4)
        else:
            el = note.Note(e.pitch, quarterLength=e.len16 / 4)
            if e.lyric:
                el.addLyric(e.lyric)
            if e.low:
                el.notehead = "x"                                   # visibly different: low-confidence pitch
        part.insert(e.start16 / 4, el)
    chords = chord_events(t, beatmap)
    # music21 only builds measures over the melody's note/rest content, so a chord symbol sitting
    # past the last melody note has nowhere to land and is silently dropped on write. Pad with a
    # trailing rest so every chord symbol falls inside a real measure.
    melody_end = sum(e.len16 for e in melody) / 4
    if chords:
        chords_end = max(e.start_beat + e.beats for e in chords)
        if chords_end > melody_end:
            part.insert(melody_end, note.Rest(quarterLength=chords_end - melody_end))
    for e in chords:
        if e.chord is None:
            continue
        kw = {"root": _m21_name(e.chord.root), "kind": KIND.get(e.chord.quality, "major")}
        if e.chord.bass:
            kw["bass"] = _m21_name(e.chord.bass)
        part.insert(e.start_beat, harmony.ChordSymbol(**kw))
        if e.chord.confidence < LOW_CONFIDENCE or e.chord.quality not in KIND:
            part.insert(e.start_beat, expressions.TextExpression("?"))
    _, kept, total = selected_melody(t)
    score = stream.Score()
    score.insert(0, metadata.Metadata(
        title=title,
        composer=f"Song Composer — transcribed from take {t.take} — melody: {kept} of {total} notes shown"))
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(dest))
