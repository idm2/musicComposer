import pytest

from songcomposer.models import Analysis, Chord, GlobalInfo, LyricLine, Note, Transcription, Word
from songcomposer.render.leadsheet import chord_events, melody_events
from songcomposer.render.timing import BeatMap

BEATS = [i * 0.6 for i in range(16)]


def ch(harte, symbol, onset, dur, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


@pytest.fixture
def t():
    notes = [Note(pitch=64, onset=0.6, duration=0.6, confidence=0.9, stem="vocals"),
             Note(pitch=67, onset=1.2, duration=1.2, confidence=0.3, stem="vocals"),
             Note(pitch=40, onset=0.0, duration=2.0, confidence=0.9, stem="bass")]
    words = [Word(word="Glass", start=0.62, end=1.1, confidence=0.9), Word(word="hour", start=1.2, end=2.0, confidence=0.9)]
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=notes,
                 chords=[ch("C:maj", "C", 0.0, 2.4), ch("Bb:min7", "Bbm7", 4.8, 2.4, conf=0.2)],
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=9.6))
    return Transcription(take=1, analysis=a, lines=[LyricLine(section="V", text="Glass hour", start=0.62, end=2.0, words=words, confidence=0.9)])


def test_melody_events_are_contiguous_with_rests_and_lyrics(t):
    ev = melody_events(t, BeatMap(BEATS, BEATS[::4], 4))
    assert [(e.start16, e.len16, e.pitch, e.low, e.lyric) for e in ev] == [
        (0, 4, None, False, None), (4, 4, 64, False, "Glass"), (8, 8, 67, True, "hour")]
    assert sum(e.len16 for e in ev) % 16 == 0


def test_chord_events_fill_gaps_with_none(t):
    ev = chord_events(t, BeatMap(BEATS, BEATS[::4], 4))
    assert [(e.start_beat, e.beats, e.chord.symbol if e.chord else None) for e in ev] == [(0, 4, "C"), (4, 4, None), (8, 4, "Bbm7")]


def test_musicxml_roundtrip(t, tmp_path):
    m21 = pytest.importorskip("music21")
    from songcomposer.render.musicxml import write_musicxml
    dest = tmp_path / "song.musicxml"
    write_musicxml("Glass Hour", t, dest)
    score = m21.converter.parse(str(dest))
    symbols = list(score.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert [s.root().name for s in symbols] == ["C", "B-"] and symbols[1].chordKind == "minor-seventh"
    pitched = [n for n in score.recurse().notes if isinstance(n, m21.note.Note)]
    assert [n.pitch.midi for n in pitched] == [64, 67] and pitched[0].lyric == "Glass"
    # confidence must survive for a NOTE too, not just a chord: pitch 67 (confidence 0.3) is the
    # low-confidence note write_musicxml marks with notehead="x"; pitch 64 (confidence 0.9) must not be.
    assert pitched[1].notehead == "x" and pitched[0].notehead != "x"
    assert "?" in [e.content for e in score.recurse().getElementsByClass(m21.expressions.TextExpression)]
    # music21's MusicXML importer deletes metadata.title when it equals movementName (it assumes the
    # writer duplicated a work-title into movement-title, which is exactly what our writer does) —
    # bestTitle is what actually survives the round trip. See xmlToM21.MusicXMLImporter.xmlWorkToMetadata.
    assert score.metadata.bestTitle == "Glass Hour"


def test_importing_musicxml_does_not_import_music21_eagerly():
    """Architectural pin: music21 is heavy (~10s import) and must only be imported lazily, inside
    write_musicxml. Run in a subprocess so the current session's already-imported modules cannot
    mask a regression (pattern: tests/test_tab.py)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.render.musicxml; "
         "assert 'music21' not in sys.modules, 'render.musicxml imported music21 eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
