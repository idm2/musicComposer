"""I1: tab.txt and the engraved outputs (MusicXML/LilyPond, via leadsheet.melody_events) must
describe the SAME melody — both consume render.leadsheet.selected_melody()."""
import re

from songcomposer.models import Analysis, GlobalInfo, LyricLine, Note, Transcription
from songcomposer.render.leadsheet import melody_events, selected_melody
from songcomposer.render.tab import render_tab
from songcomposer.render.timing import BeatMap

BEATS = [i * 0.6 for i in range(32)]


def make_analysis(notes: list[Note]) -> Analysis:
    return Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=notes,
                     global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100,
                                            time_signature="4/4", loudness_lufs=-14, duration_s=19.2))


def _tab_kept_total(out: str) -> tuple[int, int]:
    m = re.search(r"(\d+) of (\d+) transcribed vocal notes selected for the melody line", out)
    assert m, out
    return int(m.group(1)), int(m.group(2))


def test_tab_and_leadsheet_agree_on_the_same_melody():
    below_floor = Note(pitch=60, onset=1.0, duration=0.3, confidence=0.05, stem="vocals")   # basic-pitch artefact
    outside_span = Note(pitch=62, onset=5.0, duration=0.3, confidence=0.9, stem="vocals")    # never sung
    kept_one = Note(pitch=64, onset=10.5, duration=0.3, confidence=0.9, stem="vocals")
    kept_two = Note(pitch=67, onset=10.8, duration=0.3, confidence=0.9, stem="vocals")
    line = LyricLine(section="Verse 1", text="la la", start=10.0, end=11.0, words=[], confidence=0.9)
    t = Transcription(take=1, analysis=make_analysis([below_floor, outside_span, kept_one, kept_two]), lines=[line])

    melody, kept, total = selected_melody(t)
    assert (total, kept) == (4, 2)
    assert {n.pitch for n in melody} == {64, 67}

    beatmap = BeatMap.from_analysis(t.analysis)
    events = melody_events(t, beatmap)
    assert {e.pitch for e in events if e.pitch is not None} == {64, 67}

    assert _tab_kept_total(render_tab("T", t)) == (2, 4)


def test_note_below_the_floor_and_note_outside_every_sung_span_are_absent_from_both():
    below_floor = Note(pitch=60, onset=10.2, duration=0.3, confidence=0.2, stem="vocals")   # inside span, low conf
    outside_span = Note(pitch=61, onset=0.0, duration=0.3, confidence=0.95, stem="vocals")  # confident but never sung
    line = LyricLine(section="Verse 1", text="la", start=10.0, end=11.0, words=[], confidence=0.9)
    t = Transcription(take=1, analysis=make_analysis([below_floor, outside_span]), lines=[line])

    melody, kept, total = selected_melody(t)
    assert (total, kept) == (2, 0) and melody == []

    beatmap = BeatMap.from_analysis(t.analysis)
    events = melody_events(t, beatmap)
    assert all(e.pitch is None for e in events)

    assert _tab_kept_total(render_tab("T", t)) == (0, 2)


def test_no_timed_lines_keeps_every_note_above_the_floor_in_both():
    below_floor = Note(pitch=60, onset=1.0, duration=0.3, confidence=0.1, stem="vocals")
    above_floor = Note(pitch=64, onset=2.0, duration=0.3, confidence=0.9, stem="vocals")
    line = LyricLine(section="Verse 1", text="never sung", start=None, end=None, words=[], confidence=0.0)
    t = Transcription(take=1, analysis=make_analysis([below_floor, above_floor]), lines=[line])

    melody, kept, total = selected_melody(t)
    assert (total, kept) == (2, 1) and melody[0].pitch == 64

    beatmap = BeatMap.from_analysis(t.analysis)
    events = melody_events(t, beatmap)
    assert {e.pitch for e in events if e.pitch is not None} == {64}

    assert _tab_kept_total(render_tab("T", t)) == (1, 2)


def test_disclosure_reports_the_selected_count_even_when_fewer_are_actually_emitted():
    """Follow-up finding on I1: melody_events() drops a note whose quantised slot collides with the
    previous one, and tab.py's ASCII grid can overwrite a colliding slot with a later note — so what
    each renderer actually EMITS can be fewer than what selected_melody() SELECTED. The shared
    disclosure must keep reporting the selection count (the number genuinely identical everywhere),
    never an emitted count that would silently re-diverge between renderers."""
    a_note = Note(pitch=64, onset=10.00, duration=0.3, confidence=0.9, stem="vocals")
    b_note = Note(pitch=67, onset=10.02, duration=0.3, confidence=0.9, stem="vocals")  # same 16th-note slot as a_note
    line = LyricLine(section="Verse 1", text="la la", start=10.0, end=11.0, words=[], confidence=0.9)
    t = Transcription(take=1, analysis=make_analysis([a_note, b_note]), lines=[line])

    melody, kept, total = selected_melody(t)
    assert (total, kept) == (2, 2)                                # both notes are selected

    beatmap = BeatMap.from_analysis(t.analysis)
    assert beatmap.sixteenth(a_note.onset) == beatmap.sixteenth(b_note.onset)     # confirm the collision
    emitted = [e for e in melody_events(t, beatmap) if e.pitch is not None]
    assert len(emitted) == 1                                      # but only one survives quantisation

    assert _tab_kept_total(render_tab("T", t)) == (2, 2)          # disclosure still says 2 of 2, not 1
