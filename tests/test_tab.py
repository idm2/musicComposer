from songcomposer.models import Analysis, Chord, GlobalInfo, LyricLine, Note, Section, Transcription
from songcomposer.render import guitar
from songcomposer.render.tab import render_tab
from songcomposer.render.timing import BeatMap

BEATS = [i * 0.6 for i in range(32)]


def ch(harte, symbol, onset=0.0, dur=2.4, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


def test_open_shapes_and_generated_barres():
    assert guitar.shape_for(ch("C:maj", "C")) == "x32010"
    assert guitar.shape_for(ch("A:min7", "Am7")) == "x02010"
    assert guitar.shape_for(ch("F#:min", "F#m")) == "244222"               # E-shape barre at fret 2
    assert guitar.shape_for(ch("D:7", "D7")) == "xx0212"
    assert guitar.shape_for(ch("D#:maj", "D#")) == "11.13.13.12.11.11"
    assert guitar.shape_for(ch("C:weird", "C(weird)")) is None


def test_fret_positions_prefer_staying_put():
    pos = guitar.fret_positions([64, 65, 67])                              # E4 F4 G4
    assert pos[0] in [(5, 0), (4, 5)]
    assert all(abs(b[1] - a[1]) <= 4 for a, b in zip(pos, pos[1:]))
    assert guitar.into_range(28) == 40 and guitar.into_range(90) == 78


def test_strum_detection_and_pattern():
    notes = [Note(pitch=p, onset=t + j * 0.01, duration=0.3, confidence=0.8, stem="other")
             for t in (0.0, 0.6, 0.9, 1.2, 1.8, 2.1) for j, p in enumerate((48, 55, 60, 64))]
    notes.append(Note(pitch=70, onset=1.5, duration=0.2, confidence=0.8, stem="other"))      # a lone note: not a strum
    onsets = guitar.strum_onsets(notes)
    assert [round(o, 1) for o in onsets] == [0.0, 0.6, 0.9, 1.2, 1.8, 2.1]
    assert guitar.bar_pattern(onsets, 0, BeatMap(BEATS, BEATS[::4], 4)) == "D-DUD-DU"


def test_render_tab_sections():
    melody = [Note(pitch=64, onset=0.0, duration=0.6, confidence=0.9, stem="vocals"),
              Note(pitch=67, onset=0.6, duration=0.6, confidence=0.3, stem="vocals")]
    strums = [Note(pitch=p, onset=t, duration=0.3, confidence=0.8, stem="other") for t in (0.0, 0.6, 1.2, 1.8) for p in (48, 55, 60)]
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=melody + strums,
                 chords=[ch("C:maj", "C"), ch("C:weird", "C(weird)", 2.4, 2.4, conf=0.2)],
                 sections=[Section(label="Verse 1", start=0.0, end=19.2, confidence=0.9)],
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=19.2))
    out = render_tab("Glass Hour", Transcription(take=1, analysis=a, lines=[]))
    assert f"{'C':<11}x32010" in out
    assert f"{'C(weird)?':<11}(no shape in dictionary — work it out from the chord name)" in out
    assert "Verse 1: strummed   | D-D-D-D- |" in out
    assert "e|0---" + "-" * 12 + "(3)-" in out                           # E4 open on beat 1, low-confidence G4 (in parentheses) on beat 2
    assert "(n) = low-confidence note" in out


def test_zero_length_section_says_not_detected_even_when_a_strum_overlaps_its_instant():
    # A section the generator never sang is now recorded zero-length with confidence 0.0 (concurrent fix
    # to analysis.sections). A real strum pattern that happens to fall in the same bar as that instant
    # must never be reported as a detected rhythm for a section that isn't in the audio.
    strums = [Note(pitch=p, onset=t, duration=0.3, confidence=0.8, stem="other")
              for t in (9.6, 10.2, 10.8, 11.4) for p in (48, 55, 60)]                      # a real strum in bar 4
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=strums,
                 chords=[ch("C:maj", "C")],
                 sections=[Section(label="Bridge", start=9.6, end=9.6, confidence=0.0)],   # zero length, sits in bar 4
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                        loudness_lufs=-14, duration_s=19.2))
    out = render_tab("Glass Hour", Transcription(take=1, analysis=a, lines=[]))
    assert "Bridge: not detected in the audio" in out
    assert "Bridge: strummed" not in out


def make_analysis(notes, chords=(), sections=(), duration_s=19.2):
    return Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=notes,
                     chords=list(chords), sections=list(sections),
                     global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                            loudness_lufs=-14, duration_s=duration_s))


# ---- C1: "strummed" requires real evidence -----------------------------------------------------


def test_sparse_single_stroke_bars_are_not_called_strummed():
    # two isolated single-stroke groups in two different bars: real strum events, but far too sparse
    # and inconsistent to call "strummed". Must fall to the new honest in-between state.
    strums = ([Note(pitch=p, onset=0.0, duration=0.3, confidence=0.8, stem="other") for p in (48, 55, 60)] +
              [Note(pitch=p, onset=3.0, duration=0.3, confidence=0.8, stem="other") for p in (48, 55, 60)])
    a = make_analysis(strums, chords=[ch("C:maj", "C")], sections=[Section(label="Verse 1", start=0.0, end=4.8, confidence=0.9)])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "Verse 1: strummed" not in out
    assert "Verse 1: strums detected but no consistent pattern — listen and choose your own" in out


def test_common_pattern_thresholds():
    # below min_hits: perfect coverage, but every bar has only one stroke
    assert guitar.common_pattern(["D-------", "D-------", "D-------"], min_hits=3, min_coverage=0.5) is None
    # below min_coverage: every bar has enough hits, but no pattern repeats in more than 1 of 4 bars
    assert guitar.common_pattern(["D-D-D-D-", "U-U-U-U-", "D-U-D-U-", "U-D-U-D-"], min_hits=3, min_coverage=0.5) is None
    # both satisfied: a real, consistent pattern
    assert guitar.common_pattern(["D-D-D-D-", "D-D-D-D-", "U-------"], min_hits=3, min_coverage=0.5) == "D-D-D-D-"
    # unthresholded callers keep the old, permissive behaviour
    assert guitar.common_pattern(["D-------", "--------"]) == "D-------"


# ---- I3: the last bar of a section must not be dropped from the strum search ------------------


def test_last_bar_of_a_section_is_included_in_the_strum_search():
    # section ends at sixteenth 33 (inside bar 2, bar16=16). The old `last = sixteenth(end)//bar16`
    # arithmetic yields range(0, 2) and never examines bar 2, even though a clean 4-stroke pattern lives there.
    strums = [Note(pitch=p, onset=t, duration=0.3, confidence=0.8, stem="other")
              for t in (4.8, 5.4, 6.0, 6.6) for p in (48, 55, 60)]
    a = make_analysis(strums, chords=[ch("C:maj", "C")],
                       sections=[Section(label="Verse 1", start=0.0, end=4.95, confidence=0.9)])   # sixteenth(4.95) == 33
    assert BeatMap.from_analysis(a).sixteenth(4.95) == 33          # pin down the reviewer's exact scenario
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "Verse 1: strummed   | D-D-D-D- |" in out


# ---- C2: the melody is clipped to sung spans and reports what it dropped ----------------------


def test_melody_is_clipped_to_sung_spans_and_reports_what_was_dropped():
    before = Note(pitch=64, onset=5.0, duration=0.3, confidence=0.9, stem="vocals")   # instrumental-intro artefact
    inside = Note(pitch=67, onset=10.5, duration=0.3, confidence=0.9, stem="vocals")  # within the line's padded span
    line = LyricLine(section="Verse 1", text="la la", start=10.0, end=11.0, words=[], confidence=0.9)

    with_artefact = render_tab("T", Transcription(take=1, analysis=make_analysis([before, inside]), lines=[line]))
    without_artefact = render_tab("T", Transcription(take=1, analysis=make_analysis([inside]), lines=[line]))

    assert "1 of 2 transcribed vocal notes selected for the melody line (1 outside sung spans or below confidence 0.3)." in with_artefact
    # the pre-line note must be fully excluded from the tabbed grid, not merely uncounted
    grid_with = with_artefact.split("MELODY")[1].split("\n", 2)[2]
    grid_without = without_artefact.split("MELODY")[1].split("\n", 2)[2]
    assert grid_with == grid_without


def test_melody_without_lyric_timing_falls_back_to_all_notes_and_says_so():
    note = Note(pitch=64, onset=1.0, duration=0.3, confidence=0.9, stem="vocals")
    line = LyricLine(section="Verse 1", text="never sung", start=None, end=None, words=[], confidence=0.0)
    a = make_analysis([note])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[line]))
    assert "no lyric timing detected — notes not clipped to sung spans" in out
    assert "1 of 1 transcribed vocal notes selected for the melody line (0 outside sung spans or below confidence 0.3)." in out


def test_confidence_floor_drops_very_low_confidence_notes_but_keeps_moderate_ones():
    dropped_note = Note(pitch=64, onset=1.0, duration=0.3, confidence=0.1, stem="vocals")   # below the 0.3 floor
    kept_note = Note(pitch=67, onset=1.6, duration=0.3, confidence=0.4, stem="vocals")       # above the floor, still low
    a = make_analysis([dropped_note, kept_note])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "1 of 2 transcribed vocal notes selected for the melody line (1 outside sung spans or below confidence 0.3)." in out
    melody_block = out.split("MELODY")[1]
    assert "(3)" in melody_block                                  # surviving note (fret 3) still shown as low-confidence


# ---- I4: the slash-chord bass-note simplification is disclosed --------------------------------


def test_slash_chord_shape_discloses_the_dropped_bass_note():
    slash = Chord(symbol="E/G#", harte="E:maj/3", root="E", quality="maj", extensions=[], bass="G#",
                  onset=0.0, duration=2.0, confidence=0.9)
    plain = ch("C:maj", "C", onset=3.0)
    a = make_analysis([], chords=[slash, plain])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "E/G#       022100  (bass note not shown — let the bass carry it)" in out
    assert "C          x32010" in out
    assert "x32010  (bass" not in out


# ---- M5: untested fallback paths ---------------------------------------------------------------


def test_picked_branch_when_notes_present_but_no_full_strum():
    notes = [Note(pitch=48, onset=1.0, duration=0.3, confidence=0.8, stem="other"),
             Note(pitch=55, onset=1.01, duration=0.3, confidence=0.8, stem="other")]     # only 2 simultaneous: no strum
    a = make_analysis(notes, chords=[ch("C:maj", "C")], sections=[Section(label="Verse 1", start=0.0, end=2.4, confidence=0.9)])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "Verse 1: picked / arpeggiated — no full strums detected; arpeggiate the chord shapes" in out


def test_no_guitar_range_accompaniment_detected_branch():
    a = make_analysis([], chords=[ch("C:maj", "C")], sections=[Section(label="Verse 1", start=0.0, end=2.4, confidence=0.9)])
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "Verse 1: no guitar-range accompaniment detected" in out


def test_render_tab_without_a_beat_grid():
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=[], downbeats=[],
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                        loudness_lufs=-14, duration_s=2.4))
    out = render_tab("T", Transcription(take=1, analysis=a, lines=[]))
    assert "No beat grid was detected — tab cannot be laid out. See chords.txt." in out
    assert "CHORD SHAPES" not in out


def test_importing_tab_does_not_import_heavy_deps():
    """Architectural pin: render/ is pure Python — no torch/librosa/numpy/music21 at module scope,
    anywhere in the import chain. Run in a subprocess so the current session's already-imported
    modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.render.tab; "
         "heavy = {'torch', 'librosa', 'numpy', 'music21'}; "
         "leaked = heavy & set(sys.modules); "
         "assert not leaked, f'render.tab imported {leaked} eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
