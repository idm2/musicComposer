from songcomposer.models import Analysis, Chord, GlobalInfo, Note, Section, Transcription
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
