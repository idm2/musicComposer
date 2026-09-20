from songcomposer.models import Analysis, Chord, GlobalInfo, LyricLine, Transcription, Word
from songcomposer.render.chordsheet import render_chordsheet
from songcomposer.render.guitar import suggest_capo, transpose_harte


def ch(harte, symbol, onset, dur, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q or "maj", extensions=[], bass=None,
                 onset=onset, duration=dur, confidence=conf)


def line(section, text, start, step=0.5):
    words = [Word(word=w, start=start + i * step, end=start + (i + 1) * step, confidence=0.9) for i, w in enumerate(text.split())]
    return LyricLine(section=section, text=text, start=words[0].start, end=words[-1].end, words=words, confidence=0.9)


def make(chords, lines):
    a = Analysis(audio_sha1="a" * 40, engines={}, chords=chords,
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                        loudness_lufs=-14, duration_s=60))
    return Transcription(take=1, analysis=a, lines=lines)


def test_chords_sit_above_the_word_they_land_on():
    t = make([ch("C:maj", "C", 10.0, 1.5), ch("A:min", "Am", 11.5, 2.0)],
             [line("Verse 1", "Glass hour hold me still", 10.0)])          # "me" starts at 11.5
    out = render_chordsheet("Glass Hour", t).splitlines()
    i = out.index("Glass hour hold me still")
    assert out[i - 2] == "[Verse 1]"
    assert out[i - 1].index("Am") == out[i].index("me")


def test_low_confidence_chords_are_marked_and_explained():
    out = render_chordsheet("T", make([ch("F:maj", "F", 10.0, 2.0, conf=0.3)], [line("Verse 1", "one two", 10.0)]))
    assert "F?" in out and "? = low-confidence" in out and "1 of 1 chords" in out


def test_intro_chords_and_unaligned_lines_are_shown_honestly():
    lines = [line("Verse 1", "one two", 10.0),
             LyricLine(section="Verse 1", text="never sung", start=None, end=None, words=[], confidence=0.0)]
    out = render_chordsheet("T", make([ch("G:maj", "G", 0.0, 4.0), ch("C:maj", "C", 10.0, 2.0)], lines))
    assert "[Intro]\nG" in out
    assert "never sung   (timing not detected — chords not placed)" in out


def test_header_has_key_tempo_and_meter():
    out = render_chordsheet("Glass Hour", make([], [line("V", "a b", 1.0)]))
    assert out.startswith("Glass Hour\n") and "Key: C major" in out and "100 BPM" in out and "4/4" in out


def test_transpose_harte():
    assert transpose_harte("Bb:min7/b7", -1) == "A:min7/b7"
    assert transpose_harte("C:maj", 2) == "D:maj"


def test_capo_turns_flat_key_shapes_into_open_shapes():
    capo, shapes = suggest_capo([ch("Bb:maj", "Bb", 0, 4), ch("Eb:maj", "Eb", 4, 4), ch("F:maj", "F", 8, 4), ch("G:min", "Gm", 12, 4)])
    assert capo == 3 and shapes == {"Bb": "G", "Eb": "C", "F": "D", "Gm": "Em"}


def test_no_capo_when_the_shapes_are_already_open():
    assert suggest_capo([ch("G:maj", "G", 0, 4), ch("C:maj", "C", 4, 4), ch("D:maj", "D", 8, 4)]) == (0, {})


def test_trailing_chords_after_an_outro_section_do_not_duplicate_its_header():
    # the last SUNG section is already "Outro"; chords ringing out after it must not print a second [Outro]
    t = make([ch("A:maj", "A", 10.0, 1.0), ch("D:maj", "D", 20.0, 1.0)],
             [line("Outro", "with you home", 10.0)])
    out = render_chordsheet("T", t)
    assert out.count("[Outro]") == 1
    assert out.rstrip("\n").splitlines()[-1] == "D"


def test_importing_chordsheet_does_not_import_heavy_deps():
    """Architectural pin: render/ is pure Python — no torch/librosa/numpy/music21 at module scope,
    anywhere in the import chain. Run in a subprocess so the current session's already-imported
    modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.render.chordsheet; "
         "heavy = {'torch', 'librosa', 'numpy', 'music21'}; "
         "leaked = heavy & set(sys.modules); "
         "assert not leaked, f'render.chordsheet imported {leaked} eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
