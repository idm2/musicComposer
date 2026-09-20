import subprocess
import sys

import pytest

from songcomposer import transcribe
from songcomposer.models import SongSpec, SpecSection, Word

SPEC = SongSpec(title="T", style_prompt="s", target_duration_s=60, sections=[
    SpecSection(name="Verse 1", duration_s=30, lines=["Glass hour, hold me still", "Turn the morning down"]),
    SpecSection(name="Chorus", duration_s=30, lines=["Until you come around"])])


def heard(*triples):
    return [Word(word=w, start=s, end=e, confidence=0.9) for w, s, e in triples]


def test_exact_match_gives_spec_spelling_with_heard_timing():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2),
              ("turn", 5.0, 5.3), ("the", 5.3, 5.4), ("morning", 5.4, 6.0), ("down", 6.0, 6.8),
              ("until", 20.0, 20.4), ("you", 20.4, 20.6), ("come", 20.6, 21.0), ("around", 21.0, 22.0))
    lines = transcribe.align_lines(SPEC, h)
    assert [(l.section, l.start, l.end) for l in lines] == [("Verse 1", 1.0, 3.2), ("Verse 1", 5.0, 6.8), ("Chorus", 20.0, 22.0)]
    assert [w.word for w in lines[0].words] == ["Glass", "hour,", "hold", "me", "still"]      # OUR spelling
    assert lines[0].confidence == pytest.approx(0.9)


def test_a_misheard_word_is_interpolated_and_marked_zero_confidence():
    h = heard(("glass", 1.0, 1.3), ("our", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2))
    line = transcribe.align_lines(SPEC, h)[0]
    hour = line.words[1]
    assert hour.word == "hour," and hour.confidence == 0.0 and 1.3 <= hour.start <= hour.end <= 2.0
    assert line.confidence == pytest.approx(0.9 * 4 / 5)


def test_an_unsung_line_is_reported_not_invented():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2))
    lines = transcribe.align_lines(SPEC, h)
    assert (lines[1].start, lines[1].end, lines[1].words, lines[1].confidence) == (None, None, [], 0.0)


def test_sections_follow_the_aligned_lines():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2),
              ("until", 20.0, 20.4), ("you", 20.4, 20.6), ("come", 20.6, 21.0), ("around", 21.0, 22.0))
    secs = transcribe.sections_from_lines(transcribe.align_lines(SPEC, h), 40.0)
    assert [(s.label, s.start, s.end) for s in secs] == [("Verse 1", 1.0, 20.0), ("Chorus", 20.0, 40.0)]


def test_importing_transcribe_does_not_import_the_analysis_stack():
    """Architectural pin: transcribe.py imports analyze() from songcomposer.analysis, which is
    import-light by design (heavy libraries are imported lazily inside analyze() itself). Run in
    a subprocess so the current session's already-imported stack cannot mask a regression."""
    heavy = ["torch", "librosa", "faster_whisper", "basic_pitch", "demucs", "beat_this", "lv_chordia"]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.transcribe; "
         + "; ".join(f"assert {m!r} not in sys.modules, 'transcribe imported {m} eagerly'" for m in heavy)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
