import subprocess
import sys

import pytest

from songcomposer import transcribe
from songcomposer.jsonio import write_model
from songcomposer.models import (Analysis, Chosen, GlobalInfo, LyricLine, SongSpec, SpecSection, Word)
from songcomposer.paths import SongPaths

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


def _line(section, start, end, confidence):
    return LyricLine(section=section, text="x", start=start, end=end, words=[], confidence=confidence)


def test_a_fully_unaligned_middle_section_is_reported_not_dropped():
    """Reviewer's exact repro: Verse 1 aligned -> Hook (0 aligned lines) -> Verse 2 aligned.
    Hook must not disappear, and Verse 1 must not silently swallow its span."""
    lines = [_line("Verse 1", 1.0, 3.0, 0.9),
             _line("Hook", None, None, 0.0),
             _line("Hook", None, None, 0.0),
             _line("Verse 2", 50.0, 52.0, 0.8)]
    secs = transcribe.sections_from_lines(lines, 120.0)
    assert [(s.label, s.start, s.end) for s in secs] == [
        ("Verse 1", 1.0, 50.0), ("Hook", 50.0, 50.0), ("Verse 2", 50.0, 120.0)]
    assert secs[1].start == secs[1].end and secs[1].confidence == 0.0


def test_a_fully_unaligned_first_section_appears_before_the_first_aligned_one():
    lines = [_line("Hook", None, None, 0.0),
             _line("Verse 1", 5.0, 7.0, 0.9)]
    secs = transcribe.sections_from_lines(lines, 60.0)
    assert [(s.label, s.start, s.end) for s in secs] == [("Hook", 5.0, 5.0), ("Verse 1", 5.0, 60.0)]
    assert secs[0].confidence == 0.0


def test_a_fully_unaligned_last_section_sits_at_the_end():
    lines = [_line("Verse 1", 2.0, 4.0, 1.0),
             _line("Outro", None, None, 0.0)]
    secs = transcribe.sections_from_lines(lines, 90.0)
    assert [(s.label, s.start, s.end) for s in secs] == [("Verse 1", 2.0, 90.0), ("Outro", 90.0, 90.0)]
    assert secs[1].confidence == 0.0


def test_structure_engine_is_relabelled_when_the_lyric_derived_override_fires(root, monkeypatch):  # noqa: F811
    """M6: run_transcribe replaces result.sections with sections_from_lines(...) when our own form
    beats the DSP guess, but engines['structure'] still claimed 'librosa agglomerative + heard
    labels' — a lie about which ear actually produced the sections that shipped."""
    p = SongPaths("demo")
    write_model(p.spec, SongSpec(title="T", style_prompt="s", target_duration_s=10,
                                 sections=[SpecSection(name="Verse 1", duration_s=10, lines=["Glass hour"])]))
    write_model(p.chosen, Chosen(take=1, file="take-1.mp3", provider="fake", sha1="a" * 40, chosen_at="now"))
    p.takes_dir.mkdir(parents=True, exist_ok=True)
    (p.takes_dir / "take-1.mp3").write_bytes(b"x")

    words = [Word(word="Glass", start=0.0, end=0.4, confidence=0.9), Word(word="hour", start=0.4, end=0.8, confidence=0.9)]
    fake = Analysis(audio_sha1="a" * 40, engines={"structure": "librosa agglomerative + heard labels"},
                    global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                           loudness_lufs=-14, duration_s=10.0),
                    lyrics=words)
    monkeypatch.setattr(transcribe, "analyze", lambda *a, **k: fake)
    monkeypatch.setattr(transcribe, "to_wav",
                        lambda src, dst: (dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b"wav")))

    out = transcribe.run_transcribe("demo")
    assert out.analysis.engines["structure"] == "aligned spec lyrics"


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
