import pytest

from songcomposer.generate.prompt import build_request
from songcomposer.models import Analysis, GlobalInfo, SongSpec, SpecSection, Subjective

SPEC = SongSpec(title="Glass Hour", style_prompt="sparse indie folk, melancholy", negative_style="edm, autotune",
                vocal_gender="male", target_duration_s=150,
                sections=[SpecSection(name="Verse 1", lines=["one line here", "two line here"], duration_s=75),
                          SpecSection(name="Chorus", lines=["sing it back"], duration_s=75)])
ANALYSIS = Analysis(
    audio_sha1="a" * 40, engines={},
    global_info=GlobalInfo(key="A minor", key_confidence=0.8, tempo_bpm=91.6, time_signature="3/4",
                           loudness_lufs=-14, duration_s=200),
    subjective=Subjective(genre_tags=["folk"], instrumentation=["fingerpicked acoustic guitar", "upright bass"],
                          timbre="warm and woody", vocal_character="breathy low tenor", vocal_gender="male",
                          production="dry, intimate", emotional_arc="", arrangement_density=[], sections=[]))


def test_loose_uses_spec_style_only():
    r = build_request(SPEC, ANALYSIS, "loose")
    assert r.style_prompt == "sparse indie folk, melancholy"


def test_medium_adds_tempo_instruments_and_voice_but_not_key():
    s = build_request(SPEC, ANALYSIS, "medium").style_prompt
    assert "92 BPM" in s and "fingerpicked acoustic guitar" in s and "breathy low tenor" in s
    assert "A minor" not in s


def test_close_adds_key_meter_and_production():
    s = build_request(SPEC, ANALYSIS, "close").style_prompt
    assert "A minor" in s and "3/4" in s and "dry, intimate" in s and "warm and woody" in s


def test_note_is_appended_and_partial_analysis_is_tolerated():
    r = build_request(SPEC, Analysis(audio_sha1="a" * 40, engines={}), "close", note="slower, more space")
    assert r.style_prompt == "sparse indie folk, melancholy, slower, more space"
    assert build_request(SPEC, None, "medium").style_prompt == "sparse indie folk, melancholy"


def test_lyrics_text_has_section_headers():
    assert build_request(SPEC, None, "loose").lyrics_text() == (
        "[Verse 1]\none line here\ntwo line here\n\n[Chorus]\nsing it back")


def test_bad_fidelity_rejected():
    with pytest.raises(ValueError, match="fidelity"):
        build_request(SPEC, None, "exact")
