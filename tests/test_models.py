import pytest
from pydantic import ValidationError

from songcomposer.models import Analysis, Chord, Note, SongSpec, SpecSection


def test_confidence_is_mandatory_and_bounded():
    with pytest.raises(ValidationError):
        Note(pitch=60, onset=0, duration=1, stem="vocals")                 # no confidence
    with pytest.raises(ValidationError):
        Note(pitch=60, onset=0, duration=1, stem="vocals", confidence=1.5)
    with pytest.raises(ValidationError):
        Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None,
              onset=0, duration=2)                                          # no confidence


def test_models_are_instrument_neutral():
    """Global constraint: nothing guitar-specific upstream of render/."""
    banned = {"fret", "string", "capo", "tab", "fingering"}
    for model in (Note, Chord, Analysis):
        assert not banned & set(model.model_fields), model


def test_analysis_minimal_roundtrip():
    a = Analysis(audio_sha1="a" * 40, engines={"subjective": "google/gemini-2.5-pro"})
    again = Analysis.model_validate_json(a.model_dump_json())
    assert again == a and again.chords == [] and again.global_info is None


def test_songspec_lyrics_text_and_duration():
    spec = SongSpec(title="Glass Hour", style_prompt="sparse indie folk", negative_style="",
                    target_duration_s=120,
                    sections=[SpecSection(name="Verse 1", lines=["a b", "c d"], duration_s=30),
                              SpecSection(name="Break", lines=[], duration_s=10)])
    assert spec.all_lines() == ["a b", "c d"]
    assert spec.sections_total_s() == 40
