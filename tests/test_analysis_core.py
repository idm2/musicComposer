import pytest

from songcomposer import analysis
from songcomposer.analysis import cache, subjective
from songcomposer.config import Config
from songcomposer.models import Subjective

HEARD = Subjective(
    genre_tags=["indie folk"], instrumentation=["fingerpicked acoustic guitar", "upright bass"],
    timbre="warm, woody, close-miked", vocal_character="breathy low tenor, restrained",
    vocal_gender="male", production="dry, intimate, tape-ish", emotional_arc="resigned → hopeful",
    arrangement_density=[], sections=[])


def test_cached_runs_once_per_key(tmp_path):
    calls = []

    def work():
        calls.append(1)
        return {"v": 1}

    assert cache.cached(tmp_path, "a" * 40, "beats", "1", work) == {"v": 1}
    assert cache.cached(tmp_path, "a" * 40, "beats", "1", work) == {"v": 1}
    assert len(calls) == 1
    cache.cached(tmp_path, "a" * 40, "beats", "2", work)      # version bump invalidates
    cache.cached(tmp_path, "b" * 40, "beats", "1", work)      # different audio invalidates
    assert len(calls) == 3


def test_subjective_model_cannot_carry_chords_key_or_tempo():
    """Two-ears rule, enforced structurally: the LLM ear has nowhere to put a chord."""
    fields = set(Subjective.model_fields)
    assert not {"chords", "key", "tempo", "tempo_bpm", "progression"} & fields


def test_listen_sends_audio_and_caches(tmp_path, sine_wav, monkeypatch):
    sent = []

    def fake_chat_json(messages, model, out_type, **kw):
        sent.append((messages, model))
        return HEARD

    monkeypatch.setattr(subjective, "chat_json", fake_chat_json)
    got = subjective.listen(sine_wav, tmp_path, "google/gemini-2.5-pro")
    assert got == HEARD
    parts = sent[0][0][-1]["content"]
    assert any(p["type"] == "input_audio" for p in parts)
    assert "chord" in sent[0][0][0]["content"].lower()        # the prompt forbids chord guessing
    subjective.listen(sine_wav, tmp_path, "google/gemini-2.5-pro")
    assert len(sent) == 1                                      # second call served from cache


def test_analyze_subjective_only_records_engines_and_leaves_chords_empty(tmp_path, sine_wav, monkeypatch):
    monkeypatch.setattr(subjective, "chat_json", lambda *a, **k: HEARD)
    a = analysis.analyze(sine_wav, tmp_path, Config(), ears={"subjective"})
    assert a.subjective == HEARD and a.chords == [] and a.global_info is None
    assert a.engines == {"subjective": "google/gemini-2.5-pro"}
    assert len(a.audio_sha1) == 40


def test_analyze_rejects_unknown_ear(tmp_path, sine_wav):
    with pytest.raises(ValueError, match="unknown ear"):
        analysis.analyze(sine_wav, tmp_path, Config(), ears={"vibes"})
