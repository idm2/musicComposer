import json

import httpx
import pytest

from songcomposer.generate.elevenlabs import ElevenLabsProvider, style_tags
from songcomposer.generate.provider import GenerationRequest
from songcomposer.jsonio import read_json
from songcomposer.models import SpecSection

REQ = GenerationRequest(title="Glass Hour", style_prompt="sparse indie folk, breathy female vocal, 92 BPM",
                        negative_style="edm, autotune", target_duration_s=150, n_takes=3,
                        sections=[SpecSection(name="Verse 1", lines=["a b c", "d e f"], duration_s=75,
                                              style_notes="solo fingerpicked guitar"),
                                  SpecSection(name="Chorus", lines=["g h i"], duration_s=75)])


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")


def test_style_tags_split_trim_dedupe_and_cap():
    assert style_tags(" folk,  breathy vocal , folk,") == ["folk", "breathy vocal"]
    assert len(style_tags(",".join(f"t{i}" for i in range(80)))) == 50


def test_v1_model_is_refused():
    with pytest.raises(ValueError, match="chunks"):
        ElevenLabsProvider("music_v1")


def test_payload_is_a_v2_chunk_plan_with_three_seeds():
    p = ElevenLabsProvider("music_v2_5").payload(REQ)
    assert p["model_id"] == "music_v2_5" and p["seeds"] == [11, 22, 33]
    c0, c1 = p["composition_plan"]["chunks"]
    assert c0["text"] == "[Verse 1]\na b c\nd e f" and c0["duration_ms"] == 75000
    assert c0["positive_styles"] == ["sparse indie folk", "breathy female vocal", "92 BPM", "solo fingerpicked guitar"]
    assert c0["negative_styles"] == ["edm", "autotune"] and c0["context_adherence"] == "high"
    assert c1["positive_styles"] == ["sparse indie folk", "breathy female vocal", "92 BPM"]


def test_estimate_is_per_minute_times_takes():
    e = ElevenLabsProvider("music_v2_5").estimate(REQ)
    assert e.usd == pytest.approx(3 * 2.5 * 0.15)


def test_generate_makes_one_call_per_seed_and_writes_audio(tmp_path, monkeypatch):
    from songcomposer.generate import elevenlabs as mod
    monkeypatch.setattr(mod, "probe_duration", lambda p: 148.0)
    seen = []

    def handler(req: httpx.Request):
        assert req.headers["xi-api-key"] == "el-test"
        assert req.url.params["output_format"] == "mp3_44100_192"
        seen.append(json.loads(req.content))
        return httpx.Response(200, content=b"ID3" + b"\x00" * 1000)

    got = []
    cost = ElevenLabsProvider("music_v2_5", client=httpx.Client(transport=httpx.MockTransport(handler))).generate(
        REQ, tmp_path, 5, got.append)
    assert [b["seed"] for b in seen] == [11, 22, 33]
    assert all("chunks" in b["composition_plan"] and b["model_id"] == "music_v2_5" and "seeds" not in b for b in seen)
    assert [t.path.name for t in got] == ["take-5.mp3", "take-6.mp3", "take-7.mp3"]
    assert cost == pytest.approx(3 * 148.0 / 60 * 0.15)


def test_rejected_plan_saves_the_suggestion_and_raises(tmp_path):
    suggestion = {"chunks": [{"text": "[Verse]\nsafe", "duration_ms": 10000, "positive_styles": ["folk"]}]}

    def handler(req):
        return httpx.Response(400, json={"detail": {"status": "bad_composition_plan", "message": "copyright",
                                                    "data": {"composition_plan_suggestion": suggestion}}})

    provider = ElevenLabsProvider("music_v2_5", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(RuntimeError, match="bad_composition_plan"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)
    assert read_json(tmp_path / "elevenlabs-plan-suggestion.json") == suggestion
