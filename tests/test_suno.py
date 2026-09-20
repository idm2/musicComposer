import json

import httpx
import pytest

from songcomposer.generate.provider import GenerationRequest
from songcomposer.generate.suno import SunoProvider, extract_tracks
from songcomposer.models import SpecSection

REQ = GenerationRequest(title="Glass Hour", style_prompt="sparse indie folk", negative_style="edm",
                        vocal_gender="female", target_duration_s=150, n_takes=3,
                        sections=[SpecSection(name="Verse 1", lines=["a b c", "d e f"], duration_s=75),
                                  SpecSection(name="Chorus", lines=["g h i"], duration_s=75)])


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "kie-test")


@pytest.mark.parametrize("result", [
    {"resultUrls": ["https://x/a.mp3", "https://x/b.mp3"]},
    {"sunoData": [{"id": "1", "audio_url": "https://x/a.mp3"}, {"id": "2", "audio_url": "https://x/b.mp3"}]},
    {"data": [{"id": "1", "audioUrl": "https://x/a.mp3"}, {"id": "2", "audioUrl": "https://x/b.mp3"}]},
    {"response": {"sunoData": [{"id": "1", "audio_url": "https://x/a.mp3"}, {"id": "2", "audio_url": "https://x/b.mp3"}]}},
])
def test_extract_tracks_accepts_every_documented_shape(result):
    assert [t["url"] for t in extract_tracks(result)] == ["https://x/a.mp3", "https://x/b.mp3"]


def test_extract_tracks_unknown_shape_is_empty():
    assert extract_tracks({"something": "else"}) == []


def test_payload_is_custom_mode_with_exact_lyrics():
    p = SunoProvider("V6").payload(REQ)
    assert p["n_requests"] == 2                                  # 3 takes at 2 tracks per request
    body = p["body"]
    assert body["model"] == "ai-music-api/generate" and "callBackUrl" not in body
    i = body["input"]
    assert i["custom_mode"] is True and i["instrumental"] is False and i["model"] == "V6"
    assert i["prompt"] == "[Verse 1]\na b c\nd e f\n\n[Chorus]\ng h i"
    assert (i["style"], i["title"], i["negative_tags"], i["vocal_gender"], i["duration"]) == (
        "sparse indie folk", "Glass Hour", "edm", "f", 150)


def test_any_gender_omits_the_field():
    assert "vocal_gender" not in SunoProvider("V6").payload(REQ.model_copy(update={"vocal_gender": "any"}))["body"]["input"]


def test_estimate_is_two_requests_and_says_provisional():
    e = SunoProvider("V6").estimate(REQ)
    assert e.usd == pytest.approx(0.12) and "provisional" in e.basis


def _server(states, credits):
    """states: per-task list of successive poll payloads."""
    created, credit_iter = [], iter(credits)

    def handler(req: httpx.Request):
        if req.url.host == "api.kie.ai":
            assert req.headers["authorization"] == "Bearer kie-test"
        else:
            assert "authorization" not in req.headers      # audio CDN host must never see the key
        url = str(req.url)
        if url.endswith("/chat/credit"):
            return httpx.Response(200, json={"code": 200, "data": next(credit_iter)})
        if url.endswith("/jobs/createTask"):
            created.append(json.loads(req.content))
            return httpx.Response(200, json={"code": 200, "msg": "success", "data": {"taskId": f"t{len(created)}"}})
        if "/jobs/recordInfo" in url:
            task = req.url.params["taskId"]
            return httpx.Response(200, json={"code": 200, "data": states[task].pop(0)})
        return httpx.Response(200, content=b"ID3" + b"\x00" * 200_000)      # the mp3 download

    return handler, created


def test_generate_polls_saves_raw_downloads_and_measures_cost(tmp_path):
    ok = lambda a, b: {"state": "success", "resultJson": json.dumps(
        {"sunoData": [{"id": a, "audio_url": f"https://cdn/{a}.mp3", "duration": 151.2},
                      {"id": b, "audio_url": f"https://cdn/{b}.mp3", "duration": 149.0}]})}
    handler, created = _server({"t1": [{"state": "queuing"}, {"state": "generating"}, ok("a", "b")],
                                "t2": [ok("c", "d")]}, credits=[1000, 976])
    got = []
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    cost = provider.generate(REQ, tmp_path, 1, got.append)
    assert len(created) == 2
    assert [t.path.name for t in got] == ["take-1.mp3", "take-2.mp3", "take-3.mp3", "take-4.mp3"]
    assert [t.provider_ref for t in got] == ["a", "b", "c", "d"] and got[0].duration_s == 151.2
    assert (tmp_path / "raw-suno-t1.json").exists() and (tmp_path / "raw-suno-t2.json").exists()
    assert cost == pytest.approx(24 * 0.005)


def test_failed_task_raises_with_provider_message(tmp_path):
    handler, _ = _server({"t1": [{"state": "fail", "failMsg": "SENSITIVE_WORD_ERROR"}], "t2": []}, credits=[10, 10])
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="SENSITIVE_WORD_ERROR"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)


def test_unparseable_success_points_at_the_raw_file(tmp_path):
    handler, _ = _server({"t1": [{"state": "success", "resultJson": "{\"odd\": 1}"}], "t2": []}, credits=[10, 10])
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="raw-suno-t1.json"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)


def test_insufficient_credits_code_in_200_body_is_an_error(tmp_path):
    def handler(req):
        if str(req.url).endswith("/chat/credit"):
            return httpx.Response(200, json={"code": 200, "data": 0})
        return httpx.Response(200, json={"code": 402, "msg": "Insufficient credits"})
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="402"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)


def test_api_key_is_never_sent_to_the_audio_host(tmp_path):
    """The download URL comes out of the provider's result JSON and points at an arbitrary
    CDN host, not api.kie.ai. The KIE_API_KEY must never be attached to that request."""
    ok = lambda a, b: {"state": "success", "resultJson": json.dumps(
        {"sunoData": [{"id": a, "audio_url": f"https://cdn/{a}.mp3", "duration": 151.2},
                      {"id": b, "audio_url": f"https://cdn/{b}.mp3", "duration": 149.0}]})}
    base_handler, _ = _server({"t1": [ok("a", "b")], "t2": [ok("c", "d")]}, credits=[1000, 976])
    download_headers = []

    def handler(req: httpx.Request):
        if req.url.host != "api.kie.ai":
            download_headers.append(req.headers)
        return base_handler(req)

    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    provider.generate(REQ, tmp_path, 1, lambda t: None)
    assert download_headers, "no download requests were recorded"
    assert all("authorization" not in h for h in download_headers)
