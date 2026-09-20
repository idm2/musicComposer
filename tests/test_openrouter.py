import base64
import json

import httpx
import pytest
from pydantic import BaseModel

from songcomposer.clients import openrouter


class Inner(BaseModel):
    n: int


class Outer(BaseModel):
    name: str
    items: list[Inner]


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def _reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_audio_part_is_base64_mp3(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"\x00\x01binary")
    part = openrouter.audio_part(f)
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mp3"
    assert base64.b64decode(part["input_audio"]["data"]) == b"\x00\x01binary"


def test_inline_refs_removes_defs():
    schema = openrouter.inline_refs(Outer.model_json_schema())
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    assert schema["properties"]["items"]["items"]["properties"]["n"]["type"] == "integer"


def test_chat_sends_auth_model_and_schema():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return _reply("hi")

    out = openrouter.chat([{"role": "user", "content": "x"}], "m/x", schema={"title": "T", "type": "object"},
                          client=_client(handler))
    assert out == "hi" and seen["auth"] == "Bearer sk-test" and seen["body"]["model"] == "m/x"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["provider"] == {"require_parameters": True}


def test_chat_raises_on_http_error_with_body():
    with pytest.raises(RuntimeError, match="402"):
        openrouter.chat([], "m", client=_client(lambda r: httpx.Response(402, json={"error": "no credit"})))


def test_chat_json_strips_fences_and_validates():
    body = '```json\n{"name": "a", "items": [{"n": 1}]}\n```'
    got = openrouter.chat_json([], "m", Outer, client=_client(lambda r: _reply(body)))
    assert got == Outer(name="a", items=[Inner(n=1)])


def test_chat_json_retries_with_validation_error_then_succeeds():
    replies = iter(['{"name": "a"}', '{"name": "a", "items": []}'])
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return _reply(next(replies))

    got = openrouter.chat_json([{"role": "user", "content": "go"}], "m", Outer, client=_client(handler))
    assert got.items == [] and len(bodies) == 2
    assert "failed validation" in bodies[1]["messages"][-1]["content"]


def test_chat_json_gives_up_after_retries():
    with pytest.raises(RuntimeError, match="valid Outer"):
        openrouter.chat_json([], "m", Outer, retries=1, client=_client(lambda r: _reply("{}")))
