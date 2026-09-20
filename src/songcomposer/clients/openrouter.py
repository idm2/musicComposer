"""OpenRouter chat client. Port of the OpenRouter section of the video project's lib.mjs."""
import base64
import copy
import re
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..env import require_env

BASE = "https://openrouter.ai/api/v1"
T = TypeVar("T", bound=BaseModel)


def audio_part(mp3_path: Path) -> dict:
    data = base64.b64encode(Path(mp3_path).read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}}


def inline_refs(schema: dict) -> dict:
    """Gemini's schema support is patchy with $ref — inline every definition."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def chat(messages: list[dict], model: str, *, schema: dict | None = None,
         client: httpx.Client | None = None) -> str:
    body: dict = {"model": model, "messages": messages}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": schema.get("title", "response"), "strict": True, "schema": schema}}
        body["provider"] = {"require_parameters": True}
    own = client is None
    client = client or httpx.Client(timeout=600.0)
    try:
        res = client.post(f"{BASE}/chat/completions", json=body, headers={
            "Authorization": f"Bearer {require_env('OPENROUTER_API_KEY')}"})
    finally:
        if own:
            client.close()
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter {model} failed ({res.status_code}): {res.text[:500]}")
    data = res.json()
    if "error" in data:
        raise RuntimeError(f"OpenRouter {model} error: {str(data['error'])[:500]}")
    return data["choices"][0]["message"]["content"]


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def chat_json(messages: list[dict], model: str, out_type: type[T], *, retries: int = 2,
              client: httpx.Client | None = None) -> T:
    schema = inline_refs(out_type.model_json_schema())
    msgs = list(messages)
    last: Exception | None = None
    for _ in range(retries + 1):
        text = chat(msgs, model, schema=schema, client=client)
        try:
            return out_type.model_validate_json(_strip_fences(text))
        except ValidationError as e:
            last = e
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."}]
    raise RuntimeError(f"model never returned a valid {out_type.__name__}: {last}")
