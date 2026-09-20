"""Kie.ai client. Port of kiePost()/kieGet()/pollUnifiedJob() from the video project's lib.mjs."""
import time
from typing import Callable

import httpx

from ..env import require_env

BASE = "https://api.kie.ai/api/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {require_env('KIE_API_KEY')}"}


def _check(res: httpx.Response, what: str) -> dict:
    try:
        data = res.json()
    except ValueError:
        data = {}
    if res.status_code != 200 or data.get("code") not in (None, 200):
        raise RuntimeError(f"Kie.ai {what} failed ({data.get('code', res.status_code)}): {res.text[:500]}")
    return data


def kie_post(path: str, body: dict, client: httpx.Client) -> dict:
    return _check(client.post(f"{BASE}{path}", json=body, headers=_headers()), f"POST {path}")


def kie_get(path: str, client: httpx.Client) -> dict:
    return _check(client.get(f"{BASE}{path}", headers=_headers()), f"GET {path}")


def poll_job(task_id: str, client: httpx.Client, *, label: str = "task", timeout_s: float = 900,
             sleep: Callable[[float], None] = time.sleep) -> dict:
    deadline = time.monotonic() + timeout_s
    delay = 4.0
    while time.monotonic() < deadline:
        data = kie_get(f"/jobs/recordInfo?taskId={task_id}", client).get("data") or {}
        state = data.get("state", "unknown")
        print(f"  [{label}] {state}      ", end="\r")
        if state == "success":
            print()
            return data
        if state == "fail":
            print()
            raise RuntimeError(f"Kie.ai {label} failed: {data.get('failMsg') or data.get('failCode') or 'no detail'}")
        sleep(delay)
        delay = min(delay + 2.0, 12.0)
    raise RuntimeError(f"Kie.ai {label} timed out after {timeout_s}s (taskId {task_id} — it may still finish; "
                       "check the Kie.ai dashboard before regenerating)")
