"""Suno via Kie.ai. Custom mode: our lyrics are sung exactly; style and title are ours."""
import json
import math
import time
from pathlib import Path
from typing import Callable

import httpx

from ..audio import probe_duration
from ..clients.kie import kie_get, kie_post, poll_job
from ..jsonio import write_json
from .provider import CostEstimate, GenerationRequest, ProviderLimits, TakeResult

TRACKS_PER_REQUEST = 2
CREDITS_PER_REQUEST = 12        # provisional — from kie.ai/suno-api marketing page, 2026-09-20
USD_PER_CREDIT = 0.005          # provisional — inferred from "12 credits ≈ $0.06"


def extract_tracks(result: dict) -> list[dict]:
    """resultJson's shape for Suno tasks is not documented. Accept every shape Kie documents anywhere."""
    if isinstance(result.get("resultUrls"), list):
        return [{"ref": url.rsplit("/", 1)[-1], "url": url} for url in result["resultUrls"]]
    for items in (result.get("sunoData"), result.get("data"), (result.get("response") or {}).get("sunoData")):
        if isinstance(items, list):
            tracks = [{"ref": str(it.get("id", "")), "url": it.get("audio_url") or it.get("audioUrl"),
                       "duration": it.get("duration")} for it in items if isinstance(it, dict)]
            tracks = [t for t in tracks if t["url"]]
            if tracks:
                return tracks
    return []


class SunoProvider:
    name = "suno"
    limits = ProviderLimits(max_title_chars=80, max_style_chars=1000, max_lyrics_chars=5000, max_line_chars=200,
                            max_lines_per_section=60, max_sections=30, min_section_s=3, max_section_s=360,
                            min_total_s=30, max_total_s=360,
                            max_negative_chars=200)   # Kie.ai 422: "negativeStyle cannot exceed 200 characters"

    def __init__(self, model: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self._client, self._sleep = model, client, sleep

    def _n_requests(self, req: GenerationRequest) -> int:
        return math.ceil(req.n_takes / TRACKS_PER_REQUEST)

    def payload(self, req: GenerationRequest) -> dict:
        inp = {"prompt": req.lyrics_text(), "custom_mode": True, "instrumental": False, "model": self.model,
               "style": req.style_prompt, "title": req.title,
               "duration": int(min(360, max(10, round(req.target_duration_s))))}
        if req.negative_style:
            inp["negative_tags"] = req.negative_style
        if req.vocal_gender != "any":
            inp["vocal_gender"] = "m" if req.vocal_gender == "male" else "f"
        return {"n_requests": self._n_requests(req), "body": {"model": "ai-music-api/generate", "input": inp}}

    def estimate(self, req: GenerationRequest) -> CostEstimate:
        n = self._n_requests(req)
        return CostEstimate(usd=n * CREDITS_PER_REQUEST * USD_PER_CREDIT,
                            basis=f"{n} Suno requests × {CREDITS_PER_REQUEST} credits, provisional pricing; "
                                  f"yields {n * TRACKS_PER_REQUEST} takes — Suno returns pairs")

    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float:
        own = self._client is None
        client = self._client or httpx.Client(timeout=120.0)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            spec = self.payload(req)
            before = kie_get("/chat/credit", client).get("data")
            print(f"  Kie.ai credit balance: {before}")
            task_ids = [kie_post("/jobs/createTask", spec["body"], client)["data"]["taskId"]
                        for _ in range(spec["n_requests"])]
            index = start_index
            for task_id in task_ids:
                data = poll_job(task_id, client, label=f"suno {task_id[:8]}", sleep=self._sleep)
                raw = dest_dir / f"raw-suno-{task_id}.json"
                write_json(raw, data)                               # BEFORE parsing: the paid URLs are in here
                tracks = extract_tracks(json.loads(data.get("resultJson") or "{}"))
                if not tracks:
                    raise RuntimeError(f"Suno succeeded but no audio URLs were recognised — the paid result is "
                                       f"saved at {raw}; add its shape to extract_tracks()")
                for track in tracks:
                    path = dest_dir / f"take-{index}.mp3"
                    part = path.with_name(path.name + ".part")
                    try:
                        # No Authorization header here: the URL points at an arbitrary CDN
                        # host named by the provider's result JSON, never at api.kie.ai —
                        # the key must not be sent to it, even across a redirect.
                        with client.stream("GET", track["url"], follow_redirects=True) as res:
                            res.raise_for_status()
                            with open(part, "wb") as f:
                                for block in res.iter_bytes():
                                    f.write(block)
                        part.replace(path)                          # atomic; Path.rename raises FileExistsError
                                                                     # on Windows when `path` already exists
                    except Exception:
                        part.unlink(missing_ok=True)
                        raise
                    duration = track.get("duration") or probe_duration(path)
                    on_take(TakeResult(provider_ref=track["ref"], path=path, duration_s=float(duration)))
                    index += 1
            after = kie_get("/chat/credit", client).get("data")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before >= after:
                return (before - after) * USD_PER_CREDIT
            return self.estimate(req).usd
        finally:
            if own:
                client.close()
