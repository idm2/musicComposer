"""ElevenLabs Music. Adapted from the video project's scripts/ai-video/generate-music.mjs:
three takes, v2-family model, per-section control through a chunks[] composition plan."""
from pathlib import Path
from typing import Callable

import httpx

from ..audio import probe_duration
from ..clients.elevenlabs import eleven_post
from ..jsonio import write_json
from .provider import CostEstimate, GenerationRequest, ProviderLimits, TakeResult

USD_PER_MINUTE = 0.15            # provisional — elevenlabs.io/pricing/api, 2026-09-20
SEEDS = [11, 22, 33, 44, 55, 66]
MAX_STYLES = 50


def style_tags(text: str) -> list[str]:
    tags: list[str] = []
    for raw in text.split(","):
        tag = raw.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:MAX_STYLES]


class ElevenLabsProvider:
    name = "elevenlabs"
    limits = ProviderLimits(max_title_chars=80, max_style_chars=1000, max_lyrics_chars=20000, max_line_chars=200,
                            max_lines_per_section=30, max_sections=30, min_section_s=3, max_section_s=120,
                            min_total_s=30, max_total_s=600)

    def __init__(self, model: str, client: httpx.Client | None = None):
        if not model.startswith("music_v2"):
            raise ValueError(f"{model}: only the v2 family is supported — it takes the chunks[] plan this provider builds")
        self.model, self._client = model, client

    def payload(self, req: GenerationRequest) -> dict:
        positive, negative = style_tags(req.style_prompt), style_tags(req.negative_style)
        chunks = []
        for s in req.sections:
            chunk = {"text": "\n".join([f"[{s.name}]"] + s.lines), "duration_ms": int(round(s.duration_s * 1000)),
                     "positive_styles": (positive + [t for t in style_tags(s.style_notes) if t not in positive])[:MAX_STYLES],
                     "context_adherence": "high"}
            if negative:
                chunk["negative_styles"] = negative
            chunks.append(chunk)
        return {"model_id": self.model, "composition_plan": {"chunks": chunks}, "seeds": SEEDS[:req.n_takes]}

    def estimate(self, req: GenerationRequest) -> CostEstimate:
        minutes = sum(s.duration_s for s in req.sections) / 60
        return CostEstimate(usd=req.n_takes * minutes * USD_PER_MINUTE,
                            basis=f"{req.n_takes} takes × {minutes:.1f} min × ${USD_PER_MINUTE}/min, provisional pricing")

    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float:
        own = self._client is None
        client = self._client or httpx.Client(timeout=900.0)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            spec = self.payload(req)
            minutes = 0.0
            for i, seed in enumerate(spec["seeds"]):
                print(f"  [elevenlabs] take {i + 1}/{len(spec['seeds'])} (seed {seed}) …")
                body = {"model_id": spec["model_id"], "composition_plan": spec["composition_plan"], "seed": seed}
                res = eleven_post("/music", body, client, params={"output_format": "mp3_44100_192"})
                if res.status_code != 200:
                    self._raise(res, dest_dir)
                path = dest_dir / f"take-{start_index + i}.mp3"
                path.write_bytes(res.content)
                duration = probe_duration(path)
                minutes += duration / 60
                on_take(TakeResult(provider_ref=res.headers.get("song-id", f"seed-{seed}"), path=path, duration_s=duration))
            return minutes * USD_PER_MINUTE
        finally:
            if own:
                client.close()

    @staticmethod
    def _raise(res: httpx.Response, dest_dir: Path) -> None:
        try:
            detail = res.json().get("detail") or {}
        except ValueError:
            detail = {}
        status = detail.get("status") if isinstance(detail, dict) else None
        suggestion = (detail.get("data") or {}).get("composition_plan_suggestion") if isinstance(detail, dict) else None
        if suggestion:
            out = dest_dir / "elevenlabs-plan-suggestion.json"
            write_json(out, suggestion)
            raise RuntimeError(f"ElevenLabs rejected the plan ({status}). Its suggested replacement is saved at {out} — "
                               "compare it with 03-spec.json, edit the spec, and re-run.")
        raise RuntimeError(f"ElevenLabs music failed ({res.status_code}, {status}): {res.text[:500]}")
