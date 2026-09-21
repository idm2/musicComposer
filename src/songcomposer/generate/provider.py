"""The provider contract. The audio generator is a swappable module (CLAUDE.md):
whichever service produces the take, the analyser transcribes it identically."""
from pathlib import Path
from typing import Callable, Literal, Protocol

from pydantic import BaseModel

from ..models import SpecSection


class ProviderLimits(BaseModel):
    max_title_chars: int
    max_style_chars: int
    max_lyrics_chars: int
    max_line_chars: int
    max_lines_per_section: int
    max_sections: int
    min_section_s: float
    max_section_s: float
    min_total_s: float
    max_total_s: float
    max_negative_chars: int | None = None   # None: same limit as max_style_chars


# Intersection of Suno V6 (title 80, style 1000, negative 200, lyrics 5000, 10–360 s) and
# ElevenLabs music_v2_5 (30 chunks, 30 lines × 200 chars, chunk 3–120 s, total ≤ 600 s).
GENERIC_LIMITS = ProviderLimits(
    max_title_chars=80, max_style_chars=1000, max_lyrics_chars=5000, max_line_chars=200,
    max_lines_per_section=30, max_sections=30, min_section_s=3, max_section_s=120,
    min_total_s=30, max_total_s=360, max_negative_chars=200)


class GenerationRequest(BaseModel):
    title: str
    style_prompt: str
    negative_style: str = ""
    vocal_gender: Literal["male", "female", "any"] = "any"
    sections: list[SpecSection]
    target_duration_s: float
    n_takes: int = 3

    def lyrics_text(self, with_notes: bool = False) -> str:
        """with_notes folds each section's style_notes into its header — "[Verse 1: sparse, guitar only]" — the
        only per-section direction a lyrics-prompt provider (Suno) ever sees."""
        def header(s: SpecSection) -> str:
            notes = s.style_notes.strip() if with_notes else ""
            return f"[{s.name}: {notes}]" if notes else f"[{s.name}]"
        return "\n\n".join("\n".join([header(s)] + s.lines) for s in self.sections)


class CostEstimate(BaseModel):
    usd: float
    basis: str


class TakeResult(BaseModel):
    provider_ref: str
    path: Path
    duration_s: float


class Provider(Protocol):
    name: str
    limits: ProviderLimits

    def payload(self, req: GenerationRequest) -> dict: ...
    def estimate(self, req: GenerationRequest) -> CostEstimate: ...
    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float: ...
