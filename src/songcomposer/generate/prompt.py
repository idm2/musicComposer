"""Fold the per-run fidelity answer into the style prompt (BUILD-SPEC decision #6).

loose  — the spec's own style words only
medium — + measured tempo, heard instrumentation and vocal character
close  — + measured key and meter, heard production and timbre

Tempo/key/meter come ONLY from the objective ear (global_info). If it has not run, they are
simply absent — nothing here invents them.
"""
from ..models import Analysis, SongSpec
from .provider import GenerationRequest

FIDELITY_LEVELS = ("loose", "medium", "close")


def build_request(spec: SongSpec, analysis: Analysis | None, fidelity: str, note: str = "",
                  n_takes: int = 3) -> GenerationRequest:
    if fidelity not in FIDELITY_LEVELS:
        raise ValueError(f"fidelity must be one of {FIDELITY_LEVELS}, got {fidelity!r}")
    parts = [spec.style_prompt.strip()]
    g = analysis.global_info if analysis else None
    s = analysis.subjective if analysis else None
    if fidelity in ("medium", "close"):
        if g:
            parts.append(f"{round(g.tempo_bpm)} BPM")
        if s:
            parts += s.instrumentation[:5]
            parts.append(s.vocal_character)
    if fidelity == "close":
        if g:
            parts.append(f"in {g.key}")
            if g.time_signature != "4/4":
                parts.append(f"{g.time_signature} time")
        if s:
            parts += [s.production, s.timbre]
    if note.strip():
        parts.append(note.strip())
    return GenerationRequest(
        title=spec.title, style_prompt=", ".join(p for p in parts if p),
        negative_style=spec.negative_style, vocal_gender=spec.vocal_gender,
        sections=spec.sections, target_duration_s=spec.target_duration_s, n_takes=n_takes)
