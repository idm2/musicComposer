"""Pre-flight validation — MANDATORY before any request reaches a paid generation API.

Ported from validate_chunk()/preflight() in the video project's
scripts/audiobook/generate_audiobook.py. Money is only spent on a request that passes every
check. If anything fails the run aborts with a per-line report and generates NOTHING.
"""
import re

from ..models import SourceInfo
from .provider import GenerationRequest, ProviderLimits

LINE_VALIDATORS = [
    ("markup", re.compile(r"[<>{}\[\]]")),                       # we add [Section] headers ourselves
    ("replacement-char", re.compile("�")),
    ("soft-hyphen", re.compile("\xad")),
    ("control-char", re.compile(r"[\x00-\x08\x0b-\x1f]")),
    ("placeholder", re.compile(r"(?i)\b(lorem ipsum|todo|tbd|placeholder|insert [a-z ]+ here|lyrics? (go|goes) here)\b")),
]
STYLE_OF = re.compile(r"(?i)\b(in the style of|sounds? like|à la|a la)\b")
MIN_WPS, MAX_WPS = 0.2, 4.5          # sung words per second; above this the model truncates or garbles, below it the section is mostly empty
SHINGLE = 6                          # consecutive shared words that count as copying the reference


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def banned_terms(source: SourceInfo | None) -> list[str]:
    """Artist / song names from the reference's provenance. They must never reach a style prompt."""
    if source is None:
        return []
    terms: list[str] = []
    for raw in filter(None, [source.uploader, source.title]):
        cleaned = re.sub(r"[\(\[].*?[\)\]]", "", raw)
        for piece in re.split(r"\s+[-–—|]\s+", cleaned):
            piece = piece.strip()
            if len(piece) >= 4 and piece not in terms:
                terms.append(piece)
    return terms


def validate_request(req: GenerationRequest, limits: ProviderLimits,
                     reference_lyrics: list[str], banned: list[str]) -> list[str]:
    problems: list[str] = []
    if not 1 <= len(req.title.strip()) <= limits.max_title_chars:
        problems.append(f"title: must be 1–{limits.max_title_chars} chars, got {len(req.title.strip())}")
    if not req.style_prompt.strip():
        problems.append("style_prompt: empty")
    if len(req.style_prompt) > limits.max_style_chars:
        problems.append(f"style_prompt: {len(req.style_prompt)} chars > limit {limits.max_style_chars}")
    if len(req.negative_style) > limits.max_style_chars:
        problems.append(f"negative_style: {len(req.negative_style)} chars > limit {limits.max_style_chars}")

    for field, text in (("style_prompt", req.style_prompt), ("negative_style", req.negative_style)):
        style_of = STYLE_OF.search(text)
        if style_of:
            problems.append(f"{field}: says '{style_of.group(0)}' — describe the sound, do not name a style of someone")

    for field, text in (("title", req.title), ("style_prompt", req.style_prompt), ("negative_style", req.negative_style)):
        for term in banned:
            if re.search(rf"(?i)\b{re.escape(term)}\b", text):
                problems.append(f"{field}: contains banned term {term!r} (the reference's artist/title)")
        for name, rx in LINE_VALIDATORS:
            m = rx.search(text)
            if m:
                problems.append(f"{field}: {name}: ...{text[max(0, m.start() - 20):m.end() + 20]}...")

    if not 2 <= len(req.sections) <= limits.max_sections:
        problems.append(f"sections: need at least 2 sections and at most {limits.max_sections}, got {len(req.sections)}")
    if sum(len(s.lines) for s in req.sections) < 4:
        problems.append("lyrics: no lyrics to sing (fewer than 4 lines in total)")

    ref = _norm_words(" ".join(reference_lyrics))
    ref_shingles = {tuple(ref[i:i + SHINGLE]) for i in range(len(ref) - SHINGLE + 1)}

    for s in req.sections:
        if not s.name.strip():
            problems.append("section: empty name")
        if not limits.min_section_s <= s.duration_s <= limits.max_section_s:
            problems.append(f"{s.name}: duration {s.duration_s}s outside {limits.min_section_s}–{limits.max_section_s}s")
        if len(s.lines) > limits.max_lines_per_section:
            problems.append(f"{s.name}: {len(s.lines)} lines > limit {limits.max_lines_per_section}")
        words = sum(len(_norm_words(ln)) for ln in s.lines)
        if s.lines and s.duration_s > 0 and not MIN_WPS <= words / s.duration_s <= MAX_WPS:
            problems.append(f"{s.name}: {words} words in {s.duration_s}s = {words / s.duration_s:.1f} words/sec "
                            f"(expected {MIN_WPS}–{MAX_WPS})")
        for i, line in enumerate(s.lines, start=1):
            where = f"{s.name} line {i}"
            if not line.strip():
                problems.append(f"{where}: empty-line")
                continue
            if len(line) > limits.max_line_chars:
                problems.append(f"{where}: line-too-long ({len(line)} > {limits.max_line_chars})")
            for name, rx in LINE_VALIDATORS:
                m = rx.search(line)
                if m:
                    problems.append(f"{where}: {name}: ...{line[max(0, m.start() - 20):m.end() + 20]}...")
            w = _norm_words(line)
            if any(tuple(w[j:j + SHINGLE]) in ref_shingles for j in range(len(w) - SHINGLE + 1)):
                problems.append(f"{where}: copies the reference lyrics ({SHINGLE}+ consecutive words): {line!r}")

    total = sum(s.duration_s for s in req.sections)
    if not limits.min_total_s <= total <= limits.max_total_s:
        problems.append(f"duration: sections total {total}s outside {limits.min_total_s}–{limits.max_total_s}s")
    if req.target_duration_s and abs(total - req.target_duration_s) > 0.15 * req.target_duration_s:
        problems.append(f"duration: sections sum to {total}s but target_duration_s is {req.target_duration_s}s (>15% apart)")
    if len(req.lyrics_text()) > limits.max_lyrics_chars:
        problems.append(f"lyrics: {len(req.lyrics_text())} chars > limit {limits.max_lyrics_chars}")
    return problems


def preflight(req: GenerationRequest, limits: ProviderLimits,
              reference_lyrics: list[str], banned: list[str]) -> None:
    problems = validate_request(req, limits, reference_lyrics, banned)
    if problems:
        print(f"PRE-FLIGHT FAILED — {len(problems)} problem(s), NOTHING was generated:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    lines = sum(len(s.lines) for s in req.sections)
    print(f"pre-flight passed: {len(req.sections)} sections, {lines} lines validated clean")
