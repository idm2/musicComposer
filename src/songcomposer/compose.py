"""Stage: compose. Brief + analysis → an ORIGINAL song: lyrics, section map, style directives."""
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from .clients.openrouter import chat_json
from .config import load_config
from .generate.preflight import banned_terms, validate_request
from .generate.prompt import build_request
from .generate.provider import GENERIC_LIMITS
from .jsonio import read_json, write_model
from .models import LOW_CONFIDENCE, Analysis, Brief, SongSpec, SourceInfo, SpecSection
from .paths import SongPaths


class ComposedSong(BaseModel):
    title: str
    style_prompt: str
    negative_style: str
    vocal_gender: Literal["male", "female", "any"]
    target_duration_s: float
    sections: list[SpecSection]


# Shared by both prompts below — SYSTEM (a reference exists) and SYSTEM_NO_REFERENCE (brief only).
# Written to stand on its own, so it never needs to name a reference either way.
SONGWRITING_RULES = """Rules:
- Original lyrics only. Never quote or closely paraphrase an existing song's lyrics.
- Concrete images over abstractions. Singable lines: natural stresses, 4-10 words per line, consistent
  metre within a section, real rhymes or deliberate near-rhymes. A chorus that earns its repetition.
- style_prompt: 10-25 words describing genre, mood, instrumentation and vocal delivery for a music
  generator. NEVER name an artist, band or song, and never write "in the style of".
- negative_style: comma-separated things to avoid.
- sections: in performance order. name like "Intro", "Verse 1", "Pre-Chorus", "Chorus", "Bridge", "Outro".
  Instrumental sections have an empty lines list. duration_s per section between 5 and 110 seconds;
  allow roughly 2-3 sung words per second. Section durations MUST sum to target_duration_s.
  style_notes: a short arrangement direction for that section ("drums drop out", "full band, harmonies").
- No brackets, braces or angle brackets inside lyric lines. Max 200 characters per line, max 30 lines per section.
- target_duration_s: follow the brief; otherwise 150-210."""

SYSTEM = f"""You are a professional songwriter. Write an ORIGINAL song that lives in the same musical
world as the reference described below, fulfilling the user's brief.

{SONGWRITING_RULES}"""

SYSTEM_NO_REFERENCE = f"""You are a professional songwriter. Write an ORIGINAL song that fulfils the user's
brief below, in whatever musical world best serves it — there is nothing else to go on, so invent it
from the ground up.

{SONGWRITING_RULES}"""


def summarise_analysis(a: Analysis) -> str:
    out = []
    g = a.global_info
    measured = []
    if g and g.key is not None:
        measured.append(f"key {g.key} (confidence {g.key_confidence:.2f})")
    if g and g.tempo_bpm is not None:
        measured.append(f"{g.tempo_bpm:.0f} BPM")
    if g and g.time_signature is not None:
        measured.append(g.time_signature)
    if g and g.duration_s is not None:
        measured.append(f"{g.duration_s:.0f}s long")
    if measured:
        out.append("Measured: " + ", ".join(measured) + ".")
    else:
        out.append("Key, tempo and chords: not measured (objective ear has not run). Do not assume any.")
    solid = [c.symbol for c in a.chords if c.confidence >= LOW_CONFIDENCE]
    if solid:
        vocab = ", ".join(sym for sym, _ in Counter(solid).most_common(8))
        out.append(f"Chord vocabulary (confident detections only): {vocab}")
    if a.sections:
        out.append("Form: " + " → ".join(f"{s.label} ({s.end - s.start:.0f}s)" for s in a.sections))
    if a.subjective:
        s = a.subjective
        out += [f"Genre: {', '.join(s.genre_tags)}", f"Instrumentation: {'; '.join(s.instrumentation)}",
                f"Timbre: {s.timbre}", f"Vocal: {s.vocal_character} ({s.vocal_gender})",
                f"Production: {s.production}", f"Emotional arc: {s.emotional_arc}"]
        if s.arrangement_density:
            out.append("Arrangement over time: " + "; ".join(
                f"{d.start:.0f}-{d.end:.0f}s {d.description}" for d in s.arrangement_density))
    if a.lyrics:
        out.append("Reference lyrics (for theme and prosody ONLY — do not reuse any line): "
                   + " ".join(w.word for w in a.lyrics)[:1500])
    return "\n".join(out)


def run_compose(song: str, force: bool = False) -> SongSpec:
    paths = SongPaths(song)
    if paths.spec.exists() and not force:
        print(f"- spec exists at {paths.spec} (use --force to rewrite)")
        return SongSpec(**read_json(paths.spec))
    analysis = Analysis(**read_json(paths.analysis)) if paths.analysis.exists() else None
    brief = Brief(**read_json(paths.require(paths.brief, "brief")))
    source = SourceInfo(**read_json(paths.source_json)) if paths.source_json.exists() else None
    config = load_config()
    banned = banned_terms(source)
    ref_lyrics = [w.word for w in analysis.lyrics] if analysis else []

    if analysis is not None:
        system = SYSTEM
        user = f"## Brief\n{brief.text}\n\n## The reference, as analysed\n{summarise_analysis(analysis)}"
    else:
        system = SYSTEM_NO_REFERENCE
        user = f"## Brief\n{brief.text}"
        print("> composing without a reference — from the brief alone")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    print(f"> composing with {config.writer_model}")
    for attempt in range(2):
        composed = chat_json(messages, config.writer_model, ComposedSong)
        spec = SongSpec(**composed.model_dump(), writer_model=config.writer_model)
        problems = validate_request(build_request(spec, None, "loose"), GENERIC_LIMITS, ref_lyrics, banned)
        if not problems:
            break
        print(f"  draft {attempt + 1} has {len(problems)} pre-flight problem(s)" + ("; asking for a fix" if attempt == 0 else ""))
        messages = messages[:2] + [{"role": "user", "content": user + "\n\n## Your previous draft failed these checks — fix them\n"
                                    + "\n".join(problems) + "\n\nPrevious draft:\n" + composed.model_dump_json()}]
    write_model(paths.spec, spec)
    print(f"  “{spec.title}” — {len(spec.sections)} sections, {len(spec.all_lines())} lines, "
          f"{spec.target_duration_s:.0f}s → {paths.spec}")
    if problems:
        print("! spec still has problems — edit 03-spec.json by hand; `generate` will refuse it as is:")
        for p in problems:
            print(f"    {p}")
    return spec
