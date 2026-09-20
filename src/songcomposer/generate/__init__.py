"""Stage: generate. The ONLY place paid generation happens.

Order is fixed and non-negotiable (CLAUDE.md hard rule):
  build request → cache check → PRE-FLIGHT → print estimated cost → explicit 'yes' → provider.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config import Config, load_config
from ..hashing import content_key
from ..jsonio import read_json, write_model
from ..models import Analysis, GenerationRun, SongSpec, SourceInfo, TakeRecord, TakesManifest
from ..paths import SongPaths
from .preflight import banned_terms, preflight
from .prompt import FIDELITY_LEVELS, build_request
from .provider import Provider, TakeResult


def get_provider(name: str, config: Config) -> Provider:
    if name == "suno":
        from .suno import SunoProvider
        return SunoProvider(config.suno_model)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsProvider
        return ElevenLabsProvider(config.elevenlabs_model)
    raise ValueError(f"unknown provider {name!r} — use suno or elevenlabs")


def ask_fidelity(input_fn: Callable[[str], str]) -> tuple[str, str]:
    print("How closely should this track the reference?\n"
          "  1  loose   — same genre and mood only\n"
          "  2  medium  — + its tempo, instrumentation and vocal character\n"
          "  3  close   — + its key, meter, production and timbre")
    while True:
        raw = input_fn("Fidelity [1/2/3]: ").strip().lower()
        if raw in ("1", "2", "3"):
            level = FIDELITY_LEVELS[int(raw) - 1]
            break
        if raw in FIDELITY_LEVELS:
            level = raw
            break
    return level, input_fn("Anything to add for this run? (enter to skip): ").strip()


def sanity_check(path: Path, duration_s: float, target_s: float) -> str | None:
    size = path.stat().st_size
    if size < 100_000:
        return f"{path.name}: suspiciously small audio ({size} bytes)"
    if duration_s < max(30.0, 0.5 * target_s) or duration_s > 2.0 * target_s:
        return f"{path.name}: duration {duration_s:.0f}s vs target {target_s:.0f}s"
    return None


def run_generate(song: str, provider_name: str | None = None, regen: bool = False,
                 fidelity: str | None = None, note: str = "",
                 input_fn: Callable[[str], str] = input, provider: Provider | None = None) -> TakesManifest:
    paths = SongPaths(song)
    config = load_config()
    spec = SongSpec(**read_json(paths.require(paths.spec, "compose")))
    analysis = Analysis(**read_json(paths.analysis)) if paths.analysis.exists() else None
    source = SourceInfo(**read_json(paths.source_json)) if paths.source_json.exists() else None
    provider = provider or get_provider(provider_name or config.default_provider, config)

    if fidelity is None:
        fidelity, note = ask_fidelity(input_fn)
    req = build_request(spec, analysis, fidelity, note)
    payload = provider.payload(req)
    request_hash = content_key(provider.name, json.dumps(payload, sort_keys=True, ensure_ascii=False))

    manifest = TakesManifest(**read_json(paths.takes_json)) if paths.takes_json.exists() else TakesManifest()
    same = [r for r in manifest.runs if r.request_hash == request_hash and r.takes]
    if same and not regen and all((paths.takes_dir / t.file).exists() for t in same[0].takes):
        print(f"- {len(same[0].takes)} {provider.name} take(s) already exist for this exact request "
              "(use --regen to pay for new ones)")
        return manifest

    # ---- HARD GATE: nothing below this line runs on unvalidated input -------------------
    preflight(req, provider.limits, [w.word for w in analysis.lyrics] if analysis else [], banned_terms(source))
    est = provider.estimate(req)
    print(f"\nprovider : {provider.name}\nstyle    : {req.style_prompt}\n"
          f"length   : {req.target_duration_s:.0f}s × {req.n_takes} takes\n"
          f"ESTIMATED COST: ${est.usd:.2f}  ({est.basis})")
    if input_fn("Type 'yes' to spend this and generate: ").strip().lower() != "yes":
        print("aborted — nothing was generated, nothing was spent")
        raise SystemExit(1)

    if regen:
        for old in [r for r in manifest.runs if r.provider == provider.name]:
            for t in old.takes:
                (paths.takes_dir / t.file).unlink(missing_ok=True)
            manifest.runs.remove(old)
    start = max([t.index for t in manifest.all_takes()], default=0) + 1
    run = GenerationRun(provider=provider.name, request_hash=request_hash, fidelity=fidelity, fidelity_note=note,
                        style_prompt=req.style_prompt, payload=payload, cost_estimate_usd=est.usd,
                        cost_actual_usd=0.0, created_at=datetime.now(timezone.utc).isoformat(), takes=[])
    manifest.runs.append(run)

    def on_take(result: TakeResult) -> None:
        index = start + len(run.takes)
        assert result.path.name == f"take-{index}.mp3", f"provider wrote {result.path.name}, expected take-{index}.mp3"
        warning = sanity_check(result.path, result.duration_s, req.target_duration_s)
        if warning:
            run.warnings.append(warning)
            print(f"  ! {warning}")
        run.takes.append(TakeRecord(index=index, file=result.path.name, provider=provider.name,
                                    provider_ref=result.provider_ref, duration_s=round(result.duration_s, 2),
                                    bytes=result.path.stat().st_size))
        write_model(paths.takes_json, manifest)          # persisted per take: paid audio is never orphaned
        print(f"  saved take {index} ({result.duration_s:.0f}s) → {result.path}")

    try:
        run.cost_actual_usd = round(provider.generate(req, paths.takes_dir, start, on_take), 4)
    except Exception as e:
        run.warnings.append(f"generation stopped early: {e}")
        raise
    finally:
        write_model(paths.takes_json, manifest)

    spec.fidelity, spec.fidelity_note = fidelity, note
    write_model(paths.spec, spec)
    print(f"done: {len(run.takes)} take(s), actual cost ${run.cost_actual_usd:.2f}. "
          f"Listen, then: songcomposer pick {song} --take N")
    return manifest
