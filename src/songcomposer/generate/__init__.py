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

MIN_TAKES = 1
MAX_TAKES = 6      # ElevenLabs' seed-list length — going higher would silently give fewer takes than asked


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


def ask_takes(input_fn: Callable[[str], str]) -> int:
    print(f"How many takes should we generate? ({MIN_TAKES}-{MAX_TAKES}, enter for the default of 3)")
    while True:
        raw = input_fn(f"Takes [{MIN_TAKES}-{MAX_TAKES}, default 3]: ").strip()
        if raw == "":
            return 3
        if raw.isdigit() and MIN_TAKES <= int(raw) <= MAX_TAKES:
            return int(raw)


def sanity_check(path: Path, duration_s: float, target_s: float) -> str | None:
    size = path.stat().st_size
    if size < 100_000:
        return f"{path.name}: suspiciously small audio ({size} bytes)"
    if duration_s < max(30.0, 0.5 * target_s) or duration_s > 2.0 * target_s:
        return f"{path.name}: duration {duration_s:.0f}s vs target {target_s:.0f}s"
    return None


def run_generate(song: str, provider_name: str | None = None, regen: bool = False,
                 fidelity: str | None = None, note: str = "", takes: int | None = None,
                 input_fn: Callable[[str], str] = input, provider: Provider | None = None) -> TakesManifest:
    paths = SongPaths(song)
    config = load_config()
    spec = SongSpec(**read_json(paths.require(paths.spec, "compose")))
    analysis = Analysis(**read_json(paths.analysis)) if paths.analysis.exists() else None
    source = SourceInfo(**read_json(paths.source_json)) if paths.source_json.exists() else None
    provider = provider or get_provider(provider_name or config.default_provider, config)

    if takes is not None and not MIN_TAKES <= takes <= MAX_TAKES:
        raise ValueError(f"takes must be between {MIN_TAKES} and {MAX_TAKES}, got {takes}")

    if analysis is None:
        # Fidelity means "how closely to track the reference" — meaningless with none to track.
        if fidelity is not None:
            raise ValueError("--fidelity has nothing to track without a reference (no 01-analysis.json) — "
                             "omit it to compose from the brief alone")
        fidelity, note = "loose", "no reference — composed from the brief alone"
    elif fidelity is None:
        fidelity, note = ask_fidelity(input_fn)
        if takes is None:
            takes = ask_takes(input_fn)
    if takes is None:
        takes = 3

    req = build_request(spec, analysis, fidelity, note, n_takes=takes)
    payload = provider.payload(req)
    # n_takes is folded into the hash explicitly: a provider's own payload can be identical for two
    # different take counts (Suno needs 2 requests for both 3 and 4 takes), which would otherwise let
    # a smaller run silently reuse a larger run's cached takes.
    request_hash = content_key(provider.name, json.dumps({"n_takes": req.n_takes, **payload},
                                                          sort_keys=True, ensure_ascii=False))

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
          f"length   : {req.target_duration_s:.0f}s\n"
          f"requested: {req.n_takes} take(s)\n"
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
        if result.path.name != f"take-{index}.mp3":
            raise RuntimeError(f"provider wrote {result.path.name}, expected take-{index}.mp3")
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
        # The provider may have submitted (and been charged for) the whole request up front,
        # even though few or no takes landed — a cost ledger must never under-state spend.
        run.cost_actual_usd = round(est.usd, 4)
        run.warnings.append("actual cost unknown after failure — recorded the full estimate as an upper bound")
        raise
    finally:
        write_model(paths.takes_json, manifest)

    spec.fidelity, spec.fidelity_note = fidelity, note
    write_model(paths.spec, spec)
    print(f"done: {len(run.takes)} take(s), actual cost ${run.cost_actual_usd:.2f}. "
          f"Listen, then: songcomposer pick {song} --take N")
    return manifest
