"""THE analysis engine. Runs twice — over the reference on the way in, and over our own
generated take on the way out. Same code path both times (BUILD-SPEC §4)."""
from pathlib import Path

from ..config import Config
from ..hashing import file_sha1
from ..models import Analysis

OBJECTIVE_EARS = ("stems", "beats", "chords", "key", "notes", "lyrics", "structure")
ALL_EARS = ("subjective",) + OBJECTIVE_EARS


def analyze(audio: Path, cache_dir: Path, config: Config, ears: set[str] | None = None,
            lyrics_hint: str = "") -> Analysis:
    ears = set(ALL_EARS) if ears is None else set(ears)
    unknown = ears - set(ALL_EARS)
    if unknown:
        raise ValueError(f"unknown ear(s): {sorted(unknown)} — valid: {', '.join(ALL_EARS)}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result = Analysis(audio_sha1=file_sha1(audio), engines={})

    if "subjective" in ears:
        from .subjective import listen
        print(f"> subjective ear ({config.listener_model})")
        result.subjective = listen(audio, cache_dir, config.listener_model)
        result.engines["subjective"] = config.listener_model

    missing = [e for e in OBJECTIVE_EARS if e not in result.engines]
    if missing:
        print(f"! PARTIAL ANALYSIS — objective ear not run ({', '.join(missing)}). "
              "No chords, key or tempo are available; nothing downstream may invent them.")
    return result
