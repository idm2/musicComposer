"""THE analysis engine. Runs twice — over the reference on the way in, and over our own
generated take on the way out. Same code path both times (BUILD-SPEC §4)."""
import importlib.util
from pathlib import Path

from ..config import Config
from ..hashing import file_sha1
from ..models import Analysis, GlobalInfo

OBJECTIVE_EARS = ("stems", "beats", "chords", "key", "notes", "lyrics", "structure")
ALL_EARS = ("subjective",) + OBJECTIVE_EARS
NEEDS_STEMS = {"chords", "key", "notes", "lyrics", "structure"}


def objective_installed() -> bool:
    return all(importlib.util.find_spec(m) for m in ("demucs", "beat_this", "lv_chordia", "basic_pitch", "faster_whisper", "librosa"))


def analyze(audio: Path, cache_dir: Path, config: Config, ears: set[str] | None = None,
            lyrics_hint: str = "") -> Analysis:
    from . import beats as beats_mod, chords as chords_mod, fuse, key as key_mod, lyrics as lyrics_mod
    from . import notes as notes_mod, stems as stems_mod, structure as structure_mod, subjective as subjective_mod

    if ears is None:
        ears = set(ALL_EARS) if objective_installed() else {"subjective"}
    ears = set(ears)
    unknown = ears - set(ALL_EARS)
    if unknown:
        raise ValueError(f"unknown ear(s): {sorted(unknown)} — valid: {', '.join(ALL_EARS)}")
    if ears & set(OBJECTIVE_EARS) and not objective_installed():
        raise RuntimeError("objective ear requested but the analysis stack is not installed — run `uv sync --extra analysis`")
    if ears & NEEDS_STEMS:
        ears.add("stems")
    if "structure" in ears:
        ears.add("beats")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result = Analysis(audio_sha1=file_sha1(audio), engines={})

    if "subjective" in ears:
        print(f"> subjective ear ({config.listener_model})")
        result.subjective = subjective_mod.listen(audio, cache_dir, config.listener_model)
        result.engines["subjective"] = config.listener_model

    stems = grid = measured = None
    if "stems" in ears:
        print("> stems (demucs)")
        stems = stems_mod.separate(audio, cache_dir)
        result.engines["stems"] = stems_mod.MODEL
    if "beats" in ears:
        print("> beats (beat_this)")
        grid = beats_mod.track_beats(audio, cache_dir)
        result.beats, result.downbeats = grid.beats, grid.downbeats
        result.engines["beats"] = "beat_this"
    if "chords" in ears:
        print("> chords (lv-chordia ×2 + spectral support)")
        found = chords_mod.detect_chords(stems.harmonic, audio, cache_dir)
        result.chords = fuse.merge_adjacent(fuse.snap_to_beats(found, grid.beats if grid else []))
        result.engines["chords"] = "lv-chordia 1.1.0"
    if "key" in ears:
        print("> key + loudness")
        measured = key_mod.measure(stems.harmonic, audio, cache_dir)
        result.engines["key"] = "krumhansl/librosa"
    if "notes" in ears:
        print("> notes (basic-pitch)")
        result.notes = notes_mod.transcribe_notes(stems, cache_dir)
        result.engines["notes"] = "basic-pitch 0.4.0"
    if "lyrics" in ears:
        print(f"> lyrics (whisper {config.whisper_model})")
        result.lyrics = lyrics_mod.transcribe_words(stems.vocals, cache_dir, config.whisper_model, hint=lyrics_hint)
        result.engines["lyrics"] = f"faster-whisper {config.whisper_model}"
    if grid or measured:
        # Build from whichever ear actually ran — `key` without `beats` (or vice versa) must not
        # throw away a computed confidence just because the OTHER half of GlobalInfo is missing.
        result.global_info = GlobalInfo(
            key=measured["key"] if measured else None,
            key_confidence=measured["key_confidence"] if measured else None,
            tempo_bpm=grid.tempo_bpm if grid else None,
            time_signature=grid.time_signature if grid else None,
            loudness_lufs=measured["loudness_lufs"] if measured else None,
            duration_s=measured["duration_s"] if measured else None)
    if "structure" in ears:
        print("> structure")
        duration = measured["duration_s"] if measured else (grid.beats[-1] if grid.beats else 0.0)
        guesses = result.subjective.sections if result.subjective else []
        result.sections = structure_mod.label_sections(
            structure_mod.boundaries(stems.harmonic, grid.downbeats, duration), guesses)
        result.engines["structure"] = "librosa agglomerative + heard labels"

    missing = [e for e in OBJECTIVE_EARS if e not in result.engines]
    if missing:
        print(f"! PARTIAL ANALYSIS — objective ear not run ({', '.join(missing)}). "
              "No chords, key or tempo are available for those; nothing downstream may invent them.")
    return result
