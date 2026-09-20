"""Content-hashed component cache. Same idea as chunk_key() in the audiobook script:
results are keyed by what went in, so unchanged audio is never re-analysed (or re-billed)."""
from pathlib import Path
from typing import Any, Callable

from ..hashing import content_key
from ..jsonio import read_json, write_json


def cached(cache_dir: Path, audio_sha1: str, component: str, version: str, fn: Callable[[], Any]) -> Any:
    path = Path(cache_dir) / f"{component}-{content_key(audio_sha1, component, version)}.json"
    if path.exists():
        return read_json(path)
    result = fn()
    write_json(path, result)
    return result
