import hashlib
from pathlib import Path


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_key(*parts: str) -> str:
    """Same idea as chunk_key() in the audiobook script: stable short key over content."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
