"""Port of loadEnv()/requireEnv() from the video project's lib.mjs."""
import os
from pathlib import Path

_loaded = False


def load_env(path: Path = Path(".env")) -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    for line in raw.splitlines():
        t = line.strip()
        if not t or t.startswith("#") or "=" not in t:
            continue
        k, v = t.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if not os.environ.get(k):
            os.environ[k] = v


def require_env(name: str) -> str:
    load_env()
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing {name} — add it to .env (see .env.example).")
    return v
