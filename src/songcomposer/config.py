import tomllib
from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    default_provider: str = "suno"
    writer_model: str = "google/gemini-2.5-pro"
    listener_model: str = "google/gemini-2.5-pro"
    whisper_model: str = "large-v3"
    suno_model: str = "V6"
    elevenlabs_model: str = "music_v2_5"


def load_config(root: Path | None = None) -> Config:
    p = (Path(root) if root else Path.cwd()) / "songcomposer.toml"
    if not p.exists():
        return Config()
    return Config(**tomllib.loads(p.read_text(encoding="utf-8")))
