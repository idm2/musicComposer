import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))   # bytes: no newline translation on Windows


def write_model(path: Path, model: BaseModel) -> None:
    write_json(path, model.model_dump(mode="json"))
