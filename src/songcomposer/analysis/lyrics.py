"""The lyrics ear: faster-whisper over the isolated vocal stem, word-level timings."""
from pathlib import Path
from typing import Iterable

from ..hashing import content_key, file_sha1
from ..models import Word
from .cache import cached

VERSION = "1"


def words_from_segments(segments: Iterable) -> list[Word]:
    out: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                out.append(Word(word=text, start=round(float(w.start), 3), end=round(float(w.end), 3),
                                confidence=round(min(1.0, max(0.0, float(w.probability))), 3)))
    return out


def _run_whisper(path: Path, model_name: str, hint: str) -> list[dict]:
    import torch  # noqa: F401 — imported FIRST: torch ships the cuDNN/cuBLAS DLLs ctranslate2 needs on Windows
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    segments, _info = model.transcribe(str(path), word_timestamps=True, vad_filter=False,
                                       condition_on_previous_text=False, initial_prompt=hint or None)
    return [w.model_dump() for w in words_from_segments(segments)]


def transcribe_words(vocals: Path, cache_dir: Path, model_name: str, hint: str = "") -> list[Word]:
    version = f"{VERSION}:{model_name}:{content_key(hint)}"
    data = cached(cache_dir, file_sha1(vocals), "lyrics", version, lambda: _run_whisper(Path(vocals), model_name, hint))
    return [Word(**d) for d in data]
