"""Polyphonic note transcription with basic-pitch, one stem at a time. Amplitude is carried as confidence."""
from pathlib import Path

from ..hashing import file_sha1
from ..models import Note
from .cache import cached
from .stems import Stems

VERSION = "1"
RANGES = {"vocals": (45, 88), "bass": (24, 60), "other": (36, 96)}       # MIDI; outside = transcription noise


def to_monophonic(notes: list[Note]) -> list[Note]:
    kept: list[Note] = []
    for note in sorted(notes, key=lambda x: (x.onset, -x.confidence)):
        if kept:
            prev = kept[-1]
            overlap = prev.onset + prev.duration - note.onset
            if overlap > 0.5 * min(prev.duration, note.duration):       # genuinely simultaneous: keep the surer one
                if note.confidence > prev.confidence:
                    kept[-1] = note
                continue
            if overlap > 0:                                             # a tail running into the next note
                kept[-1] = prev.model_copy(update={"duration": round(note.onset - prev.onset, 4)})
        kept.append(note)
    return kept


def transcribe_stem(path: Path, stem: str, cache_dir: Path) -> list[Note]:
    def work() -> list[dict]:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import predict
        _, _, events = predict(str(Path(path).resolve()), ICASSP_2022_MODEL_PATH)
        lo, hi = RANGES[stem]
        return [Note(pitch=int(p), onset=round(float(s), 4), duration=round(float(e - s), 4),
                     confidence=round(min(1.0, max(0.0, float(amp))), 3), stem=stem).model_dump()
                for s, e, p, amp, _bends in events if lo <= int(p) <= hi]

    return [Note(**d) for d in cached(cache_dir, file_sha1(path), f"notes-{stem}", VERSION, work)]


def transcribe_notes(stems: Stems, cache_dir: Path) -> list[Note]:
    out = to_monophonic(transcribe_stem(stems.vocals, "vocals", cache_dir))
    out += to_monophonic(transcribe_stem(stems.bass, "bass", cache_dir))
    out += transcribe_stem(stems.other, "other", cache_dir)
    return sorted(out, key=lambda x: x.onset)
