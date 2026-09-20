"""lyrics.json — the JSON contract the video project consumes to render lyric videos. No code coupling."""
from ..models import Transcription


def build_lyrics_json(song: str, title: str, t: Transcription) -> dict:
    g = t.analysis.global_info
    timed = [l for l in t.lines if l.start is not None]
    return {
        "schema_version": 1, "song": song, "title": title, "audio": f"{song}.mp3",
        "duration_s": g.duration_s if g else None,
        "lines": [{"section": l.section, "text": l.text, "start": l.start, "end": l.end, "confidence": l.confidence,
                   "words": [w.model_dump() for w in l.words]} for l in timed],
        "unaligned_lines": [l.text for l in t.lines if l.start is None],
    }
