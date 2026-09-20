"""The subjective ear — an audio-native LLM. Hears tone, timbre, vocal character and mood.

It hallucinates chords with total confidence, so it is never asked for them and the
Subjective model has no field that could hold one.
"""
from pathlib import Path

from ..audio import to_mp3
from ..clients.openrouter import audio_part, chat_json
from ..hashing import file_sha1
from ..models import Subjective
from .cache import cached

VERSION = "1"

SYSTEM = """You are an expert recording engineer and A&R listener. You will be given a song.
Describe ONLY what can be heard. Be specific and concrete; avoid generic praise.

Do NOT name chords, chord progressions, the key, or the tempo in BPM — a separate DSP system
measures those and your guesses would be wrong. Do NOT name the artist or song even if you recognise it.

Fields:
- genre_tags: 2-6 short genre/style tags.
- instrumentation: each audible instrument WITH how it is played ("fingerpicked nylon guitar", "brushed snare").
- timbre: the overall sonic colour.
- vocal_character: grain, register, phrasing, vibrato, restraint vs belt, doubling/harmonies.
- vocal_gender: male | female | mixed | none | unclear.
- production: space/reverb, compression, stereo width, era it evokes, lo-fi vs polished.
- emotional_arc: how the feeling moves from start to end.
- arrangement_density: spans (seconds) describing what enters/leaves ("0-22: solo guitar", "22-60: + bass, brushed kit").
- sections: your best reading of the form as labelled spans in seconds. Labels from:
  intro, verse, pre-chorus, chorus, post-chorus, bridge, solo, instrumental, outro."""


def listen(audio: Path, cache_dir: Path, model: str) -> Subjective:
    sha = file_sha1(audio)

    def work() -> dict:
        mp3 = Path(cache_dir) / f"listen-{sha[:16]}.mp3"
        if not mp3.exists():
            to_mp3(audio, mp3, bitrate="96k", mono=True)       # keeps a 5-min song under ~4 MB of base64
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": "Listen to this song and report."},
                                         audio_part(mp3)]},
        ]
        return chat_json(messages, model, Subjective).model_dump(mode="json")

    return Subjective(**cached(cache_dir, sha, "subjective", f"{VERSION}:{model}", work))
