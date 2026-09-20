"""All data contracts. One instrument-neutral structure sits between analysis and output (BUILD-SPEC §5).

RULE: nothing instrument-specific in this file. No frets, strings, capo, tab.
RULE: every Chord and Note carries confidence.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field

LOW_CONFIDENCE = 0.5
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---- the neutral musical representation ---------------------------------

class Note(BaseModel):
    pitch: int                      # MIDI note number
    onset: float                    # seconds
    duration: float                 # seconds
    confidence: Confidence
    stem: Literal["vocals", "bass", "other"]


class Chord(BaseModel):
    symbol: str                     # display form, e.g. "Am7/G"
    harte: str                      # raw recogniser label, e.g. "A:min7/b7"
    root: str
    quality: str
    extensions: list[str]
    bass: str | None
    onset: float
    duration: float
    confidence: Confidence


class Section(BaseModel):
    label: str                      # intro | verse | pre-chorus | chorus | bridge | solo | outro | ...
    start: float
    end: float
    confidence: Confidence


class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: Confidence


class GlobalInfo(BaseModel):
    key: str                        # e.g. "A minor"
    key_confidence: Confidence
    tempo_bpm: float
    time_signature: str             # e.g. "4/4"
    loudness_lufs: float | None
    duration_s: float


class DensitySpan(BaseModel):
    start: float
    end: float
    description: str


class SectionGuess(BaseModel):
    label: str
    start: float
    end: float


class Subjective(BaseModel):
    """What the audio LLM hears. NEVER contains chords, key or tempo — that ear hallucinates them."""
    genre_tags: list[str]
    instrumentation: list[str]
    timbre: str
    vocal_character: str
    vocal_gender: Literal["male", "female", "mixed", "none", "unclear"]
    production: str
    emotional_arc: str
    arrangement_density: list[DensitySpan]
    sections: list[SectionGuess]


class Analysis(BaseModel):
    schema_version: int = 1
    audio_sha1: str
    engines: dict[str, str]         # which ears ran, and with what (component → model/version)
    global_info: GlobalInfo | None = None
    beats: list[float] = []
    downbeats: list[float] = []
    chords: list[Chord] = []
    notes: list[Note] = []
    sections: list[Section] = []
    lyrics: list[Word] = []
    subjective: Subjective | None = None


# ---- stage files ----------------------------------------------------------

class SourceInfo(BaseModel):                       # 00-source.json
    origin: str                                    # URL or absolute path as given
    kind: Literal["url", "audio", "video"]
    title: str | None = None
    uploader: str | None = None
    duration_s: float
    sample_rate: int
    sha1: str
    ingested_at: str


class Brief(BaseModel):                            # 02-brief.json
    text: str
    origin: str                                    # "inline" or the file path


class SpecSection(BaseModel):
    name: str                                      # "Verse 1", "Chorus", "Bridge", "Instrumental Break"
    lines: list[str]                               # lyric lines; empty for instrumental sections
    duration_s: float
    style_notes: str = ""                          # e.g. "drums drop out, fingerpicked"


class SongSpec(BaseModel):                         # 03-spec.json
    schema_version: int = 1
    title: str
    style_prompt: str                              # genre / mood / instrumentation, NO artist names
    negative_style: str = ""
    vocal_gender: Literal["male", "female", "any"] = "any"
    target_duration_s: float
    sections: list[SpecSection]
    fidelity: Literal["loose", "medium", "close"] | None = None   # set by `generate`, not `compose`
    fidelity_note: str = ""
    writer_model: str = ""

    def all_lines(self) -> list[str]:
        return [ln for s in self.sections for ln in s.lines]

    def sections_total_s(self) -> float:
        return sum(s.duration_s for s in self.sections)


class TakeRecord(BaseModel):
    index: int
    file: str                                      # "take-2.mp3"
    provider: str
    provider_ref: str                              # provider's own id for the track
    duration_s: float
    bytes: int


class GenerationRun(BaseModel):
    provider: str
    request_hash: str
    fidelity: str
    fidelity_note: str
    style_prompt: str                              # the final prompt actually sent
    payload: dict                                  # exact provider request body/bodies
    cost_estimate_usd: float
    cost_actual_usd: float
    created_at: str
    warnings: list[str] = []
    takes: list[TakeRecord]


class TakesManifest(BaseModel):                    # 04-takes/takes.json
    runs: list[GenerationRun] = []

    def all_takes(self) -> list[TakeRecord]:
        return [t for r in self.runs for t in r.takes]


class Chosen(BaseModel):                           # 05-chosen.json
    take: int
    file: str
    provider: str
    sha1: str
    chosen_at: str


class LyricLine(BaseModel):
    section: str
    text: str
    start: float | None                            # None = could not be aligned; renderers must say so
    end: float | None
    words: list[Word]
    confidence: Confidence


class Transcription(BaseModel):                    # 06-transcription.json
    take: int
    analysis: Analysis
    lines: list[LyricLine]
