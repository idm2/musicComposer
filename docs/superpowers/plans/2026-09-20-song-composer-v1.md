# Song Composer v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A staged Python CLI (`songcomposer`) that ingests a reference song + brief, analyses the reference with two ears, writes an original song, generates three sung takes behind a Suno/ElevenLabs provider interface, and derives guitar chords, tab, PDF and MusicXML by transcribing the chosen take.

**Architecture:** Every stage is a CLI command that reads and writes inspectable JSON/audio files under `work/<song>/`; deliverables land in `out/<song>/`. One analysis engine (`songcomposer.analysis.analyze`) runs twice — over the reference and over our own chosen take — and emits one instrument-neutral, confidence-carrying structure (`Analysis`) that all renderers consume. Paid generation sits behind a `Provider` protocol and a mandatory pre-flight + cost-confirmation gate.

**Tech Stack:** Python 3.11 (uv), Typer, Pydantic v2, httpx, PyTorch+CUDA, Demucs, beat_this, chord recogniser (per spike), basic-pitch, faster-whisper, librosa, music21, LilyPond, yt-dlp, ffmpeg, pytest.

**Spec:** `docs/BUILD-SPEC.md` (authoritative), `docs/DECISIONS.md` (rationale), `docs/spikes/2026-09-20-mir-stack.md` (library spike results — read before Tasks 13–20).

## Global Constraints

- **Python 3.11 only** — `requires-python = ">=3.11,<3.12"`. Python 3.13 is on PATH; never use it. All commands run through `uv run`.
- **Pre-flight before any paid call — MANDATORY.** `generate` must refuse unvalidated input, print estimated cost, and wait for an explicit typed `yes`. No code path may reach a provider's `generate()` without passing `preflight()` first.
- **No paid API calls in the test suite.** All HTTP is fixtured with `httpx.MockTransport`.
- **Confidence travels with the data.** Every `Chord` and `Note` carries `confidence` (0.0–1.0). Renderers mark anything below `LOW_CONFIDENCE = 0.5` visibly; never drop or hide it.
- **Nothing instrument-specific upstream of `songcomposer/render/`.** No fret, string, capo or tab concept may appear in `models.py` or `analysis/`.
- **v1 is guitar-only.** No piano/violin/bass renderers.
- **UTF-8 everywhere, never via PowerShell 5.1.** Every `open()`/`read_text()`/`write_text()` passes `encoding="utf-8"`. Never use `Get-Content`/`Set-Content` on project files.
- **Deliverables stay in `out/<song>/`** of this repo. Never write into another project's tree.
- **Do not modify** `C:\dev\Logic8 Apps\Video Generator - Fal Remotion`. Read-only source of patterns and `.env` values.
- **Windows 11.** Use `subprocess.run([...])` with list args (never `shell=True`); use `pathlib.Path`; ffmpeg/ffprobe are on PATH.
- **librosa is table stakes, not the engine** — it may compute key, loudness and features; it must never be the chord recogniser.
- Commit after every task with the message given in the task. End commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Decisions this plan locks in (BUILD-SPEC §12 open questions)

| # | Question | Decision | Why |
|---|---|---|---|
| 1 | Chord library | See Task 16 — chosen from the spike doc | Spike-verified on this machine |
| 2 | LilyPond vs MuseScore | **LilyPond** for `chart.pdf`, **music21** for MusicXML | LilyPond has native `TabStaff` (computes frets itself), `FretBoards` (auto chord diagrams), `ChordNames` and `Lyrics` — a guitar chart is its home turf. We emit `.ly` text directly; no lossy MusicXML→PDF hop. |
| 3 | Port `lib.mjs` or thin Node layer | **Port to Python** (`httpx`) | One runtime. The pieces we need (env, JSON, Kie/ElevenLabs/OpenRouter POST, polling) are ~150 lines. |
| 4 | Content-hashed analysis cache | **Yes** | sha1 of audio bytes + component name + component version → `work/<song>/.cache/`. Same pattern as `chunk_key()` in the audiobook script. |
| 5 | Suno vs ElevenLabs default | Decided by the **bake-off in Task 12**, recorded in `songcomposer.toml` | Needs ears, not code. |

### Deliberate deviations from the spec — flagged, not hidden

1. **Task order puts the generation path before the objective ear.** `analyze` runs whichever ears are installed and records them in `analysis.engines`. After Task 12 (Milestone A) you can produce real songs using the subjective + lyrics ears only; Tasks 13–20 then add the DSP ear and you re-run `analyze`. No chord is ever sourced from the LLM — until the objective ear exists, `analysis.chords` is simply empty. Charts (Tasks 21–26) are impossible before the objective ear exists, so the two-ears rule is never violated.
2. **Suno returns two tracks per request.** Three takes therefore costs two requests and yields four tracks. We keep all four (`take-1..4.mp3`) rather than throw away paid audio. ElevenLabs runs yield exactly three.
3. **`wordsFromAlignment()` from `voice.mjs` is not ported.** It converts ElevenLabs *TTS* character alignment; sung takes have no such alignment. Word timings come from Whisper over the vocal stem, aligned to our known lyric lines (Task 21).
4. **Essentia is dropped** — no Windows wheels. librosa covers key/loudness/features.
5. **Suno model is `V6`, not v5.5.** Kie.ai's docs (checked 2026-09-20) mark `V5_5` and everything below "Discontinued"; `V6` is the default. Kie also moved Suno to the unified `POST /api/v1/jobs/createTask` + `GET /api/v1/jobs/recordInfo` endpoints — the same polling shape as `pollUnifiedJob()` in `lib.mjs`. Model id lives in `songcomposer.toml` (`suno_model`).
6. **ElevenLabs model is `music_v2_5`, not `music_v2`.** Docs call it "our most advanced music model". The v2 family takes a `chunks[]` composition plan, not the v1 `sections[]` shape. Model id lives in `songcomposer.toml` (`elevenlabs_model`).

## File Structure

```
pyproject.toml                  deps, CLI entry point, pytest config
songcomposer.toml               user config: default provider, model slugs (created Task 1)
.env.example                    key names only
src/songcomposer/
  __init__.py
  cli.py                        Typer app — one command per stage, thin: parse args → call stage fn
  config.py                     load songcomposer.toml with defaults
  env.py                        load_env / require_env            (port of lib.mjs loadEnv/requireEnv)
  jsonio.py                     read_json / write_json / write_model (UTF-8, LF, trailing newline)
  paths.py                      SongPaths — every file location in work/<song> and out/<song>
  hashing.py                    file_sha1 / content_key
  models.py                     ALL data contracts (pydantic): Note, Chord, Section, Word, GlobalInfo,
                                Subjective, Analysis, SourceInfo, Brief, SongSpec, TakesManifest, Chosen
  chordsym.py                   Harte chord-label parsing → root/quality/bass/display symbol
  audio.py                      ffmpeg/ffprobe helpers: to_wav, to_mp3, probe_duration
  ingest.py                     stage: ingest
  brief.py                      stage: brief
  clients/
    openrouter.py               chat / chat_json / audio_part
    kie.py                      kie_post / kie_get / poll_suno
    elevenlabs.py               eleven_post
  analysis/
    __init__.py                 analyze(audio, cache_dir, ears) → Analysis   ← THE ENGINE, RUN TWICE
    cache.py                    cached(component, version, audio_hash, cache_dir, fn)
    subjective.py               Gemini ear
    stems.py                    Demucs
    beats.py                    beat_this → beats, downbeats, tempo, time signature
    chords.py                   ChordRecognizer adapter
    key.py                      key + loudness (librosa)
    notes.py                    basic-pitch per stem
    lyrics.py                   faster-whisper on vocal stem
    structure.py                DSP boundaries + label fusion
    fuse.py                     beat-snap, merge, assemble Analysis
  compose.py                    stage: compose (LLM → SongSpec)
  generate/
    __init__.py                 run_generate(): preflight → estimate → confirm → cache → provider → sanity
    preflight.py                validate_spec() — the mandatory gate
    prompt.py                   build_request(): fidelity → style prompt
    provider.py                 Provider protocol, GenerationRequest, TakeResult, ProviderLimits, get_provider
    suno.py
    elevenlabs.py
  pick.py                       stage: pick
  transcribe.py                 stage (inside chart): analyze chosen take + align spec lyric lines
  render/
    timing.py                   beat grid: seconds → (bar, sixteenth); duration splitting
    chordsheet.py               chords.txt (+ capo suggestion)
    guitar.py                   chord-shape dictionary, fret mapping
    tab.py                      tab.txt
    musicxml.py                 <song>.musicxml via music21
    lilypond.py                 chart.ly → chart.pdf
    lyricsjson.py               lyrics.json (video-project contract)
  chart.py                      stage: chart — orchestrates transcribe + all renderers
  pipeline.py                   stage: run
tests/
  conftest.py                   tmp song dirs, sine-wav fixture, gpu marker
  synth.py                      ground-truth audio synthesised from a known progression
  test_*.py                     one per module
templates/                      project-local recipes (markdown), per video-project convention
  first-song.md
```

`work/` and `out/` are already git-ignored.

---

### Task 1: Project scaffold, env, JSON I/O, paths

**Files:**
- Create: `pyproject.toml`, `songcomposer.toml`, `.env.example`, `src/songcomposer/__init__.py`, `src/songcomposer/env.py`, `src/songcomposer/jsonio.py`, `src/songcomposer/paths.py`, `src/songcomposer/hashing.py`, `src/songcomposer/config.py`, `src/songcomposer/cli.py`
- Test: `tests/conftest.py`, `tests/test_foundation.py`

**Interfaces:**
- Produces:
  - `env.load_env(path: Path = Path(".env")) -> None`, `env.require_env(name: str) -> str`
  - `jsonio.read_json(path: Path) -> Any`, `jsonio.write_json(path: Path, data: Any) -> None`, `jsonio.write_model(path: Path, model: BaseModel) -> None`
  - `paths.SongPaths(song: str, root: Path = Path.cwd())` with attributes listed below
  - `hashing.file_sha1(path: Path) -> str` (40 hex), `hashing.content_key(*parts: str) -> str` (16 hex)
  - `config.load_config(root: Path = Path.cwd()) -> Config` with fields `default_provider: str`, `writer_model: str`, `listener_model: str`, `whisper_model: str`
  - `cli.app` — Typer app

- [ ] **Step 1: Install toolchain (skip any already present — the spike may have installed uv)**

```bash
uv --version || scoop install uv
uv python install 3.11
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "songcomposer"
version = "0.1.0"
description = "Songwriter and proof-of-concept generator"
requires-python = ">=3.11,<3.12"
dependencies = [
    "typer>=0.12",
    "pydantic>=2.7",
    "httpx>=0.27",
]

[project.scripts]
songcomposer = "songcomposer.cli:app"

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/songcomposer"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
markers = ["gpu: needs the local GPU analysis stack and model downloads (deselect with -m 'not gpu')"]
```

- [ ] **Step 3: Write `songcomposer.toml` and `.env.example`**

`songcomposer.toml`:
```toml
# User-editable defaults. default_provider is set by the Task 12 bake-off.
default_provider = "suno"
# OpenRouter slugs. Both verified present in GET https://openrouter.ai/api/v1/models on 2026-09-20.
# listener_model MUST list "audio" in its input_modalities. Newer option: google/gemini-3.1-pro-preview.
writer_model = "google/gemini-2.5-pro"
listener_model = "google/gemini-2.5-pro"
whisper_model = "large-v3"
# Kie.ai marks V5_5 and below "Discontinued" (checked 2026-09-20). Current: V6 | V6_MINI | V6_WILD.
suno_model = "V6"
# music_v1 | music_v2 | music_v2_5. The v2 family requires the chunks[] plan format (Task 11).
elevenlabs_model = "music_v2_5"
```

`.env.example`:
```
ELEVENLABS_API_KEY=
OPENROUTER_API_KEY=
KIE_API_KEY=
# Only if OpenRouter will not pass audio through to Gemini (BUILD-SPEC §10):
GOOGLE_AI_API_KEY=
```

- [ ] **Step 4: Create `.env` by copying the three needed keys from the video project — with Python, never PowerShell**

```bash
uv run --no-project python - <<'EOF'
from pathlib import Path
src = Path(r"C:\dev\Logic8 Apps\Video Generator - Fal Remotion\.env").read_text(encoding="utf-8")
want = ("ELEVENLABS_API_KEY", "OPENROUTER_API_KEY", "KIE_API_KEY")
lines = [l for l in src.splitlines() if l.split("=", 1)[0].strip() in want]
assert len(lines) == 3, f"expected 3 keys, found {len(lines)}"
Path(".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote .env with", [l.split("=")[0] for l in lines])
EOF
git check-ignore .env   # must print ".env"
```

- [ ] **Step 5: Write the failing tests**

`tests/conftest.py`:
```python
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """An empty project root; cwd is moved there so SongPaths defaults resolve inside it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def sine_wav(tmp_path) -> Path:
    out = tmp_path / "sine.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ar", "22050", "-ac", "1", str(out)],
        check=True,
    )
    return out
```

`tests/test_foundation.py`:
```python
import os

import pytest

from songcomposer import env, hashing, jsonio
from songcomposer.config import load_config
from songcomposer.paths import SongPaths


def test_load_env_parses_quotes_and_does_not_override(root, monkeypatch):
    (root / ".env").write_text('A_KEY="quoted"\n# comment\nB_KEY=plain\nC_KEY=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "fromshell")
    env._loaded = False
    env.load_env()
    assert os.environ["A_KEY"] == "quoted"
    assert os.environ["B_KEY"] == "plain"
    assert os.environ["C_KEY"] == "fromshell"


def test_require_env_missing_raises_with_name(root, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    env._loaded = False
    with pytest.raises(RuntimeError, match="NOPE_KEY"):
        env.require_env("NOPE_KEY")


def test_json_roundtrip_preserves_smart_quotes_and_accents(root):
    p = root / "deep" / "x.json"
    data = {"line": "She said “café” — déjà vu’s"}
    jsonio.write_json(p, data)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")          # no BOM
    assert "“café”".encode("utf-8") in raw               # not \u-escaped
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert jsonio.read_json(p) == data


def test_song_paths_layout(root):
    p = SongPaths("my-song")
    assert p.work == root / "work" / "my-song"
    assert p.source_wav.name == "00-source.wav"
    assert p.source_json.name == "00-source.json"
    assert p.analysis.name == "01-analysis.json"
    assert p.brief.name == "02-brief.json"
    assert p.spec.name == "03-spec.json"
    assert p.takes_dir == p.work / "04-takes"
    assert p.takes_json == p.takes_dir / "takes.json"
    assert p.chosen.name == "05-chosen.json"
    assert p.transcription.name == "06-transcription.json"
    assert p.cache == p.work / ".cache"
    assert p.out == root / "out" / "my-song"
    assert p.out_mp3 == p.out / "my-song.mp3"
    assert p.out_musicxml == p.out / "my-song.musicxml"
    assert {p.chords_txt.name, p.tab_txt.name, p.chart_pdf.name, p.lyrics_json.name} == {
        "chords.txt", "tab.txt", "chart.pdf", "lyrics.json"}


@pytest.mark.parametrize("bad", ["My Song", "../x", "", "-lead", "a_b"])
def test_song_name_rejected(root, bad):
    with pytest.raises(ValueError):
        SongPaths(bad)


def test_hashing(sine_wav):
    h = hashing.file_sha1(sine_wav)
    assert len(h) == 40 and h == hashing.file_sha1(sine_wav)
    assert len(hashing.content_key("a", "b")) == 16
    assert hashing.content_key("a", "b") != hashing.content_key("ab", "")


def test_config_defaults_when_file_missing(root):
    c = load_config()
    assert c.default_provider == "suno"
    assert c.whisper_model == "large-v3"
```

- [ ] **Step 6: Run tests, verify they fail**

Run: `uv sync && uv run pytest tests/test_foundation.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'songcomposer'` (or missing submodules).

- [ ] **Step 7: Write the implementation**

`src/songcomposer/__init__.py`:
```python
"""Song Composer — songwriter and proof-of-concept generator."""
```

`src/songcomposer/env.py`:
```python
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
```

`src/songcomposer/jsonio.py`:
```python
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
```

`src/songcomposer/paths.py`:
```python
import re
from pathlib import Path

_SONG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SongPaths:
    """Every file location for one song. Mirrors BUILD-SPEC §3."""

    def __init__(self, song: str, root: Path | None = None):
        if not _SONG_RE.match(song or ""):
            raise ValueError(f"song name must be lowercase kebab-case (a-z, 0-9, -): {song!r}")
        root = Path(root) if root else Path.cwd()
        self.song = song
        self.work = root / "work" / song
        self.out = root / "out" / song
        self.cache = self.work / ".cache"
        self.source_wav = self.work / "00-source.wav"
        self.source_json = self.work / "00-source.json"
        self.analysis = self.work / "01-analysis.json"
        self.brief = self.work / "02-brief.json"
        self.spec = self.work / "03-spec.json"
        self.takes_dir = self.work / "04-takes"
        self.takes_json = self.takes_dir / "takes.json"
        self.chosen = self.work / "05-chosen.json"
        self.transcription = self.work / "06-transcription.json"
        self.out_mp3 = self.out / f"{song}.mp3"
        self.out_musicxml = self.out / f"{song}.musicxml"
        self.chords_txt = self.out / "chords.txt"
        self.tab_txt = self.out / "tab.txt"
        self.chart_ly = self.out / "chart.ly"
        self.chart_pdf = self.out / "chart.pdf"
        self.lyrics_json = self.out / "lyrics.json"

    def take(self, index: int) -> Path:
        return self.takes_dir / f"take-{index}.mp3"

    def require(self, path: Path, made_by: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `songcomposer {made_by} {self.song}` first.")
        return path
```

`src/songcomposer/hashing.py`:
```python
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
```

`src/songcomposer/config.py`:
```python
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
```

`src/songcomposer/cli.py`:
```python
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Songwriter and proof-of-concept generator. Each stage writes an inspectable file.")


@app.callback()
def _main() -> None:
    """Stages: ingest → analyze → brief → compose → generate → pick → chart (or `run` for all)."""
```

- [ ] **Step 8: Run tests, verify they pass**

Run: `uv run pytest tests/test_foundation.py -v`
Expected: all PASS. Then `uv run songcomposer --help` prints the help text.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock songcomposer.toml .env.example src tests
git commit -m "feat: project scaffold — env, json io, song paths, config, CLI shell"
```

---

### Task 2: Data contracts and chord-symbol parsing

**Files:**
- Create: `src/songcomposer/models.py`, `src/songcomposer/chordsym.py`
- Test: `tests/test_models.py`, `tests/test_chordsym.py`

**Interfaces:**
- Produces (all in `songcomposer.models`, all pydantic `BaseModel`):
  - `LOW_CONFIDENCE = 0.5`
  - `Note(pitch:int, onset:float, duration:float, confidence:float, stem:str)` — `pitch` is a MIDI number; `stem` ∈ `vocals|bass|other`
  - `Chord(symbol:str, harte:str, root:str, quality:str, extensions:list[str], bass:str|None, onset:float, duration:float, confidence:float)`
  - `Section(label:str, start:float, end:float, confidence:float)`
  - `Word(word:str, start:float, end:float, confidence:float)`
  - `GlobalInfo(key:str, key_confidence:float, tempo_bpm:float, time_signature:str, loudness_lufs:float|None, duration_s:float)`
  - `DensitySpan(start:float, end:float, description:str)`, `SectionGuess(label:str, start:float, end:float)`
  - `Subjective(genre_tags, instrumentation, timbre, vocal_character, vocal_gender, production, emotional_arc, arrangement_density:list[DensitySpan], sections:list[SectionGuess])`
  - `Analysis(schema_version, audio_sha1, engines:dict[str,str], global_info:GlobalInfo|None, beats:list[float], downbeats:list[float], chords:list[Chord], notes:list[Note], sections:list[Section], lyrics:list[Word], subjective:Subjective|None)`
  - `SourceInfo`, `Brief`, `SpecSection`, `SongSpec`, `TakeRecord`, `GenerationRun`, `TakesManifest`, `Chosen`, `LyricLine`, `Transcription` — fields exactly as in the code below
- Produces (in `songcomposer.chordsym`): `parse_harte(label: str) -> ParsedChord | None` (returns `None` for no-chord `N`/`X`), `ParsedChord(root, quality, extensions, bass, symbol)`, `NOTE_NAMES_SHARP`, `NOTE_NAMES_FLAT`, `pitch_class(name: str) -> int`, `transpose_name(name: str, semitones: int, prefer_flats: bool) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_chordsym.py`:
```python
import pytest

from songcomposer.chordsym import parse_harte, pitch_class, transpose_name


@pytest.mark.parametrize("label,symbol,root,quality,bass", [
    ("C:maj", "C", "C", "maj", None),
    ("A:min", "Am", "A", "min", None),
    ("A:min7", "Am7", "A", "min7", None),
    ("C:maj7", "Cmaj7", "C", "maj7", None),
    ("G:7", "G7", "G", "7", None),
    ("D:sus4", "Dsus4", "D", "sus4", None),
    ("D:sus2", "Dsus2", "D", "sus2", None),
    ("B:hdim7", "Bm7b5", "B", "hdim7", None),
    ("F#:dim", "F#dim", "F#", "dim", None),
    ("Bb:maj", "Bb", "Bb", "maj", None),
    ("C:maj/3", "C/E", "C", "maj", "E"),
    ("A:min/b7", "Am/G", "A", "min", "G"),
    ("Bb:maj/5", "Bb/F", "Bb", "maj", "F"),
    ("G", "G", "G", "maj", None),              # bare root = major triad
])
def test_parse_harte(label, symbol, root, quality, bass):
    c = parse_harte(label)
    assert (c.symbol, c.root, c.quality, c.bass) == (symbol, root, quality, bass)


def test_cmaj7_and_am_are_distinct():
    """The librosa failure mode named in CLAUDE.md must be representable as two different chords."""
    assert parse_harte("C:maj7").symbol != parse_harte("A:min").symbol


def test_parenthesised_extensions_kept():
    c = parse_harte("C:maj(9)")
    assert c.quality == "maj" and c.extensions == ["9"] and c.symbol == "Cadd9"


@pytest.mark.parametrize("label", ["N", "X", ""])
def test_no_chord_is_none(label):
    assert parse_harte(label) is None


def test_unknown_quality_is_preserved_not_guessed():
    c = parse_harte("C:weird")
    assert c.quality == "weird" and c.symbol == "C(weird)"


def test_pitch_class_and_transpose():
    assert pitch_class("C") == 0 and pitch_class("F#") == 6 and pitch_class("Bb") == 10
    assert transpose_name("A", 3, prefer_flats=False) == "C"
    assert transpose_name("A", 1, prefer_flats=True) == "Bb"
    assert transpose_name("A", 1, prefer_flats=False) == "A#"
```

`tests/test_models.py`:
```python
import pytest
from pydantic import ValidationError

from songcomposer.models import Analysis, Chord, Note, SongSpec, SpecSection


def test_confidence_is_mandatory_and_bounded():
    with pytest.raises(ValidationError):
        Note(pitch=60, onset=0, duration=1, stem="vocals")                 # no confidence
    with pytest.raises(ValidationError):
        Note(pitch=60, onset=0, duration=1, stem="vocals", confidence=1.5)
    with pytest.raises(ValidationError):
        Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None,
              onset=0, duration=2)                                          # no confidence


def test_models_are_instrument_neutral():
    """Global constraint: nothing guitar-specific upstream of render/."""
    banned = {"fret", "string", "capo", "tab", "fingering"}
    for model in (Note, Chord, Analysis):
        assert not banned & set(model.model_fields), model


def test_analysis_minimal_roundtrip():
    a = Analysis(audio_sha1="a" * 40, engines={"subjective": "google/gemini-2.5-pro"})
    again = Analysis.model_validate_json(a.model_dump_json())
    assert again == a and again.chords == [] and again.global_info is None


def test_songspec_lyrics_text_and_duration():
    spec = SongSpec(title="Glass Hour", style_prompt="sparse indie folk", negative_style="",
                    target_duration_s=120,
                    sections=[SpecSection(name="Verse 1", lines=["a b", "c d"], duration_s=30),
                              SpecSection(name="Break", lines=[], duration_s=10)])
    assert spec.all_lines() == ["a b", "c d"]
    assert spec.sections_total_s() == 40
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_models.py tests/test_chordsym.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.models'` / `songcomposer.chordsym`.

- [ ] **Step 3: Write `src/songcomposer/chordsym.py`**

```python
"""Harte chord-label parsing (the 'C:min7/b7' syntax used by MIR chord recognisers).

Instrument-neutral: knows pitch names and chord qualities, nothing about guitars.
"""
import re
from dataclasses import dataclass, field

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Harte quality → display suffix
_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "7": "7", "maj7": "maj7", "min7": "m7", "minmaj7": "m(maj7)",
    "dim7": "dim7", "hdim7": "m7b5",
    "6": "6", "maj6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9", "11": "11", "min11": "m11", "13": "13", "maj13": "maj13", "min13": "m13",
    "sus2": "sus2", "sus4": "sus4", "5": "5", "1": "5",
}
# Harte interval → semitones above root
_INTERVAL = {"1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6, "b5": 6, "5": 7,
             "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3,
             "11": 5, "#11": 6, "b13": 8, "13": 9}
_LABEL_RE = re.compile(r"^([A-G][#b]?)(?::([^/()]*)(?:\(([^)]*)\))?)?(?:/(.+))?$")


@dataclass
class ParsedChord:
    root: str
    quality: str
    extensions: list[str] = field(default_factory=list)
    bass: str | None = None
    symbol: str = ""


def pitch_class(name: str) -> int:
    pc = _NATURAL[name[0]]
    for acc in name[1:]:
        pc += 1 if acc == "#" else -1
    return pc % 12


def transpose_name(name: str, semitones: int, prefer_flats: bool) -> str:
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    return names[(pitch_class(name) + semitones) % 12]


def _prefers_flats(root: str) -> bool:
    return "b" in root[1:] or root == "F"


def parse_harte(label: str) -> ParsedChord | None:
    label = (label or "").strip()
    if label in ("", "N", "X"):
        return None
    m = _LABEL_RE.match(label)
    if not m:
        return None
    root, quality, ext, bass_iv = m.group(1), m.group(2), m.group(3), m.group(4)
    quality = quality or "maj"
    extensions = [e.strip() for e in ext.split(",")] if ext else []
    bass = None
    if bass_iv and bass_iv in _INTERVAL and _INTERVAL[bass_iv] != 0:
        bass = transpose_name(root, _INTERVAL[bass_iv], _prefers_flats(root))
    if quality in _SUFFIX:
        suffix = _SUFFIX[quality]
    else:
        suffix = f"({quality})"          # unknown quality: show it verbatim, never guess
    adds = "".join(f"add{e}" for e in extensions if not e.startswith("*"))
    symbol = f"{root}{suffix}{adds}" + (f"/{bass}" if bass else "")
    return ParsedChord(root=root, quality=quality, extensions=extensions, bass=bass, symbol=symbol)
```

- [ ] **Step 4: Write `src/songcomposer/models.py`**

```python
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
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_models.py tests/test_chordsym.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/models.py src/songcomposer/chordsym.py tests/test_models.py tests/test_chordsym.py
git commit -m "feat: data contracts — instrument-neutral analysis model, stage files, Harte chord parsing"
```

---

### Task 3: `ingest` stage

**Files:**
- Create: `src/songcomposer/audio.py`, `src/songcomposer/ingest.py`
- Modify: `src/songcomposer/cli.py` (add `ingest` command), `pyproject.toml` (add `yt-dlp`)
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `SongPaths`, `SourceInfo`, `write_model`, `file_sha1`
- Produces:
  - `audio.to_wav(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> None`
  - `audio.to_mp3(src: Path, dst: Path, bitrate: str = "128k", mono: bool = False) -> None`
  - `audio.probe_duration(path: Path) -> float`
  - `ingest.classify_source(src: str) -> Literal["url","audio","video"]`
  - `ingest.run_ingest(song: str, src: str, force: bool = False) -> SourceInfo`

- [ ] **Step 1: Add the dependency**

Run: `uv add yt-dlp`

- [ ] **Step 2: Write the failing tests**

`tests/test_ingest.py`:
```python
import subprocess

import pytest

from songcomposer import ingest
from songcomposer.audio import probe_duration
from songcomposer.jsonio import read_json
from songcomposer.paths import SongPaths


@pytest.mark.parametrize("src,kind", [
    ("https://www.youtube.com/watch?v=abc", "url"),
    ("http://example.com/x.mp3", "url"),
    ("C:/music/ref.mp3", "audio"), ("ref.WAV", "audio"), ("a.flac", "audio"), ("a.m4a", "audio"),
    ("clip.mp4", "video"), ("clip.MOV", "video"), ("clip.mkv", "video"), ("clip.webm", "video"),
])
def test_classify_source(src, kind):
    assert ingest.classify_source(src) == kind


def test_classify_unknown_extension_raises():
    with pytest.raises(ValueError, match="unsupported"):
        ingest.classify_source("notes.txt")


def test_ingest_local_audio_normalises_to_44k_stereo_wav(root, sine_wav):
    info = ingest.run_ingest("demo", str(sine_wav))
    p = SongPaths("demo")
    assert p.source_wav.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name", "-of", "csv=p=0", str(p.source_wav)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "pcm_s16le,44100,2"
    assert info.kind == "audio" and info.sample_rate == 44100
    assert abs(info.duration_s - 2.0) < 0.1
    assert abs(probe_duration(p.source_wav) - 2.0) < 0.1
    on_disk = read_json(p.source_json)
    assert on_disk["sha1"] == info.sha1 and len(info.sha1) == 40
    assert on_disk["origin"] == str(sine_wav)


def test_ingest_is_noop_when_source_exists_unless_forced(root, sine_wav):
    first = ingest.run_ingest("demo", str(sine_wav))
    again = ingest.run_ingest("demo", "does-not-exist.wav")     # would fail if it actually ran
    assert again.sha1 == first.sha1
    with pytest.raises(FileNotFoundError):
        ingest.run_ingest("demo", "does-not-exist.wav", force=True)


def test_ingest_url_uses_ytdlp_then_normalises(root, sine_wav, monkeypatch):
    calls = []

    def fake_download(url, dest_dir):
        calls.append(url)
        target = dest_dir / "download.wav"
        target.write_bytes(sine_wav.read_bytes())
        return target, {"title": "Some Title", "uploader": "Some Channel"}

    monkeypatch.setattr(ingest, "_download", fake_download)
    info = ingest.run_ingest("demo", "https://youtu.be/xyz")
    assert calls == ["https://youtu.be/xyz"]
    assert (info.kind, info.title, info.uploader) == ("url", "Some Title", "Some Channel")
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: `ImportError: cannot import name 'ingest'`.

- [ ] **Step 4: Write `src/songcomposer/audio.py`**

```python
"""ffmpeg / ffprobe helpers. ffmpeg is installed via scoop and on PATH."""
import subprocess
from pathlib import Path


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, encoding="utf-8").stdout
    except FileNotFoundError as e:
        raise RuntimeError(f"{args[0]} not found on PATH — install with `scoop install ffmpeg`") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{args[0]} failed ({e.returncode}): {e.stderr[-400:]}") from e


def to_wav(src: Path, dst: Path, sample_rate: int = 44100, channels: int = 2) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
          "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)])


def to_mp3(src: Path, dst: Path, bitrate: str = "128k", mono: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn"]
    if mono:
        args += ["-ac", "1"]
    _run(args + ["-b:a", bitrate, str(dst)])


def probe_duration(path: Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    return float(out.strip())
```

- [ ] **Step 5: Write `src/songcomposer/ingest.py`**

```python
"""Stage: ingest. YouTube / video / audio file → work/<song>/00-source.wav + 00-source.json.

Format-normalises only (44.1 kHz stereo PCM — what Demucs wants). Deliberately does NOT
loudness-normalise: loudness is something the analyser measures.
"""
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .audio import probe_duration, to_wav
from .hashing import file_sha1
from .jsonio import read_json, write_model
from .models import SourceInfo
from .paths import SongPaths

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def classify_source(src: str) -> Literal["url", "audio", "video"]:
    if src.lower().startswith(("http://", "https://")):
        return "url"
    ext = Path(src).suffix.lower()
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"unsupported source type {ext!r}: {src}")


def _download(url: str, dest_dir: Path) -> tuple[Path, dict]:
    """yt-dlp best audio → dest_dir. Returns (file, metadata)."""
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bestaudio/best",
         "-o", str(dest_dir / "download.%(ext)s"), "-j", "--no-simulate", url],
        capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {proc.stderr[-600:]}")
    files = [f for f in dest_dir.glob("download.*") if f.suffix != ".part"]
    if not files:
        raise RuntimeError("yt-dlp reported success but wrote no file")
    meta = json.loads(proc.stdout.strip().splitlines()[-1])
    return files[0], {"title": meta.get("title"), "uploader": meta.get("uploader") or meta.get("channel")}


def run_ingest(song: str, src: str, force: bool = False) -> SourceInfo:
    paths = SongPaths(song)
    if paths.source_wav.exists() and paths.source_json.exists() and not force:
        print(f"- source exists at {paths.source_wav} (use --force to re-ingest)")
        return SourceInfo(**read_json(paths.source_json))

    kind = classify_source(src)
    meta: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "url":
            print(f"> downloading {src}")
            local, meta = _download(src, Path(tmp))
        else:
            local = Path(src)
            if not local.exists():
                raise FileNotFoundError(local)
        print(f"> normalising → {paths.source_wav}")
        to_wav(local, paths.source_wav)

    info = SourceInfo(
        origin=src, kind=kind, title=meta.get("title"), uploader=meta.get("uploader"),
        duration_s=round(probe_duration(paths.source_wav), 3), sample_rate=44100,
        sha1=file_sha1(paths.source_wav), ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    write_model(paths.source_json, info)
    print(f"  {info.duration_s:.1f}s  sha1={info.sha1[:12]}  → {paths.source_json}")
    return info
```

- [ ] **Step 6: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def ingest(song: str,
           from_: str = typer.Option(..., "--from", help="URL, audio file or video file"),
           force: bool = typer.Option(False, "--force", help="re-ingest even if 00-source.wav exists")) -> None:
    """Reference → work/<song>/00-source.wav + 00-source.json."""
    from .ingest import run_ingest
    run_ingest(song, from_, force=force)
```

- [ ] **Step 7: Run tests, verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/songcomposer/audio.py src/songcomposer/ingest.py src/songcomposer/cli.py tests/test_ingest.py
git commit -m "feat: ingest stage — url/audio/video to normalised wav with provenance"
```

---

### Task 4: OpenRouter client (chat, structured JSON, audio input)

**Files:**
- Create: `src/songcomposer/clients/__init__.py` (empty), `src/songcomposer/clients/openrouter.py`
- Test: `tests/test_openrouter.py`

**Interfaces:**
- Consumes: `env.require_env`
- Produces:
  - `openrouter.audio_part(mp3_path: Path) -> dict` — an `input_audio` content part (base64; OpenRouter does not accept audio URLs)
  - `openrouter.chat(messages: list[dict], model: str, *, schema: dict | None = None, client: httpx.Client | None = None) -> str`
  - `openrouter.chat_json(messages, model, out_type: type[T], *, retries: int = 2, client=None) -> T` where `T` is a pydantic model
  - `openrouter.inline_refs(schema: dict) -> dict` — pydantic JSON schema with `$defs`/`$ref` inlined

API facts (verified 2026-09-20): `POST https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer <key>`. Audio content part: `{"type":"input_audio","input_audio":{"data":"<base64>","format":"mp3"}}`. Structured output: `{"response_format":{"type":"json_schema","json_schema":{"name":..., "strict":true,"schema":{...}}}}` plus `"provider":{"require_parameters":true}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_openrouter.py`:
```python
import base64
import json

import httpx
import pytest
from pydantic import BaseModel

from songcomposer.clients import openrouter


class Inner(BaseModel):
    n: int


class Outer(BaseModel):
    name: str
    items: list[Inner]


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")


def _reply(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_audio_part_is_base64_mp3(tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"\x00\x01binary")
    part = openrouter.audio_part(f)
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "mp3"
    assert base64.b64decode(part["input_audio"]["data"]) == b"\x00\x01binary"


def test_inline_refs_removes_defs():
    schema = openrouter.inline_refs(Outer.model_json_schema())
    assert "$defs" not in schema and "$ref" not in json.dumps(schema)
    assert schema["properties"]["items"]["items"]["properties"]["n"]["type"] == "integer"


def test_chat_sends_auth_model_and_schema():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return _reply("hi")

    out = openrouter.chat([{"role": "user", "content": "x"}], "m/x", schema={"title": "T", "type": "object"},
                          client=_client(handler))
    assert out == "hi" and seen["auth"] == "Bearer sk-test" and seen["body"]["model"] == "m/x"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["provider"] == {"require_parameters": True}


def test_chat_raises_on_http_error_with_body():
    with pytest.raises(RuntimeError, match="402"):
        openrouter.chat([], "m", client=_client(lambda r: httpx.Response(402, json={"error": "no credit"})))


def test_chat_json_strips_fences_and_validates():
    body = '```json\n{"name": "a", "items": [{"n": 1}]}\n```'
    got = openrouter.chat_json([], "m", Outer, client=_client(lambda r: _reply(body)))
    assert got == Outer(name="a", items=[Inner(n=1)])


def test_chat_json_retries_with_validation_error_then_succeeds():
    replies = iter(['{"name": "a"}', '{"name": "a", "items": []}'])
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return _reply(next(replies))

    got = openrouter.chat_json([{"role": "user", "content": "go"}], "m", Outer, client=_client(handler))
    assert got.items == [] and len(bodies) == 2
    assert "failed validation" in bodies[1]["messages"][-1]["content"]


def test_chat_json_gives_up_after_retries():
    with pytest.raises(RuntimeError, match="valid Outer"):
        openrouter.chat_json([], "m", Outer, retries=1, client=_client(lambda r: _reply("{}")))
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_openrouter.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.clients'`.

- [ ] **Step 3: Write `src/songcomposer/clients/openrouter.py`** (and an empty `clients/__init__.py`)

```python
"""OpenRouter chat client. Port of the OpenRouter section of the video project's lib.mjs."""
import base64
import copy
import re
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..env import require_env

BASE = "https://openrouter.ai/api/v1"
T = TypeVar("T", bound=BaseModel)


def audio_part(mp3_path: Path) -> dict:
    data = base64.b64encode(Path(mp3_path).read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}}


def inline_refs(schema: dict) -> dict:
    """Gemini's schema support is patchy with $ref — inline every definition."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def chat(messages: list[dict], model: str, *, schema: dict | None = None,
         client: httpx.Client | None = None) -> str:
    body: dict = {"model": model, "messages": messages}
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": schema.get("title", "response"), "strict": True, "schema": schema}}
        body["provider"] = {"require_parameters": True}
    own = client is None
    client = client or httpx.Client(timeout=600.0)
    try:
        res = client.post(f"{BASE}/chat/completions", json=body, headers={
            "Authorization": f"Bearer {require_env('OPENROUTER_API_KEY')}"})
    finally:
        if own:
            client.close()
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter {model} failed ({res.status_code}): {res.text[:500]}")
    data = res.json()
    if "error" in data:
        raise RuntimeError(f"OpenRouter {model} error: {str(data['error'])[:500]}")
    return data["choices"][0]["message"]["content"]


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def chat_json(messages: list[dict], model: str, out_type: type[T], *, retries: int = 2,
              client: httpx.Client | None = None) -> T:
    schema = inline_refs(out_type.model_json_schema())
    msgs = list(messages)
    last: Exception | None = None
    for _ in range(retries + 1):
        text = chat(msgs, model, schema=schema, client=client)
        try:
            return out_type.model_validate_json(_strip_fences(text))
        except ValidationError as e:
            last = e
            msgs = msgs + [{"role": "assistant", "content": text},
                           {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."}]
    raise RuntimeError(f"model never returned a valid {out_type.__name__}: {last}")
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `uv run pytest tests/test_openrouter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/clients tests/test_openrouter.py
git commit -m "feat: OpenRouter client — chat, schema-validated JSON with retry, base64 audio input"
```

---

### Task 5: Analysis engine skeleton, content-hash cache, subjective ear, `analyze` command

**Files:**
- Create: `src/songcomposer/analysis/__init__.py`, `src/songcomposer/analysis/cache.py`, `src/songcomposer/analysis/subjective.py`
- Modify: `src/songcomposer/cli.py`
- Test: `tests/test_analysis_core.py`

**Interfaces:**
- Consumes: `openrouter.chat_json`, `openrouter.audio_part`, `audio.to_mp3`, `hashing.file_sha1`, `hashing.content_key`, `models.Analysis`, `models.Subjective`, `config.Config`
- Produces:
  - `analysis.cache.cached(cache_dir: Path, audio_sha1: str, component: str, version: str, fn: Callable[[], Any]) -> Any` — `fn` must return JSON-serialisable data
  - `analysis.subjective.VERSION: str`, `analysis.subjective.listen(audio: Path, cache_dir: Path, model: str) -> Subjective`
  - `analysis.analyze(audio: Path, cache_dir: Path, config: Config, ears: set[str] | None = None, lyrics_hint: str = "") -> Analysis` — **the engine that runs twice.** In this task it knows one ear (`subjective`); Task 20 replaces the body with the full fusion. `lyrics_hint` (our own known lyrics, used to bias Whisper when transcribing our own take) is accepted now and first used in Task 20. The signature never changes.
  - `analysis.OBJECTIVE_EARS: tuple[str, ...]` = `("stems","beats","chords","key","notes","lyrics","structure")`
  - CLI: `songcomposer analyze <song> [--ears a,b] `

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis_core.py`:
```python
import pytest

from songcomposer import analysis
from songcomposer.analysis import cache, subjective
from songcomposer.config import Config
from songcomposer.models import Subjective

HEARD = Subjective(
    genre_tags=["indie folk"], instrumentation=["fingerpicked acoustic guitar", "upright bass"],
    timbre="warm, woody, close-miked", vocal_character="breathy low tenor, restrained",
    vocal_gender="male", production="dry, intimate, tape-ish", emotional_arc="resigned → hopeful",
    arrangement_density=[], sections=[])


def test_cached_runs_once_per_key(tmp_path):
    calls = []

    def work():
        calls.append(1)
        return {"v": 1}

    assert cache.cached(tmp_path, "a" * 40, "beats", "1", work) == {"v": 1}
    assert cache.cached(tmp_path, "a" * 40, "beats", "1", work) == {"v": 1}
    assert len(calls) == 1
    cache.cached(tmp_path, "a" * 40, "beats", "2", work)      # version bump invalidates
    cache.cached(tmp_path, "b" * 40, "beats", "1", work)      # different audio invalidates
    assert len(calls) == 3


def test_subjective_model_cannot_carry_chords_key_or_tempo():
    """Two-ears rule, enforced structurally: the LLM ear has nowhere to put a chord."""
    fields = set(Subjective.model_fields)
    assert not {"chords", "key", "tempo", "tempo_bpm", "progression"} & fields


def test_listen_sends_audio_and_caches(tmp_path, sine_wav, monkeypatch):
    sent = []

    def fake_chat_json(messages, model, out_type, **kw):
        sent.append((messages, model))
        return HEARD

    monkeypatch.setattr(subjective, "chat_json", fake_chat_json)
    got = subjective.listen(sine_wav, tmp_path, "google/gemini-2.5-pro")
    assert got == HEARD
    parts = sent[0][0][-1]["content"]
    assert any(p["type"] == "input_audio" for p in parts)
    assert "chord" in sent[0][0][0]["content"].lower()        # the prompt forbids chord guessing
    subjective.listen(sine_wav, tmp_path, "google/gemini-2.5-pro")
    assert len(sent) == 1                                      # second call served from cache


def test_analyze_subjective_only_records_engines_and_leaves_chords_empty(tmp_path, sine_wav, monkeypatch):
    monkeypatch.setattr(subjective, "chat_json", lambda *a, **k: HEARD)
    a = analysis.analyze(sine_wav, tmp_path, Config(), ears={"subjective"})
    assert a.subjective == HEARD and a.chords == [] and a.global_info is None
    assert a.engines == {"subjective": "google/gemini-2.5-pro"}
    assert len(a.audio_sha1) == 40


def test_analyze_rejects_unknown_ear(tmp_path, sine_wav):
    with pytest.raises(ValueError, match="unknown ear"):
        analysis.analyze(sine_wav, tmp_path, Config(), ears={"vibes"})
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_analysis_core.py -v`
Expected: `ImportError` — `songcomposer.analysis` does not exist.

- [ ] **Step 3: Write `src/songcomposer/analysis/cache.py`**

```python
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
```

- [ ] **Step 4: Write `src/songcomposer/analysis/subjective.py`**

```python
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
```

- [ ] **Step 5: Write `src/songcomposer/analysis/__init__.py`**

```python
"""THE analysis engine. Runs twice — over the reference on the way in, and over our own
generated take on the way out. Same code path both times (BUILD-SPEC §4)."""
from pathlib import Path

from ..config import Config
from ..hashing import file_sha1
from ..models import Analysis

OBJECTIVE_EARS = ("stems", "beats", "chords", "key", "notes", "lyrics", "structure")
ALL_EARS = ("subjective",) + OBJECTIVE_EARS


def analyze(audio: Path, cache_dir: Path, config: Config, ears: set[str] | None = None,
            lyrics_hint: str = "") -> Analysis:
    ears = set(ALL_EARS) if ears is None else set(ears)
    unknown = ears - set(ALL_EARS)
    if unknown:
        raise ValueError(f"unknown ear(s): {sorted(unknown)} — valid: {', '.join(ALL_EARS)}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result = Analysis(audio_sha1=file_sha1(audio), engines={})

    if "subjective" in ears:
        from .subjective import listen
        print(f"> subjective ear ({config.listener_model})")
        result.subjective = listen(audio, cache_dir, config.listener_model)
        result.engines["subjective"] = config.listener_model

    missing = [e for e in OBJECTIVE_EARS if e not in result.engines]
    if missing:
        print(f"! PARTIAL ANALYSIS — objective ear not run ({', '.join(missing)}). "
              "No chords, key or tempo are available; nothing downstream may invent them.")
    return result
```

- [ ] **Step 6: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def analyze(song: str,
            ears: str = typer.Option("", "--ears", help="comma list; default = every ear installed")) -> None:
    """Two-ear analysis of the reference → work/<song>/01-analysis.json."""
    from .analysis import analyze as run
    from .config import load_config
    from .jsonio import write_model
    from .paths import SongPaths
    p = SongPaths(song)
    wav = p.require(p.source_wav, "ingest")
    chosen = {e.strip() for e in ears.split(",") if e.strip()} or None
    write_model(p.analysis, run(wav, p.cache, load_config(), chosen))
    print(f"  → {p.analysis}")
```

- [ ] **Step 7: Run tests, verify they pass**

Run: `uv run pytest tests/test_analysis_core.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/songcomposer/analysis src/songcomposer/cli.py tests/test_analysis_core.py
git commit -m "feat: analysis engine skeleton — content-hash cache, subjective ear, analyze command"
```

---

### Task 6: `brief` stage

**Files:**
- Create: `src/songcomposer/brief.py`
- Modify: `src/songcomposer/cli.py`
- Test: `tests/test_brief.py`

**Interfaces:**
- Produces: `brief.run_brief(song: str, from_file: str | None = None, text: str | None = None) -> Brief`; CLI `songcomposer brief <song> --from file.md | --text "..."`

- [ ] **Step 1: Write the failing tests**

`tests/test_brief.py`:
```python
import pytest

from songcomposer.brief import run_brief
from songcomposer.jsonio import read_json
from songcomposer.paths import SongPaths


def test_inline_text(root):
    b = run_brief("demo", text="A song about leaving a coastal town at dawn.")
    assert b.origin == "inline"
    assert read_json(SongPaths("demo").brief)["text"].startswith("A song about")


def test_markdown_file_with_bom_and_smart_quotes(root):
    f = root / "brief.md"
    f.write_bytes("﻿# Brief\n\nShe said “don’t go” — café lights.\n".encode("utf-8"))
    b = run_brief("demo", from_file=str(f))
    assert b.text.startswith("# Brief") and "“don’t go” — café" in b.text
    assert b.origin == str(f)


@pytest.mark.parametrize("kw", [{}, {"text": "x", "from_file": "y.md"}])
def test_exactly_one_source_required(root, kw):
    with pytest.raises(ValueError, match="exactly one"):
        run_brief("demo", **kw)


def test_too_short_rejected(root):
    with pytest.raises(ValueError, match="too short"):
        run_brief("demo", text="sad")
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_brief.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.brief'`.

- [ ] **Step 3: Write `src/songcomposer/brief.py`**

```python
"""Stage: brief. Inline text, a markdown file, or a one-liner → work/<song>/02-brief.json."""
from pathlib import Path

from .jsonio import write_model
from .models import Brief
from .paths import SongPaths


def run_brief(song: str, from_file: str | None = None, text: str | None = None) -> Brief:
    if (from_file is None) == (text is None):
        raise ValueError("give exactly one of --from <file> or --text \"...\"")
    if from_file is not None:
        body = Path(from_file).read_text(encoding="utf-8-sig")     # utf-8-sig tolerates a BOM
        origin = str(from_file)
    else:
        body, origin = text, "inline"
    body = body.strip()
    if len(body) < 10:
        raise ValueError(f"brief is too short ({len(body)} chars) — say what the song is about")
    brief = Brief(text=body, origin=origin)
    paths = SongPaths(song)
    write_model(paths.brief, brief)
    print(f"  brief ({len(body)} chars) → {paths.brief}")
    return brief
```

- [ ] **Step 4: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def brief(song: str,
          from_: str = typer.Option(None, "--from", help="markdown/text file"),
          text: str = typer.Option(None, "--text", help="inline brief or one-liner")) -> None:
    """Your brief → work/<song>/02-brief.json."""
    from .brief import run_brief
    run_brief(song, from_file=from_, text=text)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_brief.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/brief.py src/songcomposer/cli.py tests/test_brief.py
git commit -m "feat: brief stage"
```

---

### Task 7: Provider contract, prompt builder (fidelity), and the pre-flight gate

This is the MANDATORY gate from CLAUDE.md, ported from `validate_chunk()` / `preflight()` in `scripts/audiobook/generate_audiobook.py`. It validates the **final request** (after fidelity has been folded in), not just the spec.

**Files:**
- Create: `src/songcomposer/generate/__init__.py` (empty for now), `src/songcomposer/generate/provider.py`, `src/songcomposer/generate/prompt.py`, `src/songcomposer/generate/preflight.py`
- Test: `tests/test_preflight.py`, `tests/test_prompt.py`

**Interfaces:**
- Consumes: `SongSpec`, `SpecSection`, `Analysis`, `SourceInfo`
- Produces (`generate.provider`):
  - `ProviderLimits(max_title_chars:int, max_style_chars:int, max_lyrics_chars:int, max_line_chars:int, max_lines_per_section:int, max_sections:int, min_section_s:float, max_section_s:float, min_total_s:float, max_total_s:float)`
  - `GENERIC_LIMITS: ProviderLimits` — the intersection of both providers, used by `compose`
  - `GenerationRequest(title, style_prompt, negative_style, vocal_gender, sections: list[SpecSection], target_duration_s, n_takes:int=3)` with `.lyrics_text() -> str`
  - `CostEstimate(usd: float, basis: str)`, `TakeResult(provider_ref: str, path: Path, duration_s: float)`
  - `class Provider(Protocol)`: `name: str`; `limits: ProviderLimits`; `payload(req) -> dict`; `estimate(req) -> CostEstimate`; `generate(req, dest_dir: Path, start_index: int, on_take: Callable[[TakeResult], None]) -> float` (returns actual cost USD; calls `on_take` as soon as each file is safely on disk)
- Produces (`generate.prompt`): `FIDELITY_LEVELS = ("loose","medium","close")`, `build_request(spec: SongSpec, analysis: Analysis | None, fidelity: str, note: str = "", n_takes: int = 3) -> GenerationRequest`
- Produces (`generate.preflight`): `validate_request(req, limits, reference_lyrics: list[str], banned: list[str]) -> list[str]`, `banned_terms(source: SourceInfo | None) -> list[str]`, `preflight(req, limits, reference_lyrics, banned) -> None` (raises `SystemExit(1)` after printing the report)

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt.py`:
```python
import pytest

from songcomposer.generate.prompt import build_request
from songcomposer.models import Analysis, GlobalInfo, SongSpec, SpecSection, Subjective

SPEC = SongSpec(title="Glass Hour", style_prompt="sparse indie folk, melancholy", negative_style="edm, autotune",
                vocal_gender="male", target_duration_s=150,
                sections=[SpecSection(name="Verse 1", lines=["one line here", "two line here"], duration_s=75),
                          SpecSection(name="Chorus", lines=["sing it back"], duration_s=75)])
ANALYSIS = Analysis(
    audio_sha1="a" * 40, engines={},
    global_info=GlobalInfo(key="A minor", key_confidence=0.8, tempo_bpm=91.6, time_signature="3/4",
                           loudness_lufs=-14, duration_s=200),
    subjective=Subjective(genre_tags=["folk"], instrumentation=["fingerpicked acoustic guitar", "upright bass"],
                          timbre="warm and woody", vocal_character="breathy low tenor", vocal_gender="male",
                          production="dry, intimate", emotional_arc="", arrangement_density=[], sections=[]))


def test_loose_uses_spec_style_only():
    r = build_request(SPEC, ANALYSIS, "loose")
    assert r.style_prompt == "sparse indie folk, melancholy"


def test_medium_adds_tempo_instruments_and_voice_but_not_key():
    s = build_request(SPEC, ANALYSIS, "medium").style_prompt
    assert "92 BPM" in s and "fingerpicked acoustic guitar" in s and "breathy low tenor" in s
    assert "A minor" not in s


def test_close_adds_key_meter_and_production():
    s = build_request(SPEC, ANALYSIS, "close").style_prompt
    assert "A minor" in s and "3/4" in s and "dry, intimate" in s and "warm and woody" in s


def test_note_is_appended_and_partial_analysis_is_tolerated():
    r = build_request(SPEC, Analysis(audio_sha1="a" * 40, engines={}), "close", note="slower, more space")
    assert r.style_prompt == "sparse indie folk, melancholy, slower, more space"
    assert build_request(SPEC, None, "medium").style_prompt == "sparse indie folk, melancholy"


def test_lyrics_text_has_section_headers():
    assert build_request(SPEC, None, "loose").lyrics_text() == (
        "[Verse 1]\none line here\ntwo line here\n\n[Chorus]\nsing it back")


def test_bad_fidelity_rejected():
    with pytest.raises(ValueError, match="fidelity"):
        build_request(SPEC, None, "exact")
```

`tests/test_preflight.py`:
```python
import pytest

from songcomposer.generate.preflight import banned_terms, preflight, validate_request
from songcomposer.generate.provider import GENERIC_LIMITS, GenerationRequest
from songcomposer.models import SourceInfo, SpecSection


def req(**over):
    base = dict(
        title="Glass Hour", style_prompt="sparse indie folk, melancholy", negative_style="",
        vocal_gender="any", target_duration_s=120, n_takes=3,
        sections=[
            SpecSection(name="Verse 1", duration_s=60, lines=[
                "The kettle clicks off in the dark", "Your coat still hangs behind the door",
                "I count the streetlights to the park", "And lose my place at twenty-four"]),
            SpecSection(name="Chorus", duration_s=60, lines=[
                "Glass hour, hold me still", "Glass hour, against my will",
                "Turn the morning down", "Until you come around"]),
        ])
    base.update(over)
    return GenerationRequest(**base)


def problems(r, ref=(), banned=()):
    return validate_request(r, GENERIC_LIMITS, list(ref), list(banned))


def test_clean_request_passes():
    assert problems(req()) == []


@pytest.mark.parametrize("line,rule", [
    ("Hold me <break/> still", "markup"),
    ("Hold me [softly] still", "markup"),
    ("Broken � char", "replacement-char"),
    ("soft\xadhyphen line", "soft-hyphen"),
    ("TODO write this line", "placeholder"),
    ("Lyrics go here", "placeholder"),
    ("", "empty-line"),
    ("x" * 201, "line-too-long"),
])
def test_bad_lines_are_caught(line, rule):
    r = req()
    r.sections[0].lines[1] = line
    assert any(p.startswith(f"Verse 1 line 2: {rule}") for p in problems(r)), problems(r)


def test_structure_rules():
    assert any("title" in p for p in problems(req(title="")))
    assert any("title" in p for p in problems(req(title="x" * 81)))
    assert any("style_prompt" in p for p in problems(req(style_prompt=" ")))
    assert any("at least 2 sections" in p for p in problems(req(sections=req().sections[:1])))
    assert any("no lyrics" in p.lower() for p in problems(req(sections=[
        SpecSection(name="A", lines=[], duration_s=60), SpecSection(name="B", lines=[], duration_s=60)])))


def test_duration_rules():
    r = req()
    r.sections[0].duration_s = 2
    assert any("Verse 1: duration" in p for p in problems(r))
    assert any("sum to" in p for p in problems(req(target_duration_s=300)))


def test_crammed_lyrics_caught_by_words_per_second():
    r = req()
    r.sections[0].lines = ["word " * 12] * 25          # 300 words in 60 s = 5 wps
    assert any("words/sec" in p for p in problems(r))


def test_style_must_not_name_the_reference_or_say_in_the_style_of():
    assert any("banned term" in p for p in problems(req(style_prompt="folk like Bon Iver"), banned=["Bon Iver"]))
    assert any("style of" in p for p in problems(req(style_prompt="folk in the style of someone")))


def test_lines_copied_from_the_reference_are_caught():
    ref = ["and", "i", "count", "the", "streetlights", "to", "the", "park", "tonight"]
    out = problems(req(), ref=ref)
    assert any("copies the reference" in p and "streetlights" in p for p in out)


def test_banned_terms_from_source_title():
    src = SourceInfo(origin="u", kind="url", title="Bon Iver - Holocene (Official Video)", uploader="Bon Iver",
                     duration_s=1, sample_rate=44100, sha1="a" * 40, ingested_at="now")
    assert banned_terms(src) == ["Bon Iver", "Holocene"]
    assert banned_terms(None) == []


def test_preflight_exits_and_says_nothing_was_generated(capsys):
    with pytest.raises(SystemExit) as e:
        preflight(req(title=""), GENERIC_LIMITS, [], [])
    assert e.value.code == 1
    assert "NOTHING was generated" in capsys.readouterr().out


def test_preflight_passes_quietly(capsys):
    preflight(req(), GENERIC_LIMITS, [], [])
    assert "pre-flight passed" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_preflight.py tests/test_prompt.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.generate'`.

- [ ] **Step 3: Write `src/songcomposer/generate/provider.py`** (and an empty `generate/__init__.py`)

```python
"""The provider contract. The audio generator is a swappable module (CLAUDE.md):
whichever service produces the take, the analyser transcribes it identically."""
from pathlib import Path
from typing import Callable, Literal, Protocol

from pydantic import BaseModel

from ..models import SpecSection


class ProviderLimits(BaseModel):
    max_title_chars: int
    max_style_chars: int
    max_lyrics_chars: int
    max_line_chars: int
    max_lines_per_section: int
    max_sections: int
    min_section_s: float
    max_section_s: float
    min_total_s: float
    max_total_s: float


# Intersection of Suno V6 (title 80, style 1000, lyrics 5000, 10–360 s) and
# ElevenLabs music_v2_5 (30 chunks, 30 lines × 200 chars, chunk 3–120 s, total ≤ 600 s).
GENERIC_LIMITS = ProviderLimits(
    max_title_chars=80, max_style_chars=1000, max_lyrics_chars=5000, max_line_chars=200,
    max_lines_per_section=30, max_sections=30, min_section_s=3, max_section_s=120,
    min_total_s=30, max_total_s=360)


class GenerationRequest(BaseModel):
    title: str
    style_prompt: str
    negative_style: str = ""
    vocal_gender: Literal["male", "female", "any"] = "any"
    sections: list[SpecSection]
    target_duration_s: float
    n_takes: int = 3

    def lyrics_text(self) -> str:
        return "\n\n".join("\n".join([f"[{s.name}]"] + s.lines) for s in self.sections)


class CostEstimate(BaseModel):
    usd: float
    basis: str


class TakeResult(BaseModel):
    provider_ref: str
    path: Path
    duration_s: float


class Provider(Protocol):
    name: str
    limits: ProviderLimits

    def payload(self, req: GenerationRequest) -> dict: ...
    def estimate(self, req: GenerationRequest) -> CostEstimate: ...
    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float: ...
```

- [ ] **Step 4: Write `src/songcomposer/generate/prompt.py`**

```python
"""Fold the per-run fidelity answer into the style prompt (BUILD-SPEC decision #6).

loose  — the spec's own style words only
medium — + measured tempo, heard instrumentation and vocal character
close  — + measured key and meter, heard production and timbre

Tempo/key/meter come ONLY from the objective ear (global_info). If it has not run, they are
simply absent — nothing here invents them.
"""
from ..models import Analysis, SongSpec
from .provider import GenerationRequest

FIDELITY_LEVELS = ("loose", "medium", "close")


def build_request(spec: SongSpec, analysis: Analysis | None, fidelity: str, note: str = "",
                  n_takes: int = 3) -> GenerationRequest:
    if fidelity not in FIDELITY_LEVELS:
        raise ValueError(f"fidelity must be one of {FIDELITY_LEVELS}, got {fidelity!r}")
    parts = [spec.style_prompt.strip()]
    g = analysis.global_info if analysis else None
    s = analysis.subjective if analysis else None
    if fidelity in ("medium", "close"):
        if g:
            parts.append(f"{round(g.tempo_bpm)} BPM")
        if s:
            parts += s.instrumentation[:5]
            parts.append(s.vocal_character)
    if fidelity == "close":
        if g:
            parts.append(f"in {g.key}")
            if g.time_signature != "4/4":
                parts.append(f"{g.time_signature} time")
        if s:
            parts += [s.production, s.timbre]
    if note.strip():
        parts.append(note.strip())
    return GenerationRequest(
        title=spec.title, style_prompt=", ".join(p for p in parts if p),
        negative_style=spec.negative_style, vocal_gender=spec.vocal_gender,
        sections=spec.sections, target_duration_s=spec.target_duration_s, n_takes=n_takes)
```

- [ ] **Step 5: Write `src/songcomposer/generate/preflight.py`**

```python
"""Pre-flight validation — MANDATORY before any request reaches a paid generation API.

Ported from validate_chunk()/preflight() in the video project's
scripts/audiobook/generate_audiobook.py. Money is only spent on a request that passes every
check. If anything fails the run aborts with a per-line report and generates NOTHING.
"""
import re

from ..models import SourceInfo
from .provider import GenerationRequest, ProviderLimits

LINE_VALIDATORS = [
    ("markup", re.compile(r"[<>{}\[\]]")),                       # we add [Section] headers ourselves
    ("replacement-char", re.compile("�")),
    ("soft-hyphen", re.compile("\xad")),
    ("control-char", re.compile(r"[\x00-\x08\x0b-\x1f]")),
    ("placeholder", re.compile(r"(?i)\b(lorem ipsum|todo|tbd|placeholder|insert [a-z ]+ here|lyrics? (go|goes) here)\b")),
]
STYLE_OF = re.compile(r"(?i)\b(in the style of|sounds? like|à la|a la)\b")
MIN_WPS, MAX_WPS = 0.2, 4.5          # sung words per second; above this the model truncates or garbles, below it the section is mostly empty
SHINGLE = 6                          # consecutive shared words that count as copying the reference


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def banned_terms(source: SourceInfo | None) -> list[str]:
    """Artist / song names from the reference's provenance. They must never reach a style prompt."""
    if source is None:
        return []
    terms: list[str] = []
    for raw in filter(None, [source.uploader, source.title]):
        cleaned = re.sub(r"[\(\[].*?[\)\]]", "", raw)
        for piece in re.split(r"\s+[-–—|]\s+", cleaned):
            piece = piece.strip()
            if len(piece) >= 4 and piece not in terms:
                terms.append(piece)
    return terms


def validate_request(req: GenerationRequest, limits: ProviderLimits,
                     reference_lyrics: list[str], banned: list[str]) -> list[str]:
    problems: list[str] = []
    if not 1 <= len(req.title.strip()) <= limits.max_title_chars:
        problems.append(f"title: must be 1–{limits.max_title_chars} chars, got {len(req.title.strip())}")
    if not req.style_prompt.strip():
        problems.append("style_prompt: empty")
    if len(req.style_prompt) > limits.max_style_chars:
        problems.append(f"style_prompt: {len(req.style_prompt)} chars > limit {limits.max_style_chars}")
    if STYLE_OF.search(req.style_prompt):
        problems.append(f"style_prompt: says '{STYLE_OF.search(req.style_prompt).group(0)}' — describe the sound, do not name a style of someone")
    for field, text in (("title", req.title), ("style_prompt", req.style_prompt)):
        for term in banned:
            if re.search(rf"(?i)\b{re.escape(term)}\b", text):
                problems.append(f"{field}: contains banned term {term!r} (the reference's artist/title)")

    if not 2 <= len(req.sections) <= limits.max_sections:
        problems.append(f"sections: need at least 2 sections and at most {limits.max_sections}, got {len(req.sections)}")
    if sum(len(s.lines) for s in req.sections) < 4:
        problems.append("lyrics: no lyrics to sing (fewer than 4 lines in total)")

    ref = _norm_words(" ".join(reference_lyrics))
    ref_shingles = {tuple(ref[i:i + SHINGLE]) for i in range(len(ref) - SHINGLE + 1)}

    for s in req.sections:
        if not s.name.strip():
            problems.append("section: empty name")
        if not limits.min_section_s <= s.duration_s <= limits.max_section_s:
            problems.append(f"{s.name}: duration {s.duration_s}s outside {limits.min_section_s}–{limits.max_section_s}s")
        if len(s.lines) > limits.max_lines_per_section:
            problems.append(f"{s.name}: {len(s.lines)} lines > limit {limits.max_lines_per_section}")
        words = sum(len(_norm_words(ln)) for ln in s.lines)
        if s.lines and s.duration_s > 0 and not MIN_WPS <= words / s.duration_s <= MAX_WPS:
            problems.append(f"{s.name}: {words} words in {s.duration_s}s = {words / s.duration_s:.1f} words/sec "
                            f"(expected {MIN_WPS}–{MAX_WPS})")
        for i, line in enumerate(s.lines, start=1):
            where = f"{s.name} line {i}"
            if not line.strip():
                problems.append(f"{where}: empty-line")
                continue
            if len(line) > limits.max_line_chars:
                problems.append(f"{where}: line-too-long ({len(line)} > {limits.max_line_chars})")
            for name, rx in LINE_VALIDATORS:
                m = rx.search(line)
                if m:
                    problems.append(f"{where}: {name}: ...{line[max(0, m.start() - 20):m.end() + 20]}...")
            w = _norm_words(line)
            if any(tuple(w[j:j + SHINGLE]) in ref_shingles for j in range(len(w) - SHINGLE + 1)):
                problems.append(f"{where}: copies the reference lyrics ({SHINGLE}+ consecutive words): {line!r}")

    total = sum(s.duration_s for s in req.sections)
    if not limits.min_total_s <= total <= limits.max_total_s:
        problems.append(f"duration: sections total {total}s outside {limits.min_total_s}–{limits.max_total_s}s")
    if req.target_duration_s and abs(total - req.target_duration_s) > 0.15 * req.target_duration_s:
        problems.append(f"duration: sections sum to {total}s but target_duration_s is {req.target_duration_s}s (>15% apart)")
    if len(req.lyrics_text()) > limits.max_lyrics_chars:
        problems.append(f"lyrics: {len(req.lyrics_text())} chars > limit {limits.max_lyrics_chars}")
    return problems


def preflight(req: GenerationRequest, limits: ProviderLimits,
              reference_lyrics: list[str], banned: list[str]) -> None:
    problems = validate_request(req, limits, reference_lyrics, banned)
    if problems:
        print(f"PRE-FLIGHT FAILED — {len(problems)} problem(s), NOTHING was generated:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    lines = sum(len(s.lines) for s in req.sections)
    print(f"pre-flight passed: {len(req.sections)} sections, {lines} lines validated clean")
```

- [ ] **Step 6: Run tests, verify they pass**

Run: `uv run pytest tests/test_preflight.py tests/test_prompt.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/songcomposer/generate tests/test_preflight.py tests/test_prompt.py
git commit -m "feat: provider contract, fidelity prompt builder, mandatory pre-flight gate"
```

---

### Task 8: `compose` stage

**Files:**
- Create: `src/songcomposer/compose.py`
- Modify: `src/songcomposer/cli.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `chat_json`, `Analysis`, `Brief`, `SongSpec`, `SpecSection`, `build_request`, `validate_request`, `GENERIC_LIMITS`, `banned_terms`
- Produces:
  - `compose.ComposedSong` (pydantic; what the LLM returns): `title, style_prompt, negative_style, vocal_gender, target_duration_s, sections: list[SpecSection]`
  - `compose.summarise_analysis(analysis: Analysis) -> str`
  - `compose.run_compose(song: str, force: bool = False) -> SongSpec`

- [ ] **Step 1: Write the failing tests**

`tests/test_compose.py`:
```python
import pytest

from songcomposer import compose
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import Analysis, Brief, Chord, GlobalInfo, SourceInfo, SpecSection, Subjective, Word
from songcomposer.paths import SongPaths

ANALYSIS = Analysis(
    audio_sha1="a" * 40, engines={"subjective": "m"},
    global_info=GlobalInfo(key="A minor", key_confidence=0.8, tempo_bpm=92, time_signature="4/4",
                           loudness_lufs=-14, duration_s=200),
    chords=[Chord(symbol="Am", harte="A:min", root="A", quality="min", extensions=[], bass=None,
                  onset=0, duration=2, confidence=0.9),
            Chord(symbol="F", harte="F:maj", root="F", quality="maj", extensions=[], bass=None,
                  onset=2, duration=2, confidence=0.3)],
    lyrics=[Word(word="harbour", start=1, end=1.5, confidence=0.9)],
    subjective=Subjective(genre_tags=["folk"], instrumentation=["acoustic guitar"], timbre="warm",
                          vocal_character="breathy", vocal_gender="female", production="dry",
                          emotional_arc="sad to hopeful", arrangement_density=[], sections=[]))

GOOD = compose.ComposedSong(
    title="Glass Hour", style_prompt="sparse indie folk", negative_style="edm", vocal_gender="female",
    target_duration_s=120,
    sections=[SpecSection(name="Verse 1", duration_s=60, lines=["The kettle clicks off in the dark",
                          "Your coat still hangs behind the door", "I count the streetlights to the park",
                          "And lose my place at twenty-four"]),
              SpecSection(name="Chorus", duration_s=60, lines=["Glass hour, hold me still",
                          "Glass hour, against my will", "Turn the morning down", "Until you come around"])])


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    write_model(p.analysis, ANALYSIS)
    write_model(p.brief, Brief(text="A song about insomnia after a breakup.", origin="inline"))
    write_model(p.source_json, SourceInfo(origin="u", kind="url", title="Some Artist - Some Song",
                uploader="Some Artist", duration_s=200, sample_rate=44100, sha1="a" * 40, ingested_at="now"))
    return p


def test_summary_includes_measured_facts_and_marks_low_confidence_chords():
    s = compose.summarise_analysis(ANALYSIS)
    assert "A minor" in s and "92" in s and "breathy" in s
    assert "Am" in s and "F" not in s.split("Chord vocabulary")[1].split("\n")[0]   # 0.3-confidence F excluded


def test_summary_with_partial_analysis_says_so():
    s = compose.summarise_analysis(Analysis(audio_sha1="a" * 40, engines={}))
    assert "not measured" in s


def test_compose_writes_spec_and_passes_brief_and_analysis_to_writer(song, monkeypatch):
    seen = {}

    def fake(messages, model, out_type, **kw):
        seen["text"] = messages[-1]["content"]
        return GOOD

    monkeypatch.setattr(compose, "chat_json", fake)
    spec = compose.run_compose("demo")
    assert "insomnia" in seen["text"] and "A minor" in seen["text"]
    assert "Some Artist" not in seen["text"]                      # provenance never reaches the writer
    on_disk = read_json(song.spec)
    assert on_disk["title"] == "Glass Hour" and on_disk["fidelity"] is None
    assert spec.writer_model == "google/gemini-2.5-pro"


def test_compose_feeds_preflight_problems_back_once(song, monkeypatch):
    bad = GOOD.model_copy(deep=True)
    bad.sections[0].lines[0] = "TODO write this"
    replies = iter([bad, GOOD])
    prompts = []

    def fake(messages, model, out_type, **kw):
        prompts.append(messages[-1]["content"])
        return next(replies)

    monkeypatch.setattr(compose, "chat_json", fake)
    compose.run_compose("demo")
    assert len(prompts) == 2 and "placeholder" in prompts[1]


def test_compose_is_noop_when_spec_exists(song, monkeypatch):
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)
    compose.run_compose("demo")
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: pytest.fail("should not be called"))
    compose.run_compose("demo")
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_compose.py -v`
Expected: `ImportError: cannot import name 'compose'`.

- [ ] **Step 3: Write `src/songcomposer/compose.py`**

```python
"""Stage: compose. Brief + analysis → an ORIGINAL song: lyrics, section map, style directives."""
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from .clients.openrouter import chat_json
from .config import load_config
from .generate.preflight import banned_terms, validate_request
from .generate.prompt import build_request
from .generate.provider import GENERIC_LIMITS
from .jsonio import read_json, write_model
from .models import LOW_CONFIDENCE, Analysis, Brief, SongSpec, SourceInfo, SpecSection
from .paths import SongPaths


class ComposedSong(BaseModel):
    title: str
    style_prompt: str
    negative_style: str
    vocal_gender: Literal["male", "female", "any"]
    target_duration_s: float
    sections: list[SpecSection]


SYSTEM = """You are a professional songwriter. Write an ORIGINAL song that lives in the same musical
world as the reference described below, fulfilling the user's brief.

Rules:
- Original lyrics only. Never quote or closely paraphrase the reference's lyrics.
- Concrete images over abstractions. Singable lines: natural stresses, 4-10 words per line, consistent
  metre within a section, real rhymes or deliberate near-rhymes. A chorus that earns its repetition.
- style_prompt: 10-25 words describing genre, mood, instrumentation and vocal delivery for a music
  generator. NEVER name an artist, band or song, and never write "in the style of".
- negative_style: comma-separated things to avoid.
- sections: in performance order. name like "Intro", "Verse 1", "Pre-Chorus", "Chorus", "Bridge", "Outro".
  Instrumental sections have an empty lines list. duration_s per section between 5 and 110 seconds;
  allow roughly 2-3 sung words per second. Section durations MUST sum to target_duration_s.
  style_notes: a short arrangement direction for that section ("drums drop out", "full band, harmonies").
- No brackets, braces or angle brackets inside lyric lines. Max 200 characters per line, max 30 lines per section.
- target_duration_s: follow the brief; otherwise 150-210."""


def summarise_analysis(a: Analysis) -> str:
    out = []
    if a.global_info:
        g = a.global_info
        out.append(f"Measured: key {g.key} (confidence {g.key_confidence:.2f}), {g.tempo_bpm:.0f} BPM, "
                   f"{g.time_signature}, {g.duration_s:.0f}s long.")
    else:
        out.append("Key, tempo and chords: not measured (objective ear has not run). Do not assume any.")
    solid = [c.symbol for c in a.chords if c.confidence >= LOW_CONFIDENCE]
    if solid:
        vocab = ", ".join(sym for sym, _ in Counter(solid).most_common(8))
        out.append(f"Chord vocabulary (confident detections only): {vocab}")
    if a.sections:
        out.append("Form: " + " → ".join(f"{s.label} ({s.end - s.start:.0f}s)" for s in a.sections))
    if a.subjective:
        s = a.subjective
        out += [f"Genre: {', '.join(s.genre_tags)}", f"Instrumentation: {'; '.join(s.instrumentation)}",
                f"Timbre: {s.timbre}", f"Vocal: {s.vocal_character} ({s.vocal_gender})",
                f"Production: {s.production}", f"Emotional arc: {s.emotional_arc}"]
        if s.arrangement_density:
            out.append("Arrangement over time: " + "; ".join(
                f"{d.start:.0f}-{d.end:.0f}s {d.description}" for d in s.arrangement_density))
    if a.lyrics:
        out.append("Reference lyrics (for theme and prosody ONLY — do not reuse any line): "
                   + " ".join(w.word for w in a.lyrics)[:1500])
    return "\n".join(out)


def run_compose(song: str, force: bool = False) -> SongSpec:
    paths = SongPaths(song)
    if paths.spec.exists() and not force:
        print(f"- spec exists at {paths.spec} (use --force to rewrite)")
        return SongSpec(**read_json(paths.spec))
    analysis = Analysis(**read_json(paths.require(paths.analysis, "analyze")))
    brief = Brief(**read_json(paths.require(paths.brief, "brief")))
    source = SourceInfo(**read_json(paths.source_json)) if paths.source_json.exists() else None
    config = load_config()
    banned = banned_terms(source)
    ref_lyrics = [w.word for w in analysis.lyrics]

    user = f"## Brief\n{brief.text}\n\n## The reference, as analysed\n{summarise_analysis(analysis)}"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    print(f"> composing with {config.writer_model}")
    for attempt in range(2):
        composed = chat_json(messages, config.writer_model, ComposedSong)
        spec = SongSpec(**composed.model_dump(), writer_model=config.writer_model)
        problems = validate_request(build_request(spec, None, "loose"), GENERIC_LIMITS, ref_lyrics, banned)
        if not problems:
            break
        print(f"  draft {attempt + 1} has {len(problems)} pre-flight problem(s)" + ("; asking for a fix" if attempt == 0 else ""))
        messages = messages[:2] + [{"role": "user", "content": user + "\n\n## Your previous draft failed these checks — fix them\n"
                                    + "\n".join(problems) + "\n\nPrevious draft:\n" + composed.model_dump_json()}]
    write_model(paths.spec, spec)
    print(f"  “{spec.title}” — {len(spec.sections)} sections, {len(spec.all_lines())} lines, "
          f"{spec.target_duration_s:.0f}s → {paths.spec}")
    if problems:
        print("! spec still has problems — edit 03-spec.json by hand; `generate` will refuse it as is:")
        for p in problems:
            print(f"    {p}")
    return spec
```

- [ ] **Step 4: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def compose(song: str, force: bool = typer.Option(False, "--force", help="rewrite an existing spec")) -> None:
    """Brief + analysis → lyrics and song spec (work/<song>/03-spec.json). Edit the file freely afterwards."""
    from .compose import run_compose
    run_compose(song, force=force)
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_compose.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/compose.py src/songcomposer/cli.py tests/test_compose.py
git commit -m "feat: compose stage — original lyrics and song spec, pre-flight feedback loop"
```

---

### Task 9: `generate` orchestration — gate, cost confirmation, caching, manifest

Everything about spending money except the provider-specific HTTP. Tested end-to-end with a fake provider.

**Files:**
- Modify: `src/songcomposer/generate/__init__.py`, `src/songcomposer/cli.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `Provider`, `TakeResult`, `CostEstimate`, `build_request`, `FIDELITY_LEVELS`, `preflight`, `banned_terms`, `content_key`, `probe_duration`, `TakesManifest`, `GenerationRun`, `TakeRecord`, `SongSpec`, `Analysis`, `SourceInfo`
- Produces:
  - `generate.get_provider(name: str, config: Config) -> Provider` (lazy-imports `suno` / `elevenlabs`)
  - `generate.ask_fidelity(input_fn: Callable[[str], str]) -> tuple[str, str]`
  - `generate.sanity_check(path: Path, duration_s: float, target_s: float) -> str | None` (port of `sanity_check()` from the audiobook script)
  - `generate.run_generate(song, provider_name=None, regen=False, fidelity=None, note="", input_fn=input, provider=None) -> TakesManifest`
- Contract for providers: write take *i* (0-based) to `dest_dir / f"take-{start_index + i}.mp3"` and call `on_take` in order, immediately after each file is fully written.

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:
```python
import subprocess

import pytest

from songcomposer import generate
from songcomposer.generate.provider import GENERIC_LIMITS, CostEstimate, TakeResult
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import SongSpec, SpecSection
from songcomposer.paths import SongPaths

SPEC = SongSpec(
    title="Glass Hour", style_prompt="sparse indie folk", target_duration_s=120,
    sections=[SpecSection(name="Verse 1", duration_s=60, lines=[
                  "The kettle clicks off in the dark", "Your coat still hangs behind the door",
                  "I count the streetlights to the park", "And lose my place at twenty-four"]),
              SpecSection(name="Chorus", duration_s=60, lines=[
                  "Glass hour, hold me still", "Glass hour, against my will",
                  "Turn the morning down", "Until you come around"])])


@pytest.fixture(scope="session")
def mp3_bytes(tmp_path_factory):
    out = tmp_path_factory.mktemp("mp3") / "t.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=90",
                    "-b:a", "64k", str(out)], check=True)
    return out.read_bytes()


class FakeProvider:
    limits = GENERIC_LIMITS

    def __init__(self, name, mp3, n=3, fail_after=None):
        self.name, self.mp3, self.n, self.fail_after, self.calls = name, mp3, n, fail_after, 0

    def payload(self, req):
        return {"style": req.style_prompt, "lyrics": req.lyrics_text()}

    def estimate(self, req):
        return CostEstimate(usd=0.12, basis="fake")

    def generate(self, req, dest_dir, start_index, on_take):
        self.calls += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.n):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("provider blew up")
            path = dest_dir / f"take-{start_index + i}.mp3"
            path.write_bytes(self.mp3)
            on_take(TakeResult(provider_ref=f"ref-{i}", path=path, duration_s=90.0))
        return 0.11


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    write_model(p.spec, SPEC)
    return p


def answers(*values):
    it = iter(values)
    return lambda prompt: next(it)


def test_preflight_failure_never_reaches_the_provider(song, mp3_bytes):
    bad = SPEC.model_copy(deep=True)
    bad.sections[0].lines[0] = "TODO write this"
    write_model(song.spec, bad)
    fake = FakeProvider("fake", mp3_bytes)
    with pytest.raises(SystemExit):
        generate.run_generate("demo", fidelity="loose", provider=fake, input_fn=answers("yes"))
    assert fake.calls == 0 and not song.takes_json.exists()


@pytest.mark.parametrize("reply", ["y", "", "no", "Yes please"])
def test_anything_but_exact_yes_aborts_without_spending(song, mp3_bytes, reply, capsys):
    fake = FakeProvider("fake", mp3_bytes)
    with pytest.raises(SystemExit):
        generate.run_generate("demo", fidelity="loose", provider=fake, input_fn=answers(reply))
    assert fake.calls == 0
    assert "nothing was spent" in capsys.readouterr().out


def test_cost_is_printed_before_confirmation(song, mp3_bytes, capsys):
    prompts = []

    def input_fn(prompt):
        prompts.append((prompt, capsys.readouterr().out))
        return "yes"

    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes), input_fn=input_fn)
    assert "$0.12" in prompts[0][1] and "pre-flight passed" in prompts[0][1]


def test_happy_path_writes_three_takes_and_manifest(song, mp3_bytes):
    m = generate.run_generate("demo", fidelity="medium", note="more space",
                              provider=FakeProvider("fake", mp3_bytes), input_fn=answers("yes"))
    assert [t.file for t in m.all_takes()] == ["take-1.mp3", "take-2.mp3", "take-3.mp3"]
    assert all(song.take(i).exists() for i in (1, 2, 3))
    run = read_json(song.takes_json)["runs"][0]
    assert (run["provider"], run["cost_estimate_usd"], run["cost_actual_usd"]) == ("fake", 0.12, 0.11)
    assert run["fidelity"] == "medium" and "more space" in run["style_prompt"]
    assert run["payload"]["lyrics"].startswith("[Verse 1]")
    assert read_json(song.spec)["fidelity"] == "medium"


def test_rerun_is_a_noop_without_regen(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes), input_fn=answers("yes"))
    again = FakeProvider("fake", mp3_bytes)
    generate.run_generate("demo", fidelity="loose", provider=again,
                          input_fn=lambda p: pytest.fail("must not ask to spend again"))
    assert again.calls == 0


def test_second_provider_appends_takes_for_the_bakeoff(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("suno", mp3_bytes, n=4), input_fn=answers("yes"))
    m = generate.run_generate("demo", fidelity="loose", provider=FakeProvider("elevenlabs", mp3_bytes),
                              input_fn=answers("yes"))
    assert [(t.index, t.provider) for t in m.all_takes()] == [
        (1, "suno"), (2, "suno"), (3, "suno"), (4, "suno"), (5, "elevenlabs"), (6, "elevenlabs"), (7, "elevenlabs")]


def test_regen_replaces_only_that_providers_takes(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("suno", mp3_bytes, n=2), input_fn=answers("yes"))
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("elevenlabs", mp3_bytes, n=1), input_fn=answers("yes"))
    m = generate.run_generate("demo", fidelity="loose", regen=True, provider=FakeProvider("suno", mp3_bytes, n=2),
                              input_fn=answers("yes"))
    assert [(t.index, t.provider) for t in m.all_takes()] == [(3, "elevenlabs"), (4, "suno"), (5, "suno")]
    assert not song.take(1).exists() and song.take(4).exists()


def test_paid_takes_survive_a_mid_run_failure(song, mp3_bytes):
    with pytest.raises(RuntimeError, match="blew up"):
        generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes, fail_after=1),
                              input_fn=answers("yes"))
    run = read_json(song.takes_json)["runs"][0]
    assert len(run["takes"]) == 1 and "stopped early" in run["warnings"][0]


def test_sanity_check_flags_short_and_tiny(tmp_path, mp3_bytes):
    f = tmp_path / "a.mp3"
    f.write_bytes(mp3_bytes)
    assert generate.sanity_check(f, 90, 120) is None
    assert "duration" in generate.sanity_check(f, 20, 120)
    f.write_bytes(b"x" * 500)
    assert "small" in generate.sanity_check(f, 90, 120)


def test_ask_fidelity():
    assert generate.ask_fidelity(answers("2", "slower")) == ("medium", "slower")
    assert generate.ask_fidelity(answers("9", "close", "")) == ("close", "")
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_generate.py -v`
Expected: `AttributeError: module 'songcomposer.generate' has no attribute 'run_generate'`.

- [ ] **Step 3: Write `src/songcomposer/generate/__init__.py`**

```python
"""Stage: generate. The ONLY place paid generation happens.

Order is fixed and non-negotiable (CLAUDE.md hard rule):
  build request → cache check → PRE-FLIGHT → print estimated cost → explicit 'yes' → provider.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..config import Config, load_config
from ..hashing import content_key
from ..jsonio import read_json, write_model
from ..models import Analysis, GenerationRun, SongSpec, SourceInfo, TakeRecord, TakesManifest
from ..paths import SongPaths
from .preflight import banned_terms, preflight
from .prompt import FIDELITY_LEVELS, build_request
from .provider import Provider, TakeResult


def get_provider(name: str, config: Config) -> Provider:
    if name == "suno":
        from .suno import SunoProvider
        return SunoProvider(config.suno_model)
    if name == "elevenlabs":
        from .elevenlabs import ElevenLabsProvider
        return ElevenLabsProvider(config.elevenlabs_model)
    raise ValueError(f"unknown provider {name!r} — use suno or elevenlabs")


def ask_fidelity(input_fn: Callable[[str], str]) -> tuple[str, str]:
    print("How closely should this track the reference?\n"
          "  1  loose   — same genre and mood only\n"
          "  2  medium  — + its tempo, instrumentation and vocal character\n"
          "  3  close   — + its key, meter, production and timbre")
    while True:
        raw = input_fn("Fidelity [1/2/3]: ").strip().lower()
        if raw in ("1", "2", "3"):
            level = FIDELITY_LEVELS[int(raw) - 1]
            break
        if raw in FIDELITY_LEVELS:
            level = raw
            break
    return level, input_fn("Anything to add for this run? (enter to skip): ").strip()


def sanity_check(path: Path, duration_s: float, target_s: float) -> str | None:
    size = path.stat().st_size
    if size < 100_000:
        return f"{path.name}: suspiciously small audio ({size} bytes)"
    if duration_s < max(30.0, 0.5 * target_s) or duration_s > 2.0 * target_s:
        return f"{path.name}: duration {duration_s:.0f}s vs target {target_s:.0f}s"
    return None


def run_generate(song: str, provider_name: str | None = None, regen: bool = False,
                 fidelity: str | None = None, note: str = "",
                 input_fn: Callable[[str], str] = input, provider: Provider | None = None) -> TakesManifest:
    paths = SongPaths(song)
    config = load_config()
    spec = SongSpec(**read_json(paths.require(paths.spec, "compose")))
    analysis = Analysis(**read_json(paths.analysis)) if paths.analysis.exists() else None
    source = SourceInfo(**read_json(paths.source_json)) if paths.source_json.exists() else None
    provider = provider or get_provider(provider_name or config.default_provider, config)

    if fidelity is None:
        fidelity, note = ask_fidelity(input_fn)
    req = build_request(spec, analysis, fidelity, note)
    payload = provider.payload(req)
    request_hash = content_key(provider.name, json.dumps(payload, sort_keys=True, ensure_ascii=False))

    manifest = TakesManifest(**read_json(paths.takes_json)) if paths.takes_json.exists() else TakesManifest()
    same = [r for r in manifest.runs if r.request_hash == request_hash and r.takes]
    if same and not regen and all((paths.takes_dir / t.file).exists() for t in same[0].takes):
        print(f"- {len(same[0].takes)} {provider.name} take(s) already exist for this exact request "
              "(use --regen to pay for new ones)")
        return manifest

    # ---- HARD GATE: nothing below this line runs on unvalidated input -------------------
    preflight(req, provider.limits, [w.word for w in analysis.lyrics] if analysis else [], banned_terms(source))
    est = provider.estimate(req)
    print(f"\nprovider : {provider.name}\nstyle    : {req.style_prompt}\n"
          f"length   : {req.target_duration_s:.0f}s × {req.n_takes} takes\n"
          f"ESTIMATED COST: ${est.usd:.2f}  ({est.basis})")
    if input_fn("Type 'yes' to spend this and generate: ").strip().lower() != "yes":
        print("aborted — nothing was generated, nothing was spent")
        raise SystemExit(1)

    if regen:
        for old in [r for r in manifest.runs if r.provider == provider.name]:
            for t in old.takes:
                (paths.takes_dir / t.file).unlink(missing_ok=True)
            manifest.runs.remove(old)
    start = max([t.index for t in manifest.all_takes()], default=0) + 1
    run = GenerationRun(provider=provider.name, request_hash=request_hash, fidelity=fidelity, fidelity_note=note,
                        style_prompt=req.style_prompt, payload=payload, cost_estimate_usd=est.usd,
                        cost_actual_usd=0.0, created_at=datetime.now(timezone.utc).isoformat(), takes=[])
    manifest.runs.append(run)

    def on_take(result: TakeResult) -> None:
        index = start + len(run.takes)
        assert result.path.name == f"take-{index}.mp3", f"provider wrote {result.path.name}, expected take-{index}.mp3"
        warning = sanity_check(result.path, result.duration_s, req.target_duration_s)
        if warning:
            run.warnings.append(warning)
            print(f"  ! {warning}")
        run.takes.append(TakeRecord(index=index, file=result.path.name, provider=provider.name,
                                    provider_ref=result.provider_ref, duration_s=round(result.duration_s, 2),
                                    bytes=result.path.stat().st_size))
        write_model(paths.takes_json, manifest)          # persisted per take: paid audio is never orphaned
        print(f"  saved take {index} ({result.duration_s:.0f}s) → {result.path}")

    try:
        run.cost_actual_usd = round(provider.generate(req, paths.takes_dir, start, on_take), 4)
    except Exception as e:
        run.warnings.append(f"generation stopped early: {e}")
        raise
    finally:
        write_model(paths.takes_json, manifest)

    spec.fidelity, spec.fidelity_note = fidelity, note
    write_model(paths.spec, spec)
    print(f"done: {len(run.takes)} take(s), actual cost ${run.cost_actual_usd:.2f}. "
          f"Listen, then: songcomposer pick {song} --take N")
    return manifest
```

- [ ] **Step 4: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def generate(song: str,
             provider: str = typer.Option(None, "--provider", help="suno | elevenlabs (default: songcomposer.toml)"),
             regen: bool = typer.Option(False, "--regen", help="pay for fresh takes from this provider"),
             fidelity: str = typer.Option(None, "--fidelity", help="loose | medium | close (asked if omitted)"),
             note: str = typer.Option("", "--note", help="extra direction for this run")) -> None:
    """Spec → takes. Validates, prints estimated cost, and waits for an explicit 'yes' before spending."""
    from .generate import run_generate
    run_generate(song, provider_name=provider, regen=regen, fidelity=fidelity, note=note)
```

There is deliberately **no `--yes` flag.** Confirmation is always typed.

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/generate/__init__.py src/songcomposer/cli.py tests/test_generate.py
git commit -m "feat: generate orchestration — hard pre-flight gate, cost confirmation, per-take manifest, caching"
```

---

### Task 10: Suno provider (Kie.ai)

Port of `kiePost()` / `kieGet()` / `pollUnifiedJob()` from `lib.mjs`.

API facts (docs.kie.ai, checked 2026-09-20):
- `POST https://api.kie.ai/api/v1/jobs/createTask`, header `Authorization: Bearer <KIE_API_KEY>`. Body `{"model":"ai-music-api/generate","input":{...}}`. `callBackUrl` is optional on this endpoint — we poll instead.
- `input` (snake_case): `prompt` (the exact lyrics when `custom_mode` is true), `custom_mode`, `instrumental`, `model`, `style`, `title`, `negative_tags`, `vocal_gender` (`"m"`|`"f"`), `duration` (10–360, V6 family + custom mode only).
- Limits (V6): prompt 5000 chars, style 1000, title 80.
- Response `{"code":200,"msg":"success","data":{"taskId":"..."}}`. HTTP 200 with a non-200 `code` is an error (402 = insufficient credits).
- Poll `GET /api/v1/jobs/recordInfo?taskId=...` → `data.state` ∈ `waiting|queuing|generating|success|fail`; on success `data.resultJson` is a JSON **string**; `data.creditsConsumed` may be present; on fail `data.failMsg`.
- Each request returns multiple variations — 2 in every documented example.
- `GET /api/v1/chat/credit` → `{"code":200,"data":<int balance>}` (free).
- **UNVERIFIED:** the exact shape inside `resultJson` for Suno tasks. So: the raw poll response is written to disk *before* parsing, and `extract_tracks()` accepts every documented shape. If a real response matches none, the error names the raw file — the paid audio URLs are in it.
- Pricing is provisional: 12 credits ≈ $0.06 per request. Actual cost is measured from the credit balance before/after.

**Files:**
- Create: `src/songcomposer/clients/kie.py`, `src/songcomposer/generate/suno.py`
- Test: `tests/test_suno.py`

**Interfaces:**
- Produces:
  - `kie.kie_post(path: str, body: dict, client: httpx.Client) -> dict`, `kie.kie_get(path: str, client: httpx.Client) -> dict`
  - `kie.poll_job(task_id: str, client, *, label="task", timeout_s=900, sleep=time.sleep) -> dict` (returns the `data` dict on success)
  - `suno.extract_tracks(result: dict) -> list[dict]` — each `{"ref": str, "url": str}`
  - `suno.SunoProvider(model: str, client: httpx.Client | None = None, sleep=time.sleep)` implementing `Provider`

- [ ] **Step 1: Write the failing tests**

`tests/test_suno.py`:
```python
import json

import httpx
import pytest

from songcomposer.generate.provider import GenerationRequest
from songcomposer.generate.suno import SunoProvider, extract_tracks
from songcomposer.models import SpecSection

REQ = GenerationRequest(title="Glass Hour", style_prompt="sparse indie folk", negative_style="edm",
                        vocal_gender="female", target_duration_s=150, n_takes=3,
                        sections=[SpecSection(name="Verse 1", lines=["a b c", "d e f"], duration_s=75),
                                  SpecSection(name="Chorus", lines=["g h i"], duration_s=75)])


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("KIE_API_KEY", "kie-test")


@pytest.mark.parametrize("result", [
    {"resultUrls": ["https://x/a.mp3", "https://x/b.mp3"]},
    {"sunoData": [{"id": "1", "audio_url": "https://x/a.mp3"}, {"id": "2", "audio_url": "https://x/b.mp3"}]},
    {"data": [{"id": "1", "audioUrl": "https://x/a.mp3"}, {"id": "2", "audioUrl": "https://x/b.mp3"}]},
    {"response": {"sunoData": [{"id": "1", "audio_url": "https://x/a.mp3"}, {"id": "2", "audio_url": "https://x/b.mp3"}]}},
])
def test_extract_tracks_accepts_every_documented_shape(result):
    assert [t["url"] for t in extract_tracks(result)] == ["https://x/a.mp3", "https://x/b.mp3"]


def test_extract_tracks_unknown_shape_is_empty():
    assert extract_tracks({"something": "else"}) == []


def test_payload_is_custom_mode_with_exact_lyrics():
    p = SunoProvider("V6").payload(REQ)
    assert p["n_requests"] == 2                                  # 3 takes at 2 tracks per request
    body = p["body"]
    assert body["model"] == "ai-music-api/generate" and "callBackUrl" not in body
    i = body["input"]
    assert i["custom_mode"] is True and i["instrumental"] is False and i["model"] == "V6"
    assert i["prompt"] == "[Verse 1]\na b c\nd e f\n\n[Chorus]\ng h i"
    assert (i["style"], i["title"], i["negative_tags"], i["vocal_gender"], i["duration"]) == (
        "sparse indie folk", "Glass Hour", "edm", "f", 150)


def test_any_gender_omits_the_field():
    assert "vocal_gender" not in SunoProvider("V6").payload(REQ.model_copy(update={"vocal_gender": "any"}))["body"]["input"]


def test_estimate_is_two_requests_and_says_provisional():
    e = SunoProvider("V6").estimate(REQ)
    assert e.usd == pytest.approx(0.12) and "provisional" in e.basis


def _server(states, credits):
    """states: per-task list of successive poll payloads."""
    created, credit_iter = [], iter(credits)

    def handler(req: httpx.Request):
        assert req.headers["authorization"] == "Bearer kie-test"
        url = str(req.url)
        if url.endswith("/chat/credit"):
            return httpx.Response(200, json={"code": 200, "data": next(credit_iter)})
        if url.endswith("/jobs/createTask"):
            created.append(json.loads(req.content))
            return httpx.Response(200, json={"code": 200, "msg": "success", "data": {"taskId": f"t{len(created)}"}})
        if "/jobs/recordInfo" in url:
            task = req.url.params["taskId"]
            return httpx.Response(200, json={"code": 200, "data": states[task].pop(0)})
        return httpx.Response(200, content=b"ID3" + b"\x00" * 200_000)      # the mp3 download

    return handler, created


def test_generate_polls_saves_raw_downloads_and_measures_cost(tmp_path):
    ok = lambda a, b: {"state": "success", "resultJson": json.dumps(
        {"sunoData": [{"id": a, "audio_url": f"https://cdn/{a}.mp3", "duration": 151.2},
                      {"id": b, "audio_url": f"https://cdn/{b}.mp3", "duration": 149.0}]})}
    handler, created = _server({"t1": [{"state": "queuing"}, {"state": "generating"}, ok("a", "b")],
                                "t2": [ok("c", "d")]}, credits=[1000, 976])
    got = []
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    cost = provider.generate(REQ, tmp_path, 1, got.append)
    assert len(created) == 2
    assert [t.path.name for t in got] == ["take-1.mp3", "take-2.mp3", "take-3.mp3", "take-4.mp3"]
    assert [t.provider_ref for t in got] == ["a", "b", "c", "d"] and got[0].duration_s == 151.2
    assert (tmp_path / "raw-suno-t1.json").exists() and (tmp_path / "raw-suno-t2.json").exists()
    assert cost == pytest.approx(24 * 0.005)


def test_failed_task_raises_with_provider_message(tmp_path):
    handler, _ = _server({"t1": [{"state": "fail", "failMsg": "SENSITIVE_WORD_ERROR"}], "t2": []}, credits=[10, 10])
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="SENSITIVE_WORD_ERROR"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)


def test_unparseable_success_points_at_the_raw_file(tmp_path):
    handler, _ = _server({"t1": [{"state": "success", "resultJson": "{\"odd\": 1}"}], "t2": []}, credits=[10, 10])
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="raw-suno-t1.json"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)


def test_insufficient_credits_code_in_200_body_is_an_error(tmp_path):
    def handler(req):
        if str(req.url).endswith("/chat/credit"):
            return httpx.Response(200, json={"code": 200, "data": 0})
        return httpx.Response(200, json={"code": 402, "msg": "Insufficient credits"})
    provider = SunoProvider("V6", client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="402"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_suno.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.generate.suno'`.

- [ ] **Step 3: Write `src/songcomposer/clients/kie.py`**

```python
"""Kie.ai client. Port of kiePost()/kieGet()/pollUnifiedJob() from the video project's lib.mjs."""
import time
from typing import Callable

import httpx

from ..env import require_env

BASE = "https://api.kie.ai/api/v1"


def _headers() -> dict:
    return {"Authorization": f"Bearer {require_env('KIE_API_KEY')}"}


def _check(res: httpx.Response, what: str) -> dict:
    try:
        data = res.json()
    except ValueError:
        data = {}
    if res.status_code != 200 or data.get("code") not in (None, 200):
        raise RuntimeError(f"Kie.ai {what} failed ({data.get('code', res.status_code)}): {res.text[:500]}")
    return data


def kie_post(path: str, body: dict, client: httpx.Client) -> dict:
    return _check(client.post(f"{BASE}{path}", json=body, headers=_headers()), f"POST {path}")


def kie_get(path: str, client: httpx.Client) -> dict:
    return _check(client.get(f"{BASE}{path}", headers=_headers()), f"GET {path}")


def poll_job(task_id: str, client: httpx.Client, *, label: str = "task", timeout_s: float = 900,
             sleep: Callable[[float], None] = time.sleep) -> dict:
    deadline = time.monotonic() + timeout_s
    delay = 4.0
    while time.monotonic() < deadline:
        data = kie_get(f"/jobs/recordInfo?taskId={task_id}", client).get("data") or {}
        state = data.get("state", "unknown")
        print(f"  [{label}] {state}      ", end="\r")
        if state == "success":
            print()
            return data
        if state == "fail":
            print()
            raise RuntimeError(f"Kie.ai {label} failed: {data.get('failMsg') or data.get('failCode') or 'no detail'}")
        sleep(delay)
        delay = min(delay + 2.0, 12.0)
    raise RuntimeError(f"Kie.ai {label} timed out after {timeout_s}s (taskId {task_id} — it may still finish; "
                       "check the Kie.ai dashboard before regenerating)")
```

- [ ] **Step 4: Write `src/songcomposer/generate/suno.py`**

```python
"""Suno via Kie.ai. Custom mode: our lyrics are sung exactly; style and title are ours."""
import json
import math
import time
from pathlib import Path
from typing import Callable

import httpx

from ..audio import probe_duration
from ..clients.kie import kie_get, kie_post, poll_job
from ..jsonio import write_json
from .provider import CostEstimate, GenerationRequest, ProviderLimits, TakeResult

TRACKS_PER_REQUEST = 2
CREDITS_PER_REQUEST = 12        # provisional — from kie.ai/suno-api marketing page, 2026-09-20
USD_PER_CREDIT = 0.005          # provisional — inferred from "12 credits ≈ $0.06"


def extract_tracks(result: dict) -> list[dict]:
    """resultJson's shape for Suno tasks is not documented. Accept every shape Kie documents anywhere."""
    if isinstance(result.get("resultUrls"), list):
        return [{"ref": url.rsplit("/", 1)[-1], "url": url} for url in result["resultUrls"]]
    for items in (result.get("sunoData"), result.get("data"), (result.get("response") or {}).get("sunoData")):
        if isinstance(items, list):
            tracks = [{"ref": str(it.get("id", "")), "url": it.get("audio_url") or it.get("audioUrl"),
                       "duration": it.get("duration")} for it in items if isinstance(it, dict)]
            tracks = [t for t in tracks if t["url"]]
            if tracks:
                return tracks
    return []


class SunoProvider:
    name = "suno"
    limits = ProviderLimits(max_title_chars=80, max_style_chars=1000, max_lyrics_chars=5000, max_line_chars=200,
                            max_lines_per_section=60, max_sections=30, min_section_s=3, max_section_s=360,
                            min_total_s=30, max_total_s=360)

    def __init__(self, model: str, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self._client, self._sleep = model, client, sleep

    def _n_requests(self, req: GenerationRequest) -> int:
        return math.ceil(req.n_takes / TRACKS_PER_REQUEST)

    def payload(self, req: GenerationRequest) -> dict:
        inp = {"prompt": req.lyrics_text(), "custom_mode": True, "instrumental": False, "model": self.model,
               "style": req.style_prompt, "title": req.title,
               "duration": int(min(360, max(10, round(req.target_duration_s))))}
        if req.negative_style:
            inp["negative_tags"] = req.negative_style
        if req.vocal_gender != "any":
            inp["vocal_gender"] = "m" if req.vocal_gender == "male" else "f"
        return {"n_requests": self._n_requests(req), "body": {"model": "ai-music-api/generate", "input": inp}}

    def estimate(self, req: GenerationRequest) -> CostEstimate:
        n = self._n_requests(req)
        return CostEstimate(usd=n * CREDITS_PER_REQUEST * USD_PER_CREDIT,
                            basis=f"{n} Suno requests × {CREDITS_PER_REQUEST} credits, provisional pricing; "
                                  f"yields {n * TRACKS_PER_REQUEST} takes — Suno returns pairs")

    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float:
        client = self._client or httpx.Client(timeout=120.0)
        dest_dir.mkdir(parents=True, exist_ok=True)
        spec = self.payload(req)
        before = kie_get("/chat/credit", client).get("data")
        print(f"  Kie.ai credit balance: {before}")
        task_ids = [kie_post("/jobs/createTask", spec["body"], client)["data"]["taskId"]
                    for _ in range(spec["n_requests"])]
        index = start_index
        for task_id in task_ids:
            data = poll_job(task_id, client, label=f"suno {task_id[:8]}", sleep=self._sleep)
            raw = dest_dir / f"raw-suno-{task_id}.json"
            write_json(raw, data)                                   # BEFORE parsing: the paid URLs are in here
            tracks = extract_tracks(json.loads(data.get("resultJson") or "{}"))
            if not tracks:
                raise RuntimeError(f"Suno succeeded but no audio URLs were recognised — the paid result is saved at "
                                   f"{raw}; add its shape to extract_tracks()")
            for track in tracks:
                path = dest_dir / f"take-{index}.mp3"
                with client.stream("GET", track["url"]) as res:
                    res.raise_for_status()
                    with open(path, "wb") as f:
                        for block in res.iter_bytes():
                            f.write(block)
                duration = track.get("duration") or probe_duration(path)
                on_take(TakeResult(provider_ref=track["ref"], path=path, duration_s=float(duration)))
                index += 1
        after = kie_get("/chat/credit", client).get("data")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before >= after:
            return (before - after) * USD_PER_CREDIT
        return self.estimate(req).usd
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_suno.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/clients/kie.py src/songcomposer/generate/suno.py tests/test_suno.py
git commit -m "feat: Suno provider via Kie.ai — custom-mode lyrics, polling, raw-result safety net, measured cost"
```

---

### Task 11: ElevenLabs Music provider

Adapts `scripts/ai-video/generate-music.mjs` (the working call) and `elevenPost()` from `lib.mjs`: three takes, `music_v2_5`, per-section control via the `chunks[]` composition plan.

API facts (elevenlabs.io/docs, checked 2026-09-20):
- `POST https://api.elevenlabs.io/v1/music?output_format=mp3_44100_192`, header `xi-api-key`. Response body is the raw audio.
- Body: `composition_plan`, `model_id` (`music_v1`|`music_v2`|`music_v2_5`), `seed` (int; only valid with a plan — which is what we send).
- v2-family plan: `{"chunks":[{"text":"[Verse]\nline\nline","duration_ms":15000,"positive_styles":[...],"negative_styles":[...],"context_adherence":"high"}]}`. ≤ 30 chunks; each chunk 3000–120000 ms; ≤ 30 lines of ≤ 200 chars; `positive_styles`/`negative_styles` ≤ 50 entries; total ≤ 10 min.
- Rejections: HTTP 4xx with `detail.status == "bad_composition_plan"` and `detail.data.composition_plan_suggestion`.
- Pricing: $0.15 per generated minute (provisional — read via the pricing page).

**Files:**
- Create: `src/songcomposer/clients/elevenlabs.py`, `src/songcomposer/generate/elevenlabs.py`
- Test: `tests/test_elevenlabs.py`

**Interfaces:**
- Produces:
  - `clients.elevenlabs.eleven_post(path: str, body: dict, client: httpx.Client, params: dict | None = None) -> httpx.Response`
  - `generate.elevenlabs.style_tags(text: str) -> list[str]`
  - `generate.elevenlabs.ElevenLabsProvider(model: str, client: httpx.Client | None = None)` implementing `Provider`

- [ ] **Step 1: Write the failing tests**

`tests/test_elevenlabs.py`:
```python
import json

import httpx
import pytest

from songcomposer.generate.elevenlabs import ElevenLabsProvider, style_tags
from songcomposer.generate.provider import GenerationRequest
from songcomposer.jsonio import read_json
from songcomposer.models import SpecSection

REQ = GenerationRequest(title="Glass Hour", style_prompt="sparse indie folk, breathy female vocal, 92 BPM",
                        negative_style="edm, autotune", target_duration_s=150, n_takes=3,
                        sections=[SpecSection(name="Verse 1", lines=["a b c", "d e f"], duration_s=75,
                                              style_notes="solo fingerpicked guitar"),
                                  SpecSection(name="Chorus", lines=["g h i"], duration_s=75)])


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")


def test_style_tags_split_trim_dedupe_and_cap():
    assert style_tags(" folk,  breathy vocal , folk,") == ["folk", "breathy vocal"]
    assert len(style_tags(",".join(f"t{i}" for i in range(80)))) == 50


def test_v1_model_is_refused():
    with pytest.raises(ValueError, match="chunks"):
        ElevenLabsProvider("music_v1")


def test_payload_is_a_v2_chunk_plan_with_three_seeds():
    p = ElevenLabsProvider("music_v2_5").payload(REQ)
    assert p["model_id"] == "music_v2_5" and p["seeds"] == [11, 22, 33]
    c0, c1 = p["composition_plan"]["chunks"]
    assert c0["text"] == "[Verse 1]\na b c\nd e f" and c0["duration_ms"] == 75000
    assert c0["positive_styles"] == ["sparse indie folk", "breathy female vocal", "92 BPM", "solo fingerpicked guitar"]
    assert c0["negative_styles"] == ["edm", "autotune"] and c0["context_adherence"] == "high"
    assert c1["positive_styles"] == ["sparse indie folk", "breathy female vocal", "92 BPM"]


def test_estimate_is_per_minute_times_takes():
    e = ElevenLabsProvider("music_v2_5").estimate(REQ)
    assert e.usd == pytest.approx(3 * 2.5 * 0.15)


def test_generate_makes_one_call_per_seed_and_writes_audio(tmp_path, monkeypatch):
    from songcomposer.generate import elevenlabs as mod
    monkeypatch.setattr(mod, "probe_duration", lambda p: 148.0)
    seen = []

    def handler(req: httpx.Request):
        assert req.headers["xi-api-key"] == "el-test"
        assert req.url.params["output_format"] == "mp3_44100_192"
        seen.append(json.loads(req.content))
        return httpx.Response(200, content=b"ID3" + b"\x00" * 1000)

    got = []
    cost = ElevenLabsProvider("music_v2_5", client=httpx.Client(transport=httpx.MockTransport(handler))).generate(
        REQ, tmp_path, 5, got.append)
    assert [b["seed"] for b in seen] == [11, 22, 33]
    assert all("chunks" in b["composition_plan"] and b["model_id"] == "music_v2_5" and "seeds" not in b for b in seen)
    assert [t.path.name for t in got] == ["take-5.mp3", "take-6.mp3", "take-7.mp3"]
    assert cost == pytest.approx(3 * 148.0 / 60 * 0.15)


def test_rejected_plan_saves_the_suggestion_and_raises(tmp_path):
    suggestion = {"chunks": [{"text": "[Verse]\nsafe", "duration_ms": 10000, "positive_styles": ["folk"]}]}

    def handler(req):
        return httpx.Response(400, json={"detail": {"status": "bad_composition_plan", "message": "copyright",
                                                    "data": {"composition_plan_suggestion": suggestion}}})

    provider = ElevenLabsProvider("music_v2_5", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(RuntimeError, match="bad_composition_plan"):
        provider.generate(REQ, tmp_path, 1, lambda t: None)
    assert read_json(tmp_path / "elevenlabs-plan-suggestion.json") == suggestion
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_elevenlabs.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.generate.elevenlabs'`.

- [ ] **Step 3: Write `src/songcomposer/clients/elevenlabs.py`**

```python
"""ElevenLabs client. Port of elevenPost() from the video project's lib.mjs."""
import httpx

from ..env import require_env

BASE = "https://api.elevenlabs.io/v1"


def eleven_post(path: str, body: dict, client: httpx.Client, params: dict | None = None) -> httpx.Response:
    return client.post(f"{BASE}{path}", json=body, params=params or {},
                       headers={"xi-api-key": require_env("ELEVENLABS_API_KEY")})
```

- [ ] **Step 4: Write `src/songcomposer/generate/elevenlabs.py`**

```python
"""ElevenLabs Music. Adapted from the video project's scripts/ai-video/generate-music.mjs:
three takes, v2-family model, per-section control through a chunks[] composition plan."""
from pathlib import Path
from typing import Callable

import httpx

from ..audio import probe_duration
from ..clients.elevenlabs import eleven_post
from ..jsonio import write_json
from .provider import CostEstimate, GenerationRequest, ProviderLimits, TakeResult

USD_PER_MINUTE = 0.15            # provisional — elevenlabs.io/pricing/api, 2026-09-20
SEEDS = [11, 22, 33, 44, 55, 66]
MAX_STYLES = 50


def style_tags(text: str) -> list[str]:
    tags: list[str] = []
    for raw in text.split(","):
        tag = raw.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:MAX_STYLES]


class ElevenLabsProvider:
    name = "elevenlabs"
    limits = ProviderLimits(max_title_chars=80, max_style_chars=1000, max_lyrics_chars=20000, max_line_chars=200,
                            max_lines_per_section=30, max_sections=30, min_section_s=3, max_section_s=120,
                            min_total_s=30, max_total_s=600)

    def __init__(self, model: str, client: httpx.Client | None = None):
        if not model.startswith("music_v2"):
            raise ValueError(f"{model}: only the v2 family is supported — it takes the chunks[] plan this provider builds")
        self.model, self._client = model, client

    def payload(self, req: GenerationRequest) -> dict:
        positive, negative = style_tags(req.style_prompt), style_tags(req.negative_style)
        chunks = []
        for s in req.sections:
            chunk = {"text": "\n".join([f"[{s.name}]"] + s.lines), "duration_ms": int(round(s.duration_s * 1000)),
                     "positive_styles": (positive + [t for t in style_tags(s.style_notes) if t not in positive])[:MAX_STYLES],
                     "context_adherence": "high"}
            if negative:
                chunk["negative_styles"] = negative
            chunks.append(chunk)
        return {"model_id": self.model, "composition_plan": {"chunks": chunks}, "seeds": SEEDS[:req.n_takes]}

    def estimate(self, req: GenerationRequest) -> CostEstimate:
        minutes = sum(s.duration_s for s in req.sections) / 60
        return CostEstimate(usd=req.n_takes * minutes * USD_PER_MINUTE,
                            basis=f"{req.n_takes} takes × {minutes:.1f} min × ${USD_PER_MINUTE}/min, provisional pricing")

    def generate(self, req: GenerationRequest, dest_dir: Path, start_index: int,
                 on_take: Callable[[TakeResult], None]) -> float:
        client = self._client or httpx.Client(timeout=900.0)
        dest_dir.mkdir(parents=True, exist_ok=True)
        spec = self.payload(req)
        minutes = 0.0
        for i, seed in enumerate(spec["seeds"]):
            print(f"  [elevenlabs] take {i + 1}/{len(spec['seeds'])} (seed {seed}) …")
            body = {"model_id": spec["model_id"], "composition_plan": spec["composition_plan"], "seed": seed}
            res = eleven_post("/music", body, client, params={"output_format": "mp3_44100_192"})
            if res.status_code != 200:
                self._raise(res, dest_dir)
            path = dest_dir / f"take-{start_index + i}.mp3"
            path.write_bytes(res.content)
            duration = probe_duration(path)
            minutes += duration / 60
            on_take(TakeResult(provider_ref=res.headers.get("song-id", f"seed-{seed}"), path=path, duration_s=duration))
        return minutes * USD_PER_MINUTE

    @staticmethod
    def _raise(res: httpx.Response, dest_dir: Path) -> None:
        try:
            detail = res.json().get("detail") or {}
        except ValueError:
            detail = {}
        status = detail.get("status") if isinstance(detail, dict) else None
        suggestion = (detail.get("data") or {}).get("composition_plan_suggestion") if isinstance(detail, dict) else None
        if suggestion:
            out = dest_dir / "elevenlabs-plan-suggestion.json"
            write_json(out, suggestion)
            raise RuntimeError(f"ElevenLabs rejected the plan ({status}). Its suggested replacement is saved at {out} — "
                               "compare it with 03-spec.json, edit the spec, and re-run.")
        raise RuntimeError(f"ElevenLabs music failed ({res.status_code}, {status}): {res.text[:500]}")
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `uv run pytest tests/test_elevenlabs.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/clients/elevenlabs.py src/songcomposer/generate/elevenlabs.py tests/test_elevenlabs.py
git commit -m "feat: ElevenLabs Music provider — v2.5 chunk plans, three seeded takes, plan-rejection handling"
```

---

### Task 12: `pick` stage — and MILESTONE A: the first real song and the bake-off

**Files:**
- Create: `src/songcomposer/pick.py`, `templates/first-song.md`
- Modify: `src/songcomposer/cli.py`
- Test: `tests/test_pick.py`

**Interfaces:**
- Produces: `pick.run_pick(song: str, take: int) -> Chosen` — writes `05-chosen.json`, copies the take to `out/<song>/<song>.mp3`, and deletes a stale `06-transcription.json` if the choice changed.

- [ ] **Step 1: Write the failing tests**

`tests/test_pick.py`:
```python
import pytest

from songcomposer.jsonio import read_json, write_model
from songcomposer.models import GenerationRun, TakeRecord, TakesManifest
from songcomposer.paths import SongPaths
from songcomposer.pick import run_pick


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    p.takes_dir.mkdir(parents=True)
    takes = []
    for i, provider in ((1, "suno"), (2, "suno"), (3, "elevenlabs")):
        p.take(i).write_bytes(f"audio-{i}".encode())
        takes.append(TakeRecord(index=i, file=f"take-{i}.mp3", provider=provider, provider_ref="r",
                                duration_s=150, bytes=7))
    write_model(p.takes_json, TakesManifest(runs=[GenerationRun(
        provider="mixed", request_hash="h", fidelity="loose", fidelity_note="", style_prompt="s", payload={},
        cost_estimate_usd=0, cost_actual_usd=0, created_at="now", takes=takes)]))
    return p


def test_pick_records_choice_and_copies_deliverable(song):
    chosen = run_pick("demo", 3)
    assert (chosen.take, chosen.provider, chosen.file) == (3, "elevenlabs", "take-3.mp3")
    assert song.out_mp3.read_bytes() == b"audio-3"
    assert read_json(song.chosen)["sha1"] == chosen.sha1


def test_pick_unknown_take_lists_the_valid_ones(song):
    with pytest.raises(ValueError, match="1, 2, 3"):
        run_pick("demo", 9)


def test_changing_the_pick_invalidates_the_old_transcription(song):
    run_pick("demo", 1)
    song.transcription.write_text("{}", encoding="utf-8")
    run_pick("demo", 1)
    assert song.transcription.exists()            # same take: keep
    run_pick("demo", 2)
    assert not song.transcription.exists()        # different take: the old chart would describe the wrong audio
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_pick.py -v`
Expected: `ModuleNotFoundError: No module named 'songcomposer.pick'`.

- [ ] **Step 3: Write `src/songcomposer/pick.py`**

```python
"""Stage: pick. Only the chosen take is ever transcribed."""
import shutil
from datetime import datetime, timezone

from .hashing import file_sha1
from .jsonio import read_json, write_model
from .models import Chosen, TakesManifest
from .paths import SongPaths


def run_pick(song: str, take: int) -> Chosen:
    paths = SongPaths(song)
    manifest = TakesManifest(**read_json(paths.require(paths.takes_json, "generate")))
    by_index = {t.index: t for t in manifest.all_takes()}
    if take not in by_index or not (paths.takes_dir / by_index[take].file).exists():
        raise ValueError(f"no take {take} — available: {', '.join(str(i) for i in sorted(by_index))}")
    record = by_index[take]
    src = paths.takes_dir / record.file
    chosen = Chosen(take=take, file=record.file, provider=record.provider, sha1=file_sha1(src),
                    chosen_at=datetime.now(timezone.utc).isoformat())
    if paths.chosen.exists() and read_json(paths.chosen).get("sha1") != chosen.sha1:
        paths.transcription.unlink(missing_ok=True)
    write_model(paths.chosen, chosen)
    paths.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, paths.out_mp3)
    print(f"  take {take} ({record.provider}) → {paths.out_mp3}")
    return chosen
```

- [ ] **Step 4: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def pick(song: str, take: int = typer.Option(..., "--take", help="take number from 04-takes/")) -> None:
    """Choose a take → 05-chosen.json and out/<song>/<song>.mp3."""
    from .pick import run_pick
    run_pick(song, take)
```

- [ ] **Step 5: Run the whole suite, verify it passes**

Run: `uv run pytest -v`
Expected: every test so far PASSES. No network was touched.

- [ ] **Step 6: Write `templates/first-song.md`**

````markdown
# Recipe: first song + provider bake-off

Authoritative recipe for producing a song before the objective (DSP) ear exists, and for deciding the
default provider. Costs real money at step 5 — roughly $0.12 (Suno) + $1.10 (ElevenLabs, 3 × 2.5 min).

1. `uv run songcomposer ingest <song> --from "<youtube url or file>"`
2. `uv run songcomposer analyze <song> --ears subjective`
   Open `work/<song>/01-analysis.json`. Does the description match what you hear? If it reads generic,
   try `listener_model = "google/gemini-3.1-pro-preview"` in `songcomposer.toml`, delete `work/<song>/.cache/subjective-*`, re-run.
   If OpenRouter errors on the audio part, that is the BUILD-SPEC §10 risk: get a `GOOGLE_AI_API_KEY` and stop here.
3. `uv run songcomposer brief <song> --from brief.md`   (or `--text "..."`)
4. `uv run songcomposer compose <song>`
   **Read and edit `work/<song>/03-spec.json`.** The lyrics are yours to rewrite. `generate` re-validates whatever you leave there.
5. Bake-off — same spec, same fidelity answer, both providers:
   `uv run songcomposer generate <song> --provider suno --fidelity medium`
   `uv run songcomposer generate <song> --provider elevenlabs --fidelity medium`
   Each prints the pre-flight result and estimated cost and waits for you to type `yes`.
6. Listen to `work/<song>/04-takes/take-*.mp3` side by side. `takes.json` says which provider made which.
   Judge: vocal believability, lyric intelligibility, faithfulness to the section plan, production.
7. Set the winner: edit `default_provider` in `songcomposer.toml`. Record the reasoning in `docs/DECISIONS.md`
   under a new heading "Bake-off result" (licensing may override sound for client work — see DECISIONS.md).
8. `uv run songcomposer pick <song> --take N`  →  `out/<song>/<song>.mp3`

If Suno's first real response fails with "no audio URLs were recognised": the paid result is safe in
`04-takes/raw-suno-<taskId>.json`. Add the shape you see there to `extract_tracks()` and its test, then
download the URLs by hand — do not regenerate.
````

- [ ] **Step 7: Commit**

```bash
git add src/songcomposer/pick.py src/songcomposer/cli.py tests/test_pick.py templates/first-song.md
git commit -m "feat: pick stage and first-song recipe — Milestone A (songs can be generated)"
```

- [ ] **Step 8: MILESTONE A — run `templates/first-song.md` for real (human step, spends ~$1.25)**

This is the first point at which real songs come out. It also resolves three UNVERIFIED items from the API research: whether OpenRouter passes audio through to Gemini, the Suno `resultJson` shape, and real per-request cost. Record what you find at the bottom of `docs/spikes/2026-09-20-mir-stack.md` under "Live API findings". Fix forward (test first) if a parser needs a new shape.

---

## Part 2 — the objective ear (Tasks 13–20)

Read `docs/spikes/2026-09-20-mir-stack.md` first. What the spike established on this machine (Python 3.11.16, RTX 4070 SUPER, CUDA 12.8):

| Job | Library | Spike result |
|---|---|---|
| Stems | `demucs` 4.1.0 | installs, imports; model download untested |
| Beats / downbeats | `beat_this` 1.1.0, `dbn=False` | runs on CUDA |
| **Chords** | **`lv-chordia` 1.1.0** | **C / Am / F / G recovered exactly, boundaries within 40 ms.** Needs an ABSOLUTE path. **Emits no confidence** — Task 16 computes one. |
| Notes | `basic-pitch` 0.4.0 (ONNX) | runs; forces `numpy==1.26.4` |
| Lyrics | `faster-whisper` 1.2.1 | runs CUDA float16 — *with torch imported first* (torch ships the cuDNN DLLs) |
| Key / loudness / structure features | `librosa` 0.11.0 (+ `pyloudnorm`) | fine |
| ~~madmom~~, ~~vamp/Chordino~~ | — | **cannot build: no MSVC on this machine. Dropped.** |

Testing strategy for this part is BUILD-SPEC §11: **manufacture ground truth.** `tests/synth.py` renders a progression we wrote, in a known key at a known tempo; every component must recover it. Tests that need model weights or the GPU are marked `@pytest.mark.gpu`; fast runs use `uv run pytest -m "not gpu"`.

---

### Task 13: Analysis dependencies and the synthetic ground-truth fixture

**Files:**
- Modify: `pyproject.toml`, `tests/conftest.py`
- Create: `tests/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces (`tests/synth.py`): constants `SR=44100`, `BPM=100`, `BAR_S=2.4`, `DURATION_S=38.4`; `TRUTH: dict` with keys `tempo_bpm`, `key`, `time_signature`, `chords: list[tuple[float, float, str]]` (onset, end, Harte label), `melody: list[tuple[float, int]]` (onset, MIDI pitch); `render(parts: set[str]) -> np.ndarray`; `write(path: Path, parts: set[str] = ALL_PARTS) -> Path`
- Produces (`conftest.py`): session fixture `synth_song` → object with `.mix`, `.harmonic`, `.melody` (Paths) and `.truth`

- [ ] **Step 1: Add the analysis stack to `pyproject.toml`**

Resolve a commit to pin `beat_this` (the spike used `main.zip`, which is a moving target):

```bash
git ls-remote https://github.com/CPJKU/beat_this HEAD
```

Add to `pyproject.toml`, substituting the 40-character hash printed above for `COMMIT`:

```toml
[project.optional-dependencies]
analysis = [
    "numpy==1.26.4",                 # basic-pitch's tensorflow pin forces numpy<2; demucs does not declare numpy at all
    "torch==2.11.0",
    "torchaudio==2.11.0",
    "demucs==4.1.0",
    "librosa==0.11.0",
    "soundfile==0.14.0",
    "pyloudnorm>=0.1.1",
    "pretty-midi==0.2.11.post0",
    "basic-pitch[onnx]==0.4.0",
    "faster-whisper==1.2.1",
    "lv-chordia==1.1.0",
    "beat-this @ https://github.com/CPJKU/beat_this/archive/COMMIT.zip",
]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Also add `"music21==10.5.0"` to the main `dependencies` list (pure Python; used by the MusicXML renderer in Task 24).

- [ ] **Step 2: Install and verify CUDA**

```bash
uv sync --extra analysis
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `2.11.0+cu128 True NVIDIA GeForce RTX 4070 SUPER`. If `False`, the CPU wheel was resolved — check the `[tool.uv.sources]` block. The venv will be ~6 GB.

- [ ] **Step 3: Write the failing test**

`tests/test_synth.py`:
```python
import pytest

sf = pytest.importorskip("soundfile")


def test_synth_matches_its_own_truth(synth_song):
    info = sf.info(str(synth_song.mix))
    assert info.samplerate == 44100 and abs(info.duration - 38.4) < 0.01
    t = synth_song.truth
    assert (t["tempo_bpm"], t["key"], t["time_signature"]) == (100, "C major", "4/4")
    assert [c[2] for c in t["chords"][:4]] == ["C:maj", "A:min", "F:maj", "G:maj"]
    assert t["chords"][1][0] == pytest.approx(4.8)                 # two bars per chord
    assert len(t["melody"]) == 64 and t["melody"][0] == (0.0, 72)
    assert synth_song.harmonic.exists() and synth_song.melody.exists()


def test_render_is_deterministic():
    import numpy as np
    import synth
    assert np.array_equal(synth.render(synth.ALL_PARTS), synth.render(synth.ALL_PARTS))
```

- [ ] **Step 4: Run it, verify it fails**

Run: `uv run pytest tests/test_synth.py -v`
Expected: `fixture 'synth_song' not found`.

- [ ] **Step 5: Write `tests/synth.py`**

```python
"""Manufactured ground truth (BUILD-SPEC §11). Real songs have no answer key; this one does.

16 bars of 4/4 at 100 BPM in C major: C | C | Am | Am | F | F | G | G, twice.
Parts: saw-wave triads, a bass root, a kick/hat pattern, and a sine melody of chord tones in quarter notes.
"""
from pathlib import Path

import numpy as np

SR = 44100
BPM = 100
BEAT_S = 60 / BPM
BAR_S = 4 * BEAT_S
PROGRESSION = [("C:maj", [60, 64, 67]), ("A:min", [57, 60, 64]), ("F:maj", [53, 57, 60]), ("G:maj", [55, 59, 62])]
BARS_PER_CHORD = 2
REPEATS = 2
N_BARS = len(PROGRESSION) * BARS_PER_CHORD * REPEATS
DURATION_S = N_BARS * BAR_S
ALL_PARTS = frozenset({"harmony", "bass", "drums", "melody"})


def _bar_chords() -> list[tuple[str, list[int]]]:
    return [chord for _ in range(REPEATS) for chord in PROGRESSION for _ in range(BARS_PER_CHORD)]


def _melody_pitches(triad: list[int]) -> list[int]:
    root, third, fifth = (p + 12 for p in triad)
    return [root, third, fifth, third]


TRUTH = {
    "tempo_bpm": BPM, "key": "C major", "time_signature": "4/4",
    "chords": [(i * BARS_PER_CHORD * BAR_S, (i + 1) * BARS_PER_CHORD * BAR_S, PROGRESSION[i % 4][0])
               for i in range(len(PROGRESSION) * REPEATS)],
    "melody": [(round(bar * BAR_S + beat * BEAT_S, 6), pitch)
               for bar, (_, triad) in enumerate(_bar_chords())
               for beat, pitch in enumerate(_melody_pitches(triad))],
}


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _saw(freq: float, t: np.ndarray, harmonics: int = 10) -> np.ndarray:
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))


def _place(out: np.ndarray, start_s: float, sig: np.ndarray) -> None:
    a = int(round(start_s * SR))
    b = min(len(out), a + len(sig))
    out[a:b] += sig[: b - a]


def render(parts=ALL_PARTS) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(int(round(DURATION_S * SR)))
    bar_t = np.arange(int(BAR_S * SR)) / SR
    beat_t = np.arange(int(BEAT_S * SR)) / SR
    for bar, (_, triad) in enumerate(_bar_chords()):
        start = bar * BAR_S
        if "harmony" in parts:
            env = np.minimum(1, bar_t / 0.02) * np.exp(-bar_t * 0.5)
            _place(out, start, 0.25 * env * sum(_saw(_hz(p), bar_t) for p in triad))
        if "bass" in parts:
            env = np.minimum(1, bar_t / 0.01) * np.exp(-bar_t * 0.8)
            _place(out, start, 0.5 * env * _saw(_hz(triad[0] - 24), bar_t, 4))
        for beat in range(4):
            at = start + beat * BEAT_S
            if "drums" in parts:
                kick = np.sin(2 * np.pi * 55 * beat_t) * np.exp(-beat_t * 30)
                _place(out, at, (0.9 if beat == 0 else 0.5) * kick)
                for eighth in (0, 0.5):
                    hat = rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
                    _place(out, at + eighth * BEAT_S, 0.08 * hat)
            if "melody" in parts:
                env = np.minimum(1, beat_t / 0.015) * np.minimum(1, (BEAT_S - beat_t) / 0.05)
                _place(out, at, 0.35 * env * np.sin(2 * np.pi * _hz(_melody_pitches(triad)[beat]) * beat_t))
    return (0.7 * out / np.abs(out).max()).astype(np.float32)


def write(path: Path, parts=ALL_PARTS) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = render(parts)
    sf.write(str(path), np.stack([mono, mono], axis=1), SR, subtype="PCM_16")
    return path
```

- [ ] **Step 6: Add the session fixture — append to `tests/conftest.py`**

```python
from types import SimpleNamespace


@pytest.fixture(scope="session")
def synth_song(tmp_path_factory):
    pytest.importorskip("soundfile")
    import synth
    d = tmp_path_factory.mktemp("synth")
    return SimpleNamespace(
        mix=synth.write(d / "mix.wav"),
        harmonic=synth.write(d / "harmonic.wav", {"harmony", "bass"}),
        melody=synth.write(d / "melody.wav", {"melody"}),
        truth=synth.TRUTH)
```

- [ ] **Step 7: Run tests, verify they pass; commit**

Run: `uv run pytest tests/test_synth.py -v` → PASS.
```bash
git add pyproject.toml uv.lock tests/synth.py tests/conftest.py tests/test_synth.py
git commit -m "test: analysis stack pins and synthetic ground-truth fixture"
```

---

### Task 14: Stem separation (Demucs) — runs first

**Files:**
- Create: `src/songcomposer/analysis/stems.py`
- Test: `tests/test_stems.py`

**Interfaces:**
- Produces: `stems.MODEL = "htdemucs"`, `stems.Stems(vocals: Path, drums: Path, bass: Path, other: Path, harmonic: Path)` (pydantic), `stems.separate(audio: Path, cache_dir: Path) -> Stems`. `harmonic` = bass + other summed — what the chord recogniser and key estimator listen to, with drums and vocals out of the way.

- [ ] **Step 1: Write the failing tests**

`tests/test_stems.py`:
```python
import subprocess

import pytest

sf = pytest.importorskip("soundfile")
from songcomposer.analysis import stems  # noqa: E402


def test_separate_is_cached_and_mixes_harmonic(tmp_path, synth_song, monkeypatch):
    """No GPU: fake the demucs subprocess by writing four stems where demucs would."""
    import numpy as np
    runs = []

    def fake_run(args, **kw):
        runs.append(args)
        assert args[1:3] == ["-m", "demucs"] and "-n" in args and str(synth_song.mix) == args[-1]
        out = tmp_path_out(args) / stems.MODEL / synth_song.mix.stem
        out.mkdir(parents=True)
        for name, level in (("vocals", 0.0), ("drums", 0.1), ("bass", 0.2), ("other", 0.3)):
            sf.write(str(out / f"{name}.wav"), np.full((1000, 2), level, dtype="float32"), 44100)
        return subprocess.CompletedProcess(args, 0, "", "")

    def tmp_path_out(args):
        from pathlib import Path
        return Path(args[args.index("-o") + 1])

    monkeypatch.setattr(stems.subprocess, "run", fake_run)
    got = stems.separate(synth_song.mix, tmp_path)
    data, sr = sf.read(str(got.harmonic))
    assert sr == 44100 and data[0, 0] == pytest.approx(0.5, abs=1e-3)       # bass 0.2 + other 0.3
    assert all(p.exists() for p in (got.vocals, got.drums, got.bass, got.other))
    stems.separate(synth_song.mix, tmp_path)
    assert len(runs) == 1                                                    # second call: cache hit


@pytest.mark.gpu
def test_real_demucs_separates_the_synth(tmp_path, synth_song):
    got = stems.separate(synth_song.mix, tmp_path)
    assert abs(sf.info(str(got.harmonic)).duration - 38.4) < 0.1
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_stems.py -m "not gpu" -v` → `ImportError: cannot import name 'stems'`.

- [ ] **Step 3: Write `src/songcomposer/analysis/stems.py`**

```python
"""Demucs stem separation — run FIRST. Chord detection on a full mix is fighting percussion;
splitting out drums and vocals is the single biggest accuracy gain available (BUILD-SPEC §4)."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..hashing import content_key, file_sha1

MODEL = "htdemucs"
VERSION = "1"
NAMES = ("vocals", "drums", "bass", "other")


class Stems(BaseModel):
    vocals: Path
    drums: Path
    bass: Path
    other: Path
    harmonic: Path


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def separate(audio: Path, cache_dir: Path) -> Stems:
    audio = Path(audio).resolve()
    out = Path(cache_dir) / f"stems-{content_key(file_sha1(audio), MODEL, VERSION)}"
    result = Stems(**{n: out / f"{n}.wav" for n in NAMES}, harmonic=out / "harmonic.wav")
    if all(p.exists() for p in result.model_dump().values()):
        return result
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-m", "demucs", "-n", MODEL, "-d", _device(), "-o", tmp, str(audio)],
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"demucs failed: {proc.stderr[-800:]}")
        produced = Path(tmp) / MODEL / audio.stem
        for n in NAMES:
            shutil.move(str(produced / f"{n}.wav"), str(out / f"{n}.wav"))
    import soundfile as sf
    bass, sr = sf.read(str(result.bass))
    other, _ = sf.read(str(result.other))
    n = min(len(bass), len(other))
    sf.write(str(result.harmonic), bass[:n] + other[:n], sr, subtype="PCM_16")
    return result
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_stems.py -v` (includes the GPU test; first run downloads the htdemucs weights, ~80 MB). Expected: PASS.
  If the real run fails inside `torchaudio.save` with a message about `torchcodec`, run `uv add --optional analysis torchcodec` and re-run — newer torchaudio delegates file writing to it.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/analysis/stems.py tests/test_stems.py pyproject.toml uv.lock
git commit -m "feat: demucs stem separation with content-hash cache and harmonic mixdown"
```

---

### Task 15: Beats, downbeats, tempo, time signature

**Files:**
- Create: `src/songcomposer/analysis/beats.py`
- Test: `tests/test_beats.py`

**Interfaces:**
- Produces: `beats.BeatGrid(beats: list[float], downbeats: list[float], tempo_bpm: float, time_signature: str)`, `beats.summarise_beats(beats: list[float], downbeats: list[float]) -> BeatGrid`, `beats.track_beats(audio: Path, cache_dir: Path) -> BeatGrid`

- [ ] **Step 1: Write the failing tests**

`tests/test_beats.py`:
```python
import pytest

from songcomposer.analysis.beats import summarise_beats, track_beats


def test_tempo_and_meter_from_a_clean_grid():
    beats = [i * 0.6 for i in range(32)]
    g = summarise_beats(beats, beats[::4])
    assert g.tempo_bpm == pytest.approx(100.0) and g.time_signature == "4/4"


def test_waltz():
    beats = [i * 0.5 for i in range(30)]
    assert summarise_beats(beats, beats[::3]).time_signature == "3/4"


def test_median_ignores_a_dropped_beat():
    beats = [i * 0.6 for i in range(32)]
    del beats[10]
    assert summarise_beats(beats, beats[::4]).tempo_bpm == pytest.approx(100.0)


def test_no_beats_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no beat"):
        summarise_beats([0.5], [])


@pytest.mark.gpu
def test_beat_this_recovers_the_synth_tempo(tmp_path, synth_song):
    g = track_beats(synth_song.mix, tmp_path)
    assert abs(g.tempo_bpm - synth_song.truth["tempo_bpm"]) < 2.0
    assert g.time_signature == "4/4"
    assert abs(g.downbeats[1] - g.downbeats[0] - 2.4) < 0.1
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_beats.py -m "not gpu" -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/songcomposer/analysis/beats.py`**

```python
"""Beat and downbeat tracking with beat_this (CPJKU, 2024). dbn=False: the DBN post-processor needs madmom,
which cannot be built on this machine."""
from collections import Counter
from pathlib import Path
from statistics import median

from pydantic import BaseModel

from ..hashing import file_sha1
from .cache import cached

VERSION = "1"


class BeatGrid(BaseModel):
    beats: list[float]
    downbeats: list[float]
    tempo_bpm: float
    time_signature: str


def summarise_beats(beats: list[float], downbeats: list[float]) -> BeatGrid:
    if len(beats) < 2:
        raise ValueError("no beat found — cannot derive tempo")
    tempo = 60.0 / median(b - a for a, b in zip(beats, beats[1:]))
    counts = [sum(1 for b in beats if lo - 0.02 <= b < hi - 0.02) for lo, hi in zip(downbeats, downbeats[1:])]
    per_bar = Counter(counts).most_common(1)[0][0] if counts else 4
    return BeatGrid(beats=[round(b, 4) for b in beats], downbeats=[round(d, 4) for d in downbeats],
                    tempo_bpm=round(tempo, 2), time_signature=f"{per_bar}/4")


def track_beats(audio: Path, cache_dir: Path) -> BeatGrid:
    def work() -> dict:
        import torch
        from beat_this.inference import File2Beats
        device = "cuda" if torch.cuda.is_available() else "cpu"
        beats, downbeats = File2Beats(checkpoint_path="final0", device=device, dbn=False)(str(Path(audio).resolve()))
        return summarise_beats([float(b) for b in beats], [float(d) for d in downbeats]).model_dump()

    return BeatGrid(**cached(cache_dir, file_sha1(audio), "beats", VERSION, work))
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_beats.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/analysis/beats.py tests/test_beats.py
git commit -m "feat: beat/downbeat tracking, tempo and time signature"
```

---

### Task 16: Chord recognition — with a confidence we compute ourselves

`lv-chordia` transcribes chords well but reports **no confidence**, and "confidence travels with the data" is a hard rule. We derive confidence from two independent pieces of evidence, neither of which is the recogniser's own opinion:

1. **Spectral support (60%)** — cosine similarity between the segment's measured mean chroma and the pitch-class template of the chord that was claimed. *This is librosa doing what it is good at: measuring, not naming.* It never proposes a chord; it only checks the one proposed.
2. **Cross-pass agreement (40%)** — lv-chordia runs twice, on the harmonic stem and on the full mix. The fraction of the segment's duration where both passes give the same root and quality.

**Files:**
- Create: `src/songcomposer/analysis/chords.py`
- Modify: `src/songcomposer/chordsym.py` (add `chord_pitch_classes`)
- Test: `tests/test_chords.py`

**Interfaces:**
- Produces:
  - `chordsym.chord_pitch_classes(parsed: ParsedChord) -> set[int]`
  - `chords.spectral_support(chroma_mean: Sequence[float], pitch_classes: set[int]) -> float` (0–1)
  - `chords.agreement(onset: float, end: float, root: str, quality: str, other_pass: list[dict]) -> float` (0–1)
  - `chords.score(support: float, agree: float) -> float`
  - `chords.recognise(audio: Path, cache_dir: Path) -> list[dict]` — raw `{"start_time","end_time","chord"}` from lv-chordia, cached
  - `chords.detect_chords(harmonic: Path, mix: Path, cache_dir: Path) -> list[Chord]`

- [ ] **Step 1: Write the failing tests**

`tests/test_chords.py`:
```python
import pytest

from songcomposer.chordsym import chord_pitch_classes, parse_harte

np = pytest.importorskip("numpy")
from songcomposer.analysis import chords  # noqa: E402


@pytest.mark.parametrize("label,pcs", [
    ("C:maj", {0, 4, 7}), ("A:min", {9, 0, 4}), ("C:maj7", {0, 4, 7, 11}), ("A:min7", {9, 0, 4, 7}),
    ("G:7", {7, 11, 2, 5}), ("D:sus4", {2, 7, 9}), ("D:sus2", {2, 4, 9}), ("B:hdim7", {11, 2, 5, 9}),
    ("C:maj(9)", {0, 4, 7, 2}), ("C:weird", {0, 4, 7}),
])
def test_chord_pitch_classes(label, pcs):
    assert chord_pitch_classes(parse_harte(label)) == pcs


def chroma(*pcs, floor=0.05):
    v = np.full(12, floor)
    v[list(pcs)] = 1.0
    return v


def test_support_is_high_when_the_spectrum_matches_and_low_when_it_does_not():
    c_major = chroma(0, 4, 7)
    assert chords.spectral_support(c_major, {0, 4, 7}) > 0.9
    assert chords.spectral_support(c_major, {6, 10, 1}) < 0.2          # F# major claimed over C major audio


def test_support_separates_cmaj7_from_am():
    """The exact confusion CLAUDE.md warns about: the evidence must prefer the right one."""
    heard = chroma(0, 4, 7, 11)                                        # C E G B
    assert chords.spectral_support(heard, {0, 4, 7, 11}) > chords.spectral_support(heard, {9, 0, 4})


def test_support_of_silence_is_zero():
    assert chords.spectral_support(np.zeros(12), {0, 4, 7}) == 0.0


def test_agreement_is_the_overlap_fraction_with_same_root_and_quality():
    other = [{"start_time": 0.0, "end_time": 3.0, "chord": "C:maj"}, {"start_time": 3.0, "end_time": 8.0, "chord": "A:min"}]
    assert chords.agreement(0.0, 4.0, "C", "maj", other) == pytest.approx(0.75)
    assert chords.agreement(0.0, 4.0, "G", "maj", other) == 0.0
    assert chords.agreement(0.0, 4.0, "C", "maj", []) == 0.0


def test_score_bounds_and_weights():
    assert chords.score(1.0, 1.0) == 1.0 and chords.score(0.0, 0.0) == 0.0
    assert chords.score(0.8, 0.0) < 0.6 < chords.score(0.8, 1.0)
    assert chords.score(0.4, 1.0) == pytest.approx(0.4)               # support at the floor contributes nothing


@pytest.mark.gpu
def test_lv_chordia_recovers_the_synth_progression_with_confidence(tmp_path, synth_song):
    got = chords.detect_chords(synth_song.harmonic, synth_song.mix, tmp_path)
    truth = synth_song.truth["chords"]

    def label_at(t):
        return next((c.harte for c in got if c.onset <= t < c.onset + c.duration), None)

    probes = [(a + b) / 2 for a, b, _ in truth]
    assert [label_at(t) for t in probes] == [lab for _, _, lab in truth]
    assert all(c.confidence >= 0.5 for c in got), [(c.symbol, c.confidence) for c in got]
    assert all(0.0 <= c.confidence <= 1.0 for c in got)
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_chords.py -m "not gpu" -v` → `ImportError: cannot import name 'chord_pitch_classes'`.

- [ ] **Step 3: Add `chord_pitch_classes` — append to `src/songcomposer/chordsym.py`**

```python
_QUALITY_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8), "5": (0, 7), "1": (0,),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10), "6": (0, 4, 7, 9), "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 2), "maj9": (0, 4, 7, 11, 2), "min9": (0, 3, 7, 10, 2),
    "11": (0, 4, 7, 10, 2, 5), "min11": (0, 3, 7, 10, 2, 5),
    "13": (0, 4, 7, 10, 2, 9), "maj13": (0, 4, 7, 11, 2, 9), "min13": (0, 3, 7, 10, 2, 9),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
}


def chord_pitch_classes(parsed: ParsedChord) -> set[int]:
    """Pitch classes sounding in the chord. Unknown qualities fall back to a major triad on the root."""
    root = pitch_class(parsed.root)
    intervals = set(_QUALITY_INTERVALS.get(parsed.quality, (0, 4, 7)))
    for ext in parsed.extensions:
        if ext.startswith("*"):
            intervals.discard(_INTERVAL.get(ext[1:], -1))
        elif ext in _INTERVAL:
            intervals.add(_INTERVAL[ext])
    return {(root + i) % 12 for i in intervals}
```

- [ ] **Step 4: Write `src/songcomposer/analysis/chords.py`**

```python
"""Chord recognition: lv-chordia names the chords; we measure how far the spectrum backs each one up.

librosa appears here ONLY as a measuring instrument for confidence. It never names a chord.
"""
from pathlib import Path
from typing import Sequence

from ..chordsym import chord_pitch_classes, parse_harte
from ..hashing import file_sha1
from ..models import Chord
from .cache import cached

VERSION = "1"
SUPPORT_FLOOR, SUPPORT_CEIL = 0.4, 0.9       # cosine similarity range mapped onto 0..1
W_SUPPORT, W_AGREE = 0.6, 0.4


def spectral_support(chroma_mean: Sequence[float], pitch_classes: set[int]) -> float:
    import numpy as np
    v = np.asarray(chroma_mean, dtype=float)
    template = np.zeros(12)
    template[list(pitch_classes)] = 1.0
    denom = np.linalg.norm(v) * np.linalg.norm(template)
    return float(v @ template / denom) if denom > 0 else 0.0


def agreement(onset: float, end: float, root: str, quality: str, other_pass: list[dict]) -> float:
    if end <= onset:
        return 0.0
    shared = 0.0
    for seg in other_pass:
        parsed = parse_harte(seg["chord"])
        if parsed and (parsed.root, parsed.quality) == (root, quality):
            shared += max(0.0, min(end, seg["end_time"]) - max(onset, seg["start_time"]))
    return min(1.0, shared / (end - onset))


def score(support: float, agree: float) -> float:
    scaled = min(1.0, max(0.0, (support - SUPPORT_FLOOR) / (SUPPORT_CEIL - SUPPORT_FLOOR)))
    return round(W_SUPPORT * scaled + W_AGREE * agree, 3)


def recognise(audio: Path, cache_dir: Path) -> list[dict]:
    def work() -> list[dict]:
        from lv_chordia.chord_recognition import chord_recognition
        # ABSOLUTE path: lv-chordia resolves relative paths against its own package directory.
        return chord_recognition(str(Path(audio).resolve()), chord_dict_name="submission")

    return cached(cache_dir, file_sha1(audio), "lvchordia", VERSION, work)


def detect_chords(harmonic: Path, mix: Path, cache_dir: Path) -> list[Chord]:
    import librosa
    import numpy as np
    primary, second = recognise(harmonic, cache_dir), recognise(mix, cache_dir)
    y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    out: list[Chord] = []
    for seg in primary:
        parsed = parse_harte(seg["chord"])
        if parsed is None:                                   # "N" — no chord
            continue
        onset, end = float(seg["start_time"]), float(seg["end_time"])
        frames = (times >= onset + 0.05) & (times < end - 0.05)
        mean = chroma[:, frames].mean(axis=1) if frames.any() else np.zeros(12)
        conf = score(spectral_support(mean, chord_pitch_classes(parsed)),
                     agreement(onset, end, parsed.root, parsed.quality, second))
        out.append(Chord(symbol=parsed.symbol, harte=seg["chord"], root=parsed.root, quality=parsed.quality,
                         extensions=parsed.extensions, bass=parsed.bass, onset=round(onset, 3),
                         duration=round(end - onset, 3), confidence=conf))
    return out
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_chords.py -v` → PASS (GPU test included; lv-chordia runs a 5-model ensemble, expect ~20 s).
  If the GPU test's confidence assertion fails while the labels are right, print `(symbol, support, agreement)` per chord and tune `SUPPORT_FLOOR`/`SUPPORT_CEIL` — they are the only free parameters. Do not weaken the label assertion.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/chordsym.py src/songcomposer/analysis/chords.py tests/test_chords.py
git commit -m "feat: chord recognition (lv-chordia) with evidence-based confidence"
```

---

### Task 17: Key and loudness

**Files:**
- Create: `src/songcomposer/analysis/key.py`
- Test: `tests/test_key.py`

**Interfaces:**
- Produces: `key.estimate_key(chroma_mean: Sequence[float]) -> tuple[str, float]` (e.g. `("A minor", 0.74)`), `key.measure(harmonic: Path, mix: Path, cache_dir: Path) -> dict` with keys `key`, `key_confidence`, `loudness_lufs`, `duration_s`

- [ ] **Step 1: Write the failing tests**

`tests/test_key.py`:
```python
import pytest

np = pytest.importorskip("numpy")
from songcomposer.analysis import key  # noqa: E402


def test_major_and_minor_profiles_are_recovered():
    assert key.estimate_key(key.MAJOR)[0] == "C major"
    assert key.estimate_key(np.roll(key.MINOR, 9))[0] == "A minor"
    assert key.estimate_key(np.roll(key.MAJOR, 10))[0] == "Bb major"


def test_flat_chroma_is_unknown_not_a_guess():
    assert key.estimate_key(np.ones(12)) == ("unknown", 0.0)
    assert 0.5 < key.estimate_key(key.MAJOR)[1] <= 1.0


def test_measure_on_the_synth(tmp_path, synth_song):
    pytest.importorskip("librosa")
    pytest.importorskip("pyloudnorm")
    got = key.measure(synth_song.harmonic, synth_song.mix, tmp_path)
    assert got["key"] == "C major"
    assert abs(got["duration_s"] - 38.4) < 0.05
    assert -40 < got["loudness_lufs"] < 0
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_key.py -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/analysis/key.py`**

```python
"""Key (Krumhansl–Schmuckler over the harmonic stem) and integrated loudness. This is librosa's proper job."""
from pathlib import Path
from typing import Sequence

from ..hashing import file_sha1
from .cache import cached

VERSION = "1"
MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]


def estimate_key(chroma_mean: Sequence[float]) -> tuple[str, float]:
    import numpy as np
    v = np.asarray(chroma_mean, dtype=float)
    scored = []
    for tonic in range(12):
        scored.append((float(np.corrcoef(v, np.roll(MAJOR, tonic))[0, 1]), f"{MAJOR_NAMES[tonic]} major"))
        scored.append((float(np.corrcoef(v, np.roll(MINOR, tonic))[0, 1]), f"{MINOR_NAMES[tonic]} minor"))
    scored.sort(reverse=True)
    (best, name), (runner_up, _) = scored[0], scored[1]
    if np.isnan(best):
        return "unknown", 0.0
    confidence = max(0.0, min(1.0, best)) * max(0.2, min(1.0, (best - runner_up) / 0.15))
    return name, round(confidence, 3)


def measure(harmonic: Path, mix: Path, cache_dir: Path) -> dict:
    def work() -> dict:
        import librosa
        import pyloudnorm
        import soundfile as sf
        y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
        name, conf = estimate_key(librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1))
        data, rate = sf.read(str(mix))
        lufs = float(pyloudnorm.Meter(rate).integrated_loudness(data))
        return {"key": name, "key_confidence": conf, "loudness_lufs": round(lufs, 2),
                "duration_s": round(len(data) / rate, 3)}

    return cached(cache_dir, file_sha1(mix), "key", VERSION, work)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_key.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/analysis/key.py tests/test_key.py
git commit -m "feat: key estimation and integrated loudness"
```

---

### Task 18: Note transcription (basic-pitch)

**Files:**
- Create: `src/songcomposer/analysis/notes.py`
- Test: `tests/test_notes.py`

**Interfaces:**
- Consumes: `Stems`, `Note`
- Produces: `notes.to_monophonic(notes: list[Note]) -> list[Note]`, `notes.transcribe_stem(path: Path, stem: str, cache_dir: Path) -> list[Note]`, `notes.transcribe_notes(stems: Stems, cache_dir: Path) -> list[Note]` — vocals and bass are forced monophonic; `other` stays polyphonic.

- [ ] **Step 1: Write the failing tests**

`tests/test_notes.py`:
```python
import pytest

from songcomposer.analysis import notes
from songcomposer.models import Note


def n(pitch, onset, dur, conf):
    return Note(pitch=pitch, onset=onset, duration=dur, confidence=conf, stem="vocals")


def test_monophonic_keeps_the_more_confident_of_two_overlapping_notes():
    got = notes.to_monophonic([n(60, 0.0, 1.0, 0.9), n(72, 0.1, 0.9, 0.3), n(62, 1.0, 0.5, 0.8)])
    assert [(x.pitch, x.onset) for x in got] == [(60, 0.0), (62, 1.0)]


def test_monophonic_trims_a_tail_that_runs_into_the_next_note():
    got = notes.to_monophonic([n(60, 0.0, 1.0, 0.9), n(62, 0.8, 0.6, 0.9)])
    assert got[0].duration == pytest.approx(0.8) and got[1].onset == 0.8


def test_basic_pitch_recovers_the_synth_melody(tmp_path, synth_song):
    pytest.importorskip("basic_pitch")
    got = notes.to_monophonic(notes.transcribe_stem(synth_song.melody, "vocals", tmp_path))
    truth = synth_song.truth["melody"]
    hits = sum(1 for onset, pitch in truth
               if any(abs(g.onset - onset) < 0.08 and g.pitch == pitch for g in got))
    assert hits / len(truth) >= 0.85, f"{hits}/{len(truth)} melody notes recovered"
    assert all(0.0 <= g.confidence <= 1.0 for g in got)
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_notes.py -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/analysis/notes.py`**

```python
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
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_notes.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/analysis/notes.py tests/test_notes.py
git commit -m "feat: note transcription per stem with basic-pitch"
```

---

### Task 19: Lyrics ear (faster-whisper on the vocal stem)

**Files:**
- Create: `src/songcomposer/analysis/lyrics.py`
- Test: `tests/test_lyrics_ear.py`

**Interfaces:**
- Produces: `lyrics.words_from_segments(segments: Iterable) -> list[Word]`, `lyrics.transcribe_words(vocals: Path, cache_dir: Path, model_name: str, hint: str = "") -> list[Word]`. `hint` is our own known lyrics when transcribing our own take — it biases Whisper toward the words we know were sung. It is empty for the reference.

- [ ] **Step 1: Write the failing tests**

`tests/test_lyrics_ear.py`:
```python
from types import SimpleNamespace as NS

import pytest

from songcomposer.analysis import lyrics


def test_words_from_segments_strips_and_carries_probability():
    segs = [NS(words=[NS(word=" Glass", start=1.0, end=1.4, probability=0.93), NS(word=" hour,", start=1.4, end=2.0, probability=0.41)]),
            NS(words=None), NS(words=[NS(word="  ", start=3, end=3.1, probability=0.9)])]
    got = lyrics.words_from_segments(segs)
    assert [(w.word, w.start, w.end, w.confidence) for w in got] == [("Glass", 1.0, 1.4, 0.93), ("hour,", 1.4, 2.0, 0.41)]


def test_hint_changes_the_cache_key(tmp_path, sine_wav, monkeypatch):
    calls = []
    monkeypatch.setattr(lyrics, "_run_whisper", lambda path, model, hint: calls.append(hint) or [])
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3")
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3")
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3", hint="glass hour hold me still")
    assert calls == ["", "glass hour hold me still"]


@pytest.mark.gpu
def test_whisper_loads_on_cuda_and_returns_a_list(tmp_path, sine_wav):
    assert isinstance(lyrics.transcribe_words(sine_wav, tmp_path, "tiny"), list)
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_lyrics_ear.py -m "not gpu" -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/analysis/lyrics.py`**

```python
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
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_lyrics_ear.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/analysis/lyrics.py tests/test_lyrics_ear.py
git commit -m "feat: lyrics ear — word-timed whisper transcription of the vocal stem"
```

---

### Task 20: Structure, fusion, and the complete `analyze()`

Sections fuse both ears the way the project intends: **boundaries are measured** (DSP, bar-aligned), **labels are heard** (the subjective ear's reading of the form). A label's confidence is how much of the measured segment the heard section covers.

**Files:**
- Create: `src/songcomposer/analysis/structure.py`, `src/songcomposer/analysis/fuse.py`
- Modify: `src/songcomposer/analysis/__init__.py` (replace `analyze` body)
- Test: `tests/test_structure_fuse.py`, `tests/test_analyze_full.py`

**Interfaces:**
- Produces:
  - `structure.boundaries(harmonic: Path, downbeats: list[float], duration_s: float) -> list[float]` (sorted, starts at 0.0, ends at `duration_s`)
  - `structure.label_sections(bounds: list[float], guesses: list[SectionGuess]) -> list[Section]`
  - `fuse.snap_to_beats(chords: list[Chord], beats: list[float], tolerance: float = 0.12) -> list[Chord]`
  - `fuse.merge_adjacent(chords: list[Chord]) -> list[Chord]` (same `harte` back-to-back → one chord, duration-weighted confidence)
  - `analysis.objective_installed() -> bool`
  - `analysis.analyze(...)` — final body, same signature as Task 5

- [ ] **Step 1: Write the failing tests**

`tests/test_structure_fuse.py`:
```python
import pytest

from songcomposer.analysis import fuse, structure
from songcomposer.models import Chord, SectionGuess


def ch(harte, onset, dur, conf=0.8):
    return Chord(symbol=harte.split(":")[0], harte=harte, root=harte.split(":")[0], quality="maj", extensions=[],
                 bass=None, onset=onset, duration=dur, confidence=conf)


def test_snap_moves_near_onsets_onto_the_beat_and_keeps_far_ones():
    got = fuse.snap_to_beats([ch("C:maj", 0.05, 2.35), ch("F:maj", 2.4, 1.0), ch("G:maj", 3.7, 1.1)], [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2])
    assert [c.onset for c in got] == [0.0, 2.4, 3.6]
    assert got[0].duration == pytest.approx(2.4)                     # end re-joined to the next onset


def test_snap_leaves_a_chord_between_beats_alone():
    assert fuse.snap_to_beats([ch("C:maj", 0.3, 1.0)], [0.0, 0.6, 1.2])[0].onset == 0.3


def test_merge_adjacent_weights_confidence_by_duration():
    got = fuse.merge_adjacent([ch("C:maj", 0, 3, 0.9), ch("C:maj", 3, 1, 0.5), ch("G:maj", 4, 2, 0.7)])
    assert [(c.harte, c.onset, c.duration) for c in got] == [("C:maj", 0, 4), ("G:maj", 4, 2)]
    assert got[0].confidence == pytest.approx(0.8)


def test_labels_come_from_the_heard_form_with_overlap_as_confidence():
    guesses = [SectionGuess(label="verse", start=0, end=20), SectionGuess(label="chorus", start=20, end=40)]
    got = structure.label_sections([0.0, 9.6, 19.2, 38.4], guesses)
    assert [(s.label, s.start, s.end) for s in got] == [("verse", 0.0, 19.2), ("chorus", 19.2, 38.4)]   # two verse bars merged
    assert got[0].confidence == pytest.approx(1.0) and 0.9 < got[1].confidence <= 1.0


def test_without_a_subjective_ear_sections_are_honestly_anonymous():
    got = structure.label_sections([0.0, 10.0, 20.0], [])
    assert [s.label for s in got] == ["part 1", "part 2"] and all(s.confidence == 0.0 for s in got)


def test_boundaries_are_bar_aligned_on_the_synth(synth_song):
    pytest.importorskip("librosa")
    downbeats = [i * 2.4 for i in range(16)]
    got = structure.boundaries(synth_song.harmonic, downbeats, 38.4)
    assert got[0] == 0.0 and got[-1] == 38.4 and got == sorted(got)
    assert all(min(abs(b - d) for d in downbeats + [38.4]) < 0.1 for b in got)


def test_too_few_downbeats_gives_one_segment(synth_song):
    assert structure.boundaries(synth_song.harmonic, [0.0], 38.4) == [0.0, 38.4]
```

`tests/test_analyze_full.py`:
```python
from pathlib import Path

import pytest

from songcomposer import analysis
from songcomposer.analysis import beats, chords, key, lyrics, notes, stems, structure, subjective
from songcomposer.config import Config
from songcomposer.models import Chord, Note, Subjective, Word

HEARD = Subjective(genre_tags=[], instrumentation=[], timbre="", vocal_character="", vocal_gender="none",
                   production="", emotional_arc="", arrangement_density=[], sections=[])


@pytest.fixture
def patched(monkeypatch, tmp_path):
    seen = {}
    fake = stems.Stems(vocals=Path("v.wav"), drums=Path("d.wav"), bass=Path("b.wav"), other=Path("o.wav"),
                       harmonic=Path("h.wav"))
    monkeypatch.setattr(analysis, "objective_installed", lambda: True)
    monkeypatch.setattr(subjective, "listen", lambda a, c, m: HEARD)
    monkeypatch.setattr(stems, "separate", lambda a, c: fake)
    monkeypatch.setattr(beats, "track_beats", lambda a, c: beats.summarise_beats([i * 0.6 for i in range(16)], [0.0, 2.4, 4.8, 7.2]))
    monkeypatch.setattr(chords, "detect_chords", lambda h, m, c: [
        Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None, onset=0.04, duration=2.36, confidence=0.9)])
    monkeypatch.setattr(key, "measure", lambda h, m, c: {"key": "C major", "key_confidence": 0.8, "loudness_lufs": -14.0, "duration_s": 9.6})
    monkeypatch.setattr(notes, "transcribe_notes", lambda s, c: [Note(pitch=60, onset=0, duration=1, confidence=0.7, stem="vocals")])
    monkeypatch.setattr(lyrics, "transcribe_words", lambda v, c, m, hint="": seen.setdefault("hint", hint) and [] or [Word(word="hi", start=0, end=1, confidence=0.9)])
    monkeypatch.setattr(structure, "boundaries", lambda h, d, dur: [0.0, dur])
    return seen


def test_full_analysis_fuses_every_ear(patched, tmp_path, sine_wav):
    a = analysis.analyze(sine_wav, tmp_path, Config(), lyrics_hint="our words")
    assert set(a.engines) == {"subjective", "stems", "beats", "chords", "key", "notes", "lyrics", "structure"}
    assert a.global_info.key == "C major" and a.global_info.tempo_bpm == 100.0 and a.global_info.time_signature == "4/4"
    assert a.chords[0].onset == 0.0                                  # snapped to the beat grid
    assert a.notes and a.lyrics and a.sections and a.subjective == HEARD
    assert patched["hint"] == "our words"


def test_subjective_only_still_works_and_warns(patched, tmp_path, sine_wav, capsys):
    a = analysis.analyze(sine_wav, tmp_path, Config(), ears={"subjective"})
    assert a.chords == [] and a.global_info is None
    assert "PARTIAL ANALYSIS" in capsys.readouterr().out


@pytest.mark.gpu
def test_end_to_end_on_the_synth_recovers_the_truth(tmp_path, synth_song):
    a = analysis.analyze(synth_song.mix, tmp_path, Config(), ears=set(analysis.OBJECTIVE_EARS) - {"lyrics"})
    t = synth_song.truth
    assert a.global_info.key == t["key"] and abs(a.global_info.tempo_bpm - 100) < 2
    right = sum(min(e, c.onset + c.duration) - max(s, c.onset)
                for s, e, lab in t["chords"] for c in a.chords
                if c.harte == lab and min(e, c.onset + c.duration) > max(s, c.onset))
    assert right / 38.4 >= 0.75, f"only {right / 38.4:.0%} of the timeline has the right chord"
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_structure_fuse.py tests/test_analyze_full.py -m "not gpu" -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/analysis/structure.py`**

```python
"""Song form. Boundaries are MEASURED (bar-synchronous features, agglomerative segmentation);
labels are HEARD (the subjective ear's sections). Confidence = how well the two agree."""
from pathlib import Path

from ..models import Section, SectionGuess

SECONDS_PER_SEGMENT = 25


def boundaries(harmonic: Path, downbeats: list[float], duration_s: float) -> list[float]:
    if len(downbeats) < 4:
        return [0.0, duration_s]
    import librosa
    import numpy as np
    y, sr = librosa.load(str(harmonic), sr=22050, mono=True)
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.util.normalize(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop), axis=1)
    bars = librosa.util.fix_frames(librosa.time_to_frames(downbeats, sr=sr, hop_length=hop), x_min=0, x_max=chroma.shape[1])
    feats = np.vstack([librosa.util.sync(chroma, bars, aggregate=np.median), librosa.util.sync(mfcc, bars)])
    k = int(min(feats.shape[1], max(2, min(12, round(duration_s / SECONDS_PER_SEGMENT)))))
    starts = librosa.segment.agglomerative(feats, k)
    times = librosa.frames_to_time(bars[starts], sr=sr, hop_length=hop)
    inner = sorted({round(float(t), 3) for t in times if 0.5 < t < duration_s - 0.5})
    return [0.0] + inner + [duration_s]


def label_sections(bounds: list[float], guesses: list[SectionGuess]) -> list[Section]:
    out: list[Section] = []
    for i, (start, end) in enumerate(zip(bounds, bounds[1:]), start=1):
        best, cover = None, 0.0
        for g in guesses:
            shared = max(0.0, min(end, g.end) - max(start, g.start))
            if shared > cover:
                best, cover = g, shared
        label = best.label if best else f"part {i}"
        conf = round(cover / (end - start), 3) if best and end > start else 0.0
        if out and best and out[-1].label == label:
            prev = out[-1]
            total = end - prev.start
            merged = (prev.confidence * (prev.end - prev.start) + conf * (end - start)) / total
            out[-1] = Section(label=label, start=prev.start, end=end, confidence=round(merged, 3))
        else:
            out.append(Section(label=label, start=start, end=end, confidence=conf))
    return out
```

- [ ] **Step 4: Write `src/songcomposer/analysis/fuse.py`**

```python
"""Small, pure fusion steps between the raw components and the final Analysis."""
from bisect import bisect_left

from ..models import Chord


def _nearest(beats: list[float], t: float) -> float:
    i = bisect_left(beats, t)
    return min(beats[max(0, i - 1):i + 1], key=lambda b: abs(b - t))


def snap_to_beats(chords: list[Chord], beats: list[float], tolerance: float = 0.12) -> list[Chord]:
    if not beats or not chords:
        return chords
    ends = [c.onset + c.duration for c in chords]
    onsets = [b if abs((b := _nearest(beats, c.onset)) - c.onset) <= tolerance else c.onset for c in chords]
    out = []
    for i, c in enumerate(chords):
        end = onsets[i + 1] if i + 1 < len(chords) and abs(chords[i + 1].onset - ends[i]) < 0.05 else ends[i]
        out.append(c.model_copy(update={"onset": round(onsets[i], 3), "duration": round(max(0.0, end - onsets[i]), 3)}))
    return out


def merge_adjacent(chords: list[Chord]) -> list[Chord]:
    out: list[Chord] = []
    for c in chords:
        if out and out[-1].harte == c.harte and abs(out[-1].onset + out[-1].duration - c.onset) < 0.05:
            p = out[-1]
            total = p.duration + c.duration
            conf = (p.confidence * p.duration + c.confidence * c.duration) / total if total else p.confidence
            out[-1] = p.model_copy(update={"duration": round(total, 3), "confidence": round(conf, 3)})
        else:
            out.append(c)
    return out
```

- [ ] **Step 5: Replace `src/songcomposer/analysis/__init__.py` with the complete engine**

```python
"""THE analysis engine. Runs twice — over the reference on the way in, and over our own
generated take on the way out. Same code path both times (BUILD-SPEC §4)."""
import importlib.util
from pathlib import Path

from ..config import Config
from ..hashing import file_sha1
from ..models import Analysis, GlobalInfo

OBJECTIVE_EARS = ("stems", "beats", "chords", "key", "notes", "lyrics", "structure")
ALL_EARS = ("subjective",) + OBJECTIVE_EARS
NEEDS_STEMS = {"chords", "key", "notes", "lyrics", "structure"}


def objective_installed() -> bool:
    return all(importlib.util.find_spec(m) for m in ("demucs", "beat_this", "lv_chordia", "basic_pitch", "faster_whisper", "librosa"))


def analyze(audio: Path, cache_dir: Path, config: Config, ears: set[str] | None = None,
            lyrics_hint: str = "") -> Analysis:
    from . import beats as beats_mod, chords as chords_mod, fuse, key as key_mod, lyrics as lyrics_mod
    from . import notes as notes_mod, stems as stems_mod, structure as structure_mod, subjective as subjective_mod

    if ears is None:
        ears = set(ALL_EARS) if objective_installed() else {"subjective"}
    ears = set(ears)
    unknown = ears - set(ALL_EARS)
    if unknown:
        raise ValueError(f"unknown ear(s): {sorted(unknown)} — valid: {', '.join(ALL_EARS)}")
    if ears & set(OBJECTIVE_EARS) and not objective_installed():
        raise RuntimeError("objective ear requested but the analysis stack is not installed — run `uv sync --extra analysis`")
    if ears & NEEDS_STEMS:
        ears.add("stems")
    if "structure" in ears:
        ears.add("beats")

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    result = Analysis(audio_sha1=file_sha1(audio), engines={})

    if "subjective" in ears:
        print(f"> subjective ear ({config.listener_model})")
        result.subjective = subjective_mod.listen(audio, cache_dir, config.listener_model)
        result.engines["subjective"] = config.listener_model

    stems = grid = measured = None
    if "stems" in ears:
        print("> stems (demucs)")
        stems = stems_mod.separate(audio, cache_dir)
        result.engines["stems"] = stems_mod.MODEL
    if "beats" in ears:
        print("> beats (beat_this)")
        grid = beats_mod.track_beats(audio, cache_dir)
        result.beats, result.downbeats = grid.beats, grid.downbeats
        result.engines["beats"] = "beat_this"
    if "chords" in ears:
        print("> chords (lv-chordia ×2 + spectral support)")
        found = chords_mod.detect_chords(stems.harmonic, audio, cache_dir)
        result.chords = fuse.merge_adjacent(fuse.snap_to_beats(found, grid.beats if grid else []))
        result.engines["chords"] = "lv-chordia 1.1.0"
    if "key" in ears:
        print("> key + loudness")
        measured = key_mod.measure(stems.harmonic, audio, cache_dir)
        result.engines["key"] = "krumhansl/librosa"
    if "notes" in ears:
        print("> notes (basic-pitch)")
        result.notes = notes_mod.transcribe_notes(stems, cache_dir)
        result.engines["notes"] = "basic-pitch 0.4.0"
    if "lyrics" in ears:
        print(f"> lyrics (whisper {config.whisper_model})")
        result.lyrics = lyrics_mod.transcribe_words(stems.vocals, cache_dir, config.whisper_model, hint=lyrics_hint)
        result.engines["lyrics"] = f"faster-whisper {config.whisper_model}"
    if grid and measured:
        result.global_info = GlobalInfo(key=measured["key"], key_confidence=measured["key_confidence"],
                                        tempo_bpm=grid.tempo_bpm, time_signature=grid.time_signature,
                                        loudness_lufs=measured["loudness_lufs"], duration_s=measured["duration_s"])
    if "structure" in ears:
        print("> structure")
        duration = measured["duration_s"] if measured else (grid.beats[-1] if grid.beats else 0.0)
        guesses = result.subjective.sections if result.subjective else []
        result.sections = structure_mod.label_sections(
            structure_mod.boundaries(stems.harmonic, grid.downbeats, duration), guesses)
        result.engines["structure"] = "librosa agglomerative + heard labels"

    missing = [e for e in OBJECTIVE_EARS if e not in result.engines]
    if missing:
        print(f"! PARTIAL ANALYSIS — objective ear not run ({', '.join(missing)}). "
              "No chords, key or tempo are available for those; nothing downstream may invent them.")
    return result
```

Note for the Task 5 tests: `test_analysis_core.py` patches `subjective.chat_json` and calls `analyze(..., ears={"subjective"})` — both still work unchanged because `analyze` now reaches the ear through `subjective_mod.listen`.

- [ ] **Step 6: Run everything**

Run: `uv run pytest -m "not gpu" -v` → all PASS. Then `uv run pytest -m gpu -v` → all PASS (several minutes on first run: model downloads).

- [ ] **Step 7: Commit**

```bash
git add src/songcomposer/analysis tests/test_structure_fuse.py tests/test_analyze_full.py
git commit -m "feat: complete two-ear analysis engine — structure, fusion, full analyze()"
```

- [ ] **Step 8: MILESTONE B — re-analyse the real reference (human step, free)**

```bash
uv run songcomposer analyze <song>
```
Open `work/<song>/01-analysis.json`. Play the reference with a guitar in hand and check the first verse and chorus of `chords` against what you hear. Note in `docs/spikes/2026-09-20-mir-stack.md` under "Real-music accuracy": how many of the first 20 chords were right, and whether the wrong ones had confidence below 0.5. **If wrong chords routinely carry high confidence, stop and retune Task 16's constants before building renderers on top.**

---

## Part 3 — transcribe our own take and chart it (Tasks 21–27)

Everything here consumes `Transcription` (the engine's second run). Everything guitar-specific lives in `src/songcomposer/render/` and nowhere else.

---

### Task 21: Transcribe the chosen take and align our lyric lines

We know exactly which words were *supposed* to be sung (the spec). Whisper tells us which words *were* sung and when. Aligning the two gives word-timed lyrics in our own spelling, and honest gaps where the singer departed from the text.

**Files:**
- Create: `src/songcomposer/transcribe.py`
- Test: `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `analysis.analyze`, `audio.to_wav`, `SongSpec`, `Word`, `LyricLine`, `Transcription`, `Section`, `Chosen`
- Produces:
  - `transcribe.align_lines(spec: SongSpec, heard: list[Word]) -> list[LyricLine]` — every `LyricLine.words` entry is a **spec** word (our spelling) with timing; matched words carry Whisper's confidence, interpolated words carry `confidence=0.0`; a line with no matches has `start=None, end=None, words=[]`
  - `transcribe.sections_from_lines(lines: list[LyricLine], duration_s: float) -> list[Section]`
  - `transcribe.run_transcribe(song: str, force: bool = False) -> Transcription`

- [ ] **Step 1: Write the failing tests**

`tests/test_transcribe.py`:
```python
import pytest

from songcomposer import transcribe
from songcomposer.models import SongSpec, SpecSection, Word

SPEC = SongSpec(title="T", style_prompt="s", target_duration_s=60, sections=[
    SpecSection(name="Verse 1", duration_s=30, lines=["Glass hour, hold me still", "Turn the morning down"]),
    SpecSection(name="Chorus", duration_s=30, lines=["Until you come around"])])


def heard(*triples):
    return [Word(word=w, start=s, end=e, confidence=0.9) for w, s, e in triples]


def test_exact_match_gives_spec_spelling_with_heard_timing():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2),
              ("turn", 5.0, 5.3), ("the", 5.3, 5.4), ("morning", 5.4, 6.0), ("down", 6.0, 6.8),
              ("until", 20.0, 20.4), ("you", 20.4, 20.6), ("come", 20.6, 21.0), ("around", 21.0, 22.0))
    lines = transcribe.align_lines(SPEC, h)
    assert [(l.section, l.start, l.end) for l in lines] == [("Verse 1", 1.0, 3.2), ("Verse 1", 5.0, 6.8), ("Chorus", 20.0, 22.0)]
    assert [w.word for w in lines[0].words] == ["Glass", "hour,", "hold", "me", "still"]      # OUR spelling
    assert lines[0].confidence == pytest.approx(0.9)


def test_a_misheard_word_is_interpolated_and_marked_zero_confidence():
    h = heard(("glass", 1.0, 1.3), ("our", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2))
    line = transcribe.align_lines(SPEC, h)[0]
    hour = line.words[1]
    assert hour.word == "hour," and hour.confidence == 0.0 and 1.3 <= hour.start <= hour.end <= 2.0
    assert line.confidence == pytest.approx(0.9 * 4 / 5)


def test_an_unsung_line_is_reported_not_invented():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2))
    lines = transcribe.align_lines(SPEC, h)
    assert (lines[1].start, lines[1].end, lines[1].words, lines[1].confidence) == (None, None, [], 0.0)


def test_sections_follow_the_aligned_lines():
    h = heard(("glass", 1.0, 1.3), ("hour", 1.3, 1.8), ("hold", 2.0, 2.3), ("me", 2.3, 2.5), ("still", 2.5, 3.2),
              ("until", 20.0, 20.4), ("you", 20.4, 20.6), ("come", 20.6, 21.0), ("around", 21.0, 22.0))
    secs = transcribe.sections_from_lines(transcribe.align_lines(SPEC, h), 40.0)
    assert [(s.label, s.start, s.end) for s in secs] == [("Verse 1", 1.0, 20.0), ("Chorus", 20.0, 40.0)]
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_transcribe.py -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/transcribe.py`**

```python
"""The engine's SECOND run: analyse our own chosen take, then align the lyric lines we wrote to what was sung.
The chart is derived from the actual recording, so the two can never disagree."""
import re
from difflib import SequenceMatcher

from .analysis import OBJECTIVE_EARS, analyze
from .audio import to_wav
from .config import load_config
from .jsonio import read_json, write_model
from .models import Chosen, LyricLine, Section, SongSpec, Transcription, Word
from .paths import SongPaths

GUESS_WORD_S = 0.3
MAX_LINE_SPAN_S = 60.0


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def _fill(times: list[tuple[float, float] | None]) -> list[tuple[float, float]]:
    """Interpolate timings for unmatched words between their matched neighbours."""
    out = list(times)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        n = j - i
        left = out[i - 1][1] if i > 0 else out[j][0] - GUESS_WORD_S * n
        right = out[j][0] if j < len(out) else left + GUESS_WORD_S * n
        step = max(0.0, right - left) / n
        for k in range(n):
            out[i + k] = (round(left + k * step, 3), round(left + (k + 1) * step, 3))
        i = j
    return out


def align_lines(spec: SongSpec, heard: list[Word]) -> list[LyricLine]:
    tokens = [(s.name, li, tok) for s in spec.sections for li, line in enumerate(s.lines) for tok in line.split()]
    a, b = [_norm(t[2]) for t in tokens], [_norm(w.word) for w in heard]
    match: dict[int, int] = {}
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            if a[blk.a + k]:
                match[blk.a + k] = blk.b + k

    lines: list[LyricLine] = []
    cursor = 0
    for s in spec.sections:
        for text in s.lines:
            n = len(text.split())
            idxs = list(range(cursor, cursor + n))
            cursor += n
            hits = [i for i in idxs if i in match]
            if hits:
                span = heard[match[hits[-1]]].end - heard[match[hits[0]]].start
            if not hits or span > MAX_LINE_SPAN_S or span < 0:
                lines.append(LyricLine(section=s.name, text=text, start=None, end=None, words=[], confidence=0.0))
                continue
            times = _fill([(heard[match[i]].start, heard[match[i]].end) if i in match else None for i in idxs])
            words = [Word(word=tokens[i][2], start=t[0], end=t[1],
                          confidence=heard[match[i]].confidence if i in match else 0.0) for i, t in zip(idxs, times)]
            mean = sum(heard[match[i]].confidence for i in hits) / len(hits)
            lines.append(LyricLine(section=s.name, text=text, start=words[0].start, end=words[-1].end, words=words,
                                   confidence=round(mean * len(hits) / n, 3)))
    return lines


def sections_from_lines(lines: list[LyricLine], duration_s: float) -> list[Section]:
    firsts: list[tuple[str, float, list[float]]] = []
    for line in lines:
        if line.start is None:
            continue
        if firsts and firsts[-1][0] == line.section:
            firsts[-1][2].append(line.confidence)
        else:
            firsts.append((line.section, line.start, [line.confidence]))
    return [Section(label=name, start=start, end=firsts[i + 1][1] if i + 1 < len(firsts) else duration_s,
                    confidence=round(sum(confs) / len(confs), 3)) for i, (name, start, confs) in enumerate(firsts)]


def run_transcribe(song: str, force: bool = False) -> Transcription:
    paths = SongPaths(song)
    chosen = Chosen(**read_json(paths.require(paths.chosen, "pick")))
    if paths.transcription.exists() and not force:
        existing = Transcription(**read_json(paths.transcription))
        if existing.take == chosen.take:
            print(f"- transcription of take {chosen.take} exists (use --force to redo)")
            return existing
    spec = SongSpec(**read_json(paths.require(paths.spec, "compose")))
    wav = paths.cache / f"chosen-{chosen.sha1[:16]}.wav"
    if not wav.exists():
        to_wav(paths.takes_dir / chosen.file, wav)
    print(f"> transcribing take {chosen.take} — same engine that analysed the reference")
    result = analyze(wav, paths.cache, load_config(), ears=set(OBJECTIVE_EARS), lyrics_hint=" ".join(spec.all_lines())[:900])
    lines = align_lines(spec, result.lyrics)
    aligned = [l for l in lines if l.start is not None]
    if len(aligned) * 2 >= len(lines) and result.global_info:          # our own form beats a DSP guess at it
        result.sections = sections_from_lines(lines, result.global_info.duration_s)
    out = Transcription(take=chosen.take, analysis=result, lines=lines)
    write_model(paths.transcription, out)
    print(f"  {len(aligned)}/{len(lines)} lyric lines aligned, {len(result.chords)} chords → {paths.transcription}")
    return out
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_transcribe.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/songcomposer/transcribe.py tests/test_transcribe.py
git commit -m "feat: transcribe chosen take (engine run #2) and align spec lyric lines to sung words"
```

---

### Task 22: Beat-grid timing, capo suggestion, and `chords.txt`

**Files:**
- Create: `src/songcomposer/render/__init__.py` (empty), `src/songcomposer/render/timing.py`, `src/songcomposer/render/guitar.py`, `src/songcomposer/render/chordsheet.py`
- Test: `tests/test_timing.py`, `tests/test_chordsheet.py`

**Interfaces:**
- Produces (`render.timing`):
  - `BeatMap(beats: list[float], downbeats: list[float], beats_per_bar: int)` with `.sixteenth(t: float) -> int` (absolute 16th index from bar 0, ≥ 0), `.beat_index(t: float) -> int`, `.bar16: int`, `.bpb: int`; classmethod `BeatMap.from_analysis(a: Analysis) -> BeatMap`
  - `split_sixteenths(n: int) -> list[int]` — into notatable lengths `16,12,8,6,4,3,2,1`
  - `split_at_bars(start16: int, len16: int, bar16: int) -> list[tuple[int, int]]`
- Produces (`render.guitar`): `transpose_harte(harte: str, semitones: int) -> str`, `suggest_capo(chords: list[Chord]) -> tuple[int, dict[str, str]]` (capo fret, sounding symbol → shape symbol; `(0, {})` when no capo helps)
- Produces (`render.chordsheet`): `label(chord: Chord) -> str` (appends `?` below `LOW_CONFIDENCE`), `render_chordsheet(title: str, t: Transcription) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_timing.py`:
```python
import pytest

from songcomposer.render.timing import BeatMap, split_at_bars, split_sixteenths

BEATS = [0.5 + i * 0.6 for i in range(16)]                 # first downbeat at 0.5 s


def test_sixteenth_positions():
    m = BeatMap(BEATS, BEATS[::4], 4)
    assert m.bar16 == 16
    assert m.sixteenth(0.5) == 0 and m.sixteenth(1.1) == 4 and m.sixteenth(0.5 + 0.15) == 1
    assert m.sixteenth(0.5 + 2.4) == 16                     # bar 2
    assert m.sixteenth(0.0) == 0                            # before the grid clamps to 0
    assert m.sixteenth(BEATS[-1] + 0.6) == 64               # extrapolates past the last beat
    assert m.beat_index(1.75) == 2


def test_pickup_beats_before_the_first_downbeat():
    m = BeatMap(BEATS, [BEATS[2]], 4)
    assert m.sixteenth(BEATS[2]) == 0 and m.sixteenth(BEATS[3]) == 4


@pytest.mark.parametrize("n,parts", [(16, [16]), (5, [4, 1]), (7, [6, 1]), (11, [8, 3]), (1, [1]), (13, [12, 1])])
def test_split_sixteenths(n, parts):
    assert split_sixteenths(n) == parts


def test_split_at_bars():
    assert split_at_bars(14, 6, 16) == [(14, 2), (16, 4)]
    assert split_at_bars(0, 40, 16) == [(0, 16), (16, 16), (32, 8)]
```

`tests/test_chordsheet.py`:
```python
from songcomposer.models import Analysis, Chord, GlobalInfo, LyricLine, Transcription, Word
from songcomposer.render.chordsheet import render_chordsheet
from songcomposer.render.guitar import suggest_capo, transpose_harte


def ch(harte, symbol, onset, dur, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q or "maj", extensions=[], bass=None,
                 onset=onset, duration=dur, confidence=conf)


def line(section, text, start, step=0.5):
    words = [Word(word=w, start=start + i * step, end=start + (i + 1) * step, confidence=0.9) for i, w in enumerate(text.split())]
    return LyricLine(section=section, text=text, start=words[0].start, end=words[-1].end, words=words, confidence=0.9)


def make(chords, lines):
    a = Analysis(audio_sha1="a" * 40, engines={}, chords=chords,
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4",
                                        loudness_lufs=-14, duration_s=60))
    return Transcription(take=1, analysis=a, lines=lines)


def test_chords_sit_above_the_word_they_land_on():
    t = make([ch("C:maj", "C", 10.0, 1.5), ch("A:min", "Am", 11.5, 2.0)],
             [line("Verse 1", "Glass hour hold me still", 10.0)])          # "me" starts at 11.5
    out = render_chordsheet("Glass Hour", t).splitlines()
    i = out.index("Glass hour hold me still")
    assert out[i - 2] == "[Verse 1]"
    assert out[i - 1] == "C" + " " * 15 + "Am"
    assert out[i - 1].index("Am") == out[i].index("me")


def test_low_confidence_chords_are_marked_and_explained():
    out = render_chordsheet("T", make([ch("F:maj", "F", 10.0, 2.0, conf=0.3)], [line("Verse 1", "one two", 10.0)]))
    assert "F?" in out and "? = low-confidence" in out and "1 of 1 chords" in out


def test_intro_chords_and_unaligned_lines_are_shown_honestly():
    lines = [line("Verse 1", "one two", 10.0),
             LyricLine(section="Verse 1", text="never sung", start=None, end=None, words=[], confidence=0.0)]
    out = render_chordsheet("T", make([ch("G:maj", "G", 0.0, 4.0), ch("C:maj", "C", 10.0, 2.0)], lines))
    assert "[Intro]\nG" in out
    assert "never sung   (timing not detected — chords not placed)" in out


def test_header_has_key_tempo_and_meter():
    out = render_chordsheet("Glass Hour", make([], [line("V", "a b", 1.0)]))
    assert out.startswith("Glass Hour\n") and "Key: C major" in out and "100 BPM" in out and "4/4" in out


def test_transpose_harte():
    assert transpose_harte("Bb:min7/b7", -1) == "A:min7/b7"
    assert transpose_harte("C:maj", 2) == "D:maj"


def test_capo_turns_flat_key_shapes_into_open_shapes():
    capo, shapes = suggest_capo([ch("Bb:maj", "Bb", 0, 4), ch("Eb:maj", "Eb", 4, 4), ch("F:maj", "F", 8, 4), ch("G:min", "Gm", 12, 4)])
    assert capo == 3 and shapes == {"Bb": "G", "Eb": "C", "F": "D", "Gm": "Em"}


def test_no_capo_when_the_shapes_are_already_open():
    assert suggest_capo([ch("G:maj", "G", 0, 4), ch("C:maj", "C", 4, 4), ch("D:maj", "D", 8, 4)]) == (0, {})
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_timing.py tests/test_chordsheet.py -v` → `ModuleNotFoundError: songcomposer.render`.

- [ ] **Step 3: Write `src/songcomposer/render/timing.py`**

```python
"""Seconds → musical position, using the measured beat grid (never an assumed constant tempo)."""
from bisect import bisect_right
from statistics import median

from ..models import Analysis

NOTATABLE = (16, 12, 8, 6, 4, 3, 2, 1)       # in sixteenths: whole, dotted half, half, dotted quarter, quarter, …


class BeatMap:
    def __init__(self, beats: list[float], downbeats: list[float], beats_per_bar: int):
        if len(beats) < 2:
            raise ValueError("need at least two beats to build a grid")
        self.beats, self.bpb = list(beats), beats_per_bar
        self.interval = median(b - a for a, b in zip(beats, beats[1:]))
        first = downbeats[0] if downbeats else beats[0]
        self.offset = min(range(len(beats)), key=lambda i: abs(beats[i] - first))     # beat index where bar 0 starts

    @classmethod
    def from_analysis(cls, a: Analysis) -> "BeatMap":
        bpb = int(a.global_info.time_signature.split("/")[0]) if a.global_info else 4
        return cls(a.beats, a.downbeats, bpb)

    @property
    def bar16(self) -> int:
        return self.bpb * 4

    def _beat_float(self, t: float) -> float:
        b = self.beats
        if t <= b[0]:
            return (t - b[0]) / self.interval
        if t >= b[-1]:
            return len(b) - 1 + (t - b[-1]) / self.interval
        i = bisect_right(b, t) - 1
        return i + (t - b[i]) / (b[i + 1] - b[i])

    def sixteenth(self, t: float) -> int:
        return max(0, round((self._beat_float(t) - self.offset) * 4))

    def beat_index(self, t: float) -> int:
        return max(0, round(self._beat_float(t) - self.offset))


def split_sixteenths(n: int) -> list[int]:
    out = []
    while n > 0:
        piece = next(p for p in NOTATABLE if p <= n)
        out.append(piece)
        n -= piece
    return out


def split_at_bars(start16: int, len16: int, bar16: int) -> list[tuple[int, int]]:
    out = []
    while len16 > 0:
        room = bar16 - start16 % bar16
        take = min(room, len16)
        out.append((start16, take))
        start16, len16 = start16 + take, len16 - take
    return out
```

- [ ] **Step 4: Write `src/songcomposer/render/guitar.py`**

```python
"""Guitar-specific knowledge. This is the ONLY package allowed to know what a fret or a capo is."""
from ..chordsym import parse_harte, transpose_name
from ..models import Chord

OPEN_FRIENDLY = {"C", "D", "E", "G", "A", "Am", "Dm", "Em", "A7", "B7", "C7", "D7", "E7", "G7", "Cmaj7", "Fmaj7",
                 "Am7", "Dm7", "Em7", "Dsus2", "Dsus4", "Asus2", "Asus4", "Esus4", "Cadd9"}
MAX_CAPO = 7
CAPO_PENALTY = 0.02


def transpose_harte(harte: str, semitones: int) -> str:
    root, sep, rest = harte.partition(":")
    bare_root, slash, bass = root.partition("/")
    new_root = transpose_name(bare_root, semitones, prefer_flats="b" in bare_root)
    return f"{new_root}{slash}{bass}{sep}{rest}"


def _shape_symbol(chord: Chord, capo: int) -> str:
    parsed = parse_harte(transpose_harte(chord.harte, -capo))
    return parsed.symbol if parsed else chord.symbol


def suggest_capo(chords: list[Chord]) -> tuple[int, dict[str, str]]:
    total = sum(c.duration for c in chords)
    if not total:
        return 0, {}

    def friendliness(capo: int) -> float:
        good = sum(c.duration for c in chords if _shape_symbol(c, capo).split("/")[0] in OPEN_FRIENDLY)
        return good / total - CAPO_PENALTY * capo

    best = max(range(MAX_CAPO + 1), key=friendliness)
    if best == 0 or friendliness(best) <= friendliness(0) + 0.1:        # only suggest a capo when it clearly helps
        return 0, {}
    return best, {c.symbol: _shape_symbol(c, best) for c in chords}
```

- [ ] **Step 5: Write `src/songcomposer/render/chordsheet.py`**

```python
"""chords.txt — lyrics with chords above the line, placed from the transcription of the actual take."""
from ..models import LOW_CONFIDENCE, Chord, LyricLine, Transcription
from .guitar import suggest_capo

LEAD_S = 0.15                 # a chord this close before a word belongs to that word
OUTRO_GAP_S = 4.0


def label(chord: Chord) -> str:
    return chord.symbol + ("?" if chord.confidence < LOW_CONFIDENCE else "")


def _rows(line: LyricLine, chords: list[Chord]) -> tuple[str, str]:
    offsets, pos = [], 0
    for w in line.words:
        found = line.text.find(w.word, pos)
        found = pos if found < 0 else found
        offsets.append(found)
        pos = found + len(w.word)
    row = ""
    for c in chords:
        k = next((i for i, w in enumerate(line.words) if w.start >= c.onset - LEAD_S), None)
        col = offsets[k] if k is not None else max(len(line.text), len(row)) + 2
        col = max(col, len(row) + 1 if row else col)
        row = row.ljust(col) + label(c)
    return row, line.text


def render_chordsheet(title: str, t: Transcription) -> str:
    a = t.analysis
    chords = sorted(a.chords, key=lambda c: c.onset)
    low = sum(1 for c in chords if c.confidence < LOW_CONFIDENCE)
    out = [title, "=" * len(title)]
    if a.global_info:
        g = a.global_info
        out.append(f"Key: {g.key} (confidence {g.key_confidence:.2f})   Tempo: {g.tempo_bpm:.0f} BPM   Time: {g.time_signature}")
    capo, shapes = suggest_capo(chords)
    if capo:
        out.append(f"Capo suggestion: fret {capo} — play " + ", ".join(f"{k}→{v}" for k, v in shapes.items()))
    out += [f"Transcribed from take {t.take}. ? = low-confidence chord (below {LOW_CONFIDENCE}): {low} of {len(chords)} chords.", ""]

    timed = [l for l in t.lines if l.start is not None]
    if timed:
        intro = [c for c in chords if c.onset < timed[0].start - LEAD_S]
        if intro:
            out += ["[Intro]", "  ".join(label(c) for c in intro), ""]
    section = None
    for line in t.lines:
        if line.section != section:
            if section is not None:
                out.append("")
            out.append(f"[{line.section}]")
            section = line.section
        if line.start is None:
            out.append(f"{line.text}   (timing not detected — chords not placed)")
            continue
        later = [l.start for l in timed if l.start > line.start]
        window_end = later[0] if later else line.end + OUTRO_GAP_S
        mine = [c for c in chords if line.start - LEAD_S <= c.onset < window_end - LEAD_S]
        row, text = _rows(line, mine)
        out += [row, text] if row else [text]
    if timed:
        outro = [c for c in chords if c.onset >= timed[-1].end + OUTRO_GAP_S - LEAD_S]
        if outro:
            out += ["", "[Outro]", "  ".join(label(c) for c in outro)]
    return "\n".join(out) + "\n"
```

- [ ] **Step 6: Run tests** — `uv run pytest tests/test_timing.py tests/test_chordsheet.py -v` → PASS.

- [ ] **Step 7: Commit**

```bash
git add src/songcomposer/render tests/test_timing.py tests/test_chordsheet.py
git commit -m "feat: beat-grid timing, capo suggestion, chord sheet renderer with visible low-confidence marks"
```

---

### Task 23: `tab.txt` — chord shapes, strumming/picking pattern, melody tab

**Files:**
- Modify: `src/songcomposer/render/guitar.py` (shapes, fret mapping, strum detection)
- Create: `src/songcomposer/render/tab.py`
- Test: `tests/test_tab.py`

**Interfaces:**
- Produces (`render.guitar`):
  - `TUNING = (40, 45, 50, 55, 59, 64)`; `shape_for(chord: Chord) -> str | None` — six characters low-E→high-e, e.g. `"x32010"`; barre shapes are dot-separated when any fret ≥ 10 (`"10.12.12.11.10.10"`); `None` when we have no honest shape
  - `into_range(pitch: int) -> int` — octave-shift into 40–79
  - `fret_positions(pitches: list[int]) -> list[tuple[int, int]]` — `(string 0=low E … 5=high e, fret)`, staying near the previous position
  - `strum_onsets(notes: list[Note]) -> list[float]` — moments where ≥ 3 `other`-stem notes start within 40 ms
  - `bar_pattern(onsets: list[float], bar_start16: int, beatmap: BeatMap) -> str` — 8 characters, `D`/`U`/`-` on the eighth-note grid
- Produces (`render.tab`): `render_tab(title: str, t: Transcription) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_tab.py`:
```python
from songcomposer.models import Analysis, Chord, GlobalInfo, Note, Section, Transcription
from songcomposer.render import guitar
from songcomposer.render.tab import render_tab
from songcomposer.render.timing import BeatMap

BEATS = [i * 0.6 for i in range(32)]


def ch(harte, symbol, onset=0.0, dur=2.4, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


def test_open_shapes_and_generated_barres():
    assert guitar.shape_for(ch("C:maj", "C")) == "x32010"
    assert guitar.shape_for(ch("A:min7", "Am7")) == "x02010"
    assert guitar.shape_for(ch("F#:min", "F#m")) == "244222"               # E-shape barre at fret 2
    assert guitar.shape_for(ch("D:7", "D7")) == "xx0212"
    assert guitar.shape_for(ch("D#:maj", "D#")) == "11.13.13.12.11.11"
    assert guitar.shape_for(ch("C:weird", "C(weird)")) is None


def test_fret_positions_prefer_staying_put():
    pos = guitar.fret_positions([64, 65, 67])                              # E4 F4 G4
    assert pos[0] in [(5, 0), (4, 5)]
    assert all(abs(b[1] - a[1]) <= 4 for a, b in zip(pos, pos[1:]))
    assert guitar.into_range(28) == 40 and guitar.into_range(90) == 78


def test_strum_detection_and_pattern():
    notes = [Note(pitch=p, onset=t + j * 0.01, duration=0.3, confidence=0.8, stem="other")
             for t in (0.0, 0.6, 0.9, 1.2, 1.8, 2.1) for j, p in enumerate((48, 55, 60, 64))]
    notes.append(Note(pitch=70, onset=1.5, duration=0.2, confidence=0.8, stem="other"))      # a lone note: not a strum
    onsets = guitar.strum_onsets(notes)
    assert [round(o, 1) for o in onsets] == [0.0, 0.6, 0.9, 1.2, 1.8, 2.1]
    assert guitar.bar_pattern(onsets, 0, BeatMap(BEATS, BEATS[::4], 4)) == "D-DUD-DU"


def test_render_tab_sections():
    melody = [Note(pitch=64, onset=0.0, duration=0.6, confidence=0.9, stem="vocals"),
              Note(pitch=67, onset=0.6, duration=0.6, confidence=0.3, stem="vocals")]
    strums = [Note(pitch=p, onset=t, duration=0.3, confidence=0.8, stem="other") for t in (0.0, 0.6, 1.2, 1.8) for p in (48, 55, 60)]
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=melody + strums,
                 chords=[ch("C:maj", "C"), ch("C:weird", "C(weird)", 2.4, 2.4, conf=0.2)],
                 sections=[Section(label="Verse 1", start=0.0, end=19.2, confidence=0.9)],
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=19.2))
    out = render_tab("Glass Hour", Transcription(take=1, analysis=a, lines=[]))
    assert f"{'C':<11}x32010" in out
    assert f"{'C(weird)?':<11}(no shape in dictionary — work it out from the chord name)" in out
    assert "Verse 1: strummed   | D-D-D-D- |" in out
    assert "e|0---" + "-" * 12 + "(3)-" in out                           # E4 open on beat 1, low-confidence G4 (in parentheses) on beat 2
    assert "(n) = low-confidence note" in out
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_tab.py -v` → `AttributeError: module ... has no attribute 'shape_for'`.

- [ ] **Step 3: Append to `src/songcomposer/render/guitar.py`**

```python
from collections import Counter  # noqa: E402

from ..chordsym import pitch_class  # noqa: E402
from ..models import Note  # noqa: E402
from .timing import BeatMap  # noqa: E402

TUNING = (40, 45, 50, 55, 59, 64)            # E2 A2 D3 G3 B3 E4
MAX_FRET = 15
SHAPES = {
    "C": "x32010", "D": "xx0232", "E": "022100", "F": "133211", "G": "320003", "A": "x02220", "B": "x24442",
    "Am": "x02210", "Dm": "xx0231", "Em": "022000", "Bm": "x24432",
    "A7": "x02020", "B7": "x21202", "C7": "x32310", "D7": "xx0212", "E7": "020100", "G7": "320001",
    "Cmaj7": "x32000", "Dmaj7": "xx0222", "Fmaj7": "xx3210", "Gmaj7": "320002", "Amaj7": "x02120",
    "Am7": "x02010", "Dm7": "xx0211", "Em7": "022030",
    "Dsus2": "xx0230", "Dsus4": "xx0233", "Asus2": "x02200", "Asus4": "x02230", "Esus4": "022200", "Cadd9": "x32030",
}
_BARRE = {"maj": (0, 2, 2, 1, 0, 0), "min": (0, 2, 2, 0, 0, 0), "7": (0, 2, 0, 1, 0, 0), "min7": (0, 2, 0, 0, 0, 0),
          "maj7": (0, 2, 1, 1, 0, 0), "sus4": (0, 2, 2, 2, 0, 0)}


def shape_for(chord: Chord) -> str | None:
    name = chord.symbol.split("/")[0]
    if name in SHAPES:
        return SHAPES[name]
    if chord.quality not in _BARRE:
        return None
    fret = (pitch_class(chord.root) - 4) % 12                       # root on the low E string
    frets = [fret + d for d in _BARRE[chord.quality]]
    return ".".join(map(str, frets)) if max(frets) >= 10 else "".join(map(str, frets))


def into_range(pitch: int) -> int:
    while pitch < TUNING[0]:
        pitch += 12
    while pitch > TUNING[-1] + MAX_FRET:
        pitch -= 12
    return pitch


def fret_positions(pitches: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    hand = 0
    for pitch in pitches:
        pitch = into_range(pitch)
        options = [(s, pitch - open_) for s, open_ in enumerate(TUNING) if 0 <= pitch - open_ <= MAX_FRET]
        best = min(options, key=lambda sf: (abs(sf[1] - hand) if sf[1] else 0.5, sf[1]))
        out.append(best)
        if best[1]:
            hand = best[1]
    return out


def strum_onsets(notes: list[Note]) -> list[float]:
    starts = sorted(n.onset for n in notes if n.stem == "other")
    out, i = [], 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] - starts[i] <= 0.04:
            j += 1
        if j - i + 1 >= 3:
            out.append(starts[i])
        i = j + 1
    return out


def bar_pattern(onsets: list[float], bar_start16: int, beatmap: BeatMap) -> str:
    hit = {(beatmap.sixteenth(o) - bar_start16) // 2 for o in onsets
           if bar_start16 <= beatmap.sixteenth(o) < bar_start16 + beatmap.bar16 and (beatmap.sixteenth(o) - bar_start16) % 2 == 0}
    return "".join(("D" if slot % 2 == 0 else "U") if slot in hit else "-" for slot in range(beatmap.bar16 // 2))


def common_pattern(patterns: list[str]) -> str | None:
    played = [p for p in patterns if p.strip("-")]
    return Counter(played).most_common(1)[0][0] if played else None
```

- [ ] **Step 4: Write `src/songcomposer/render/tab.py`**

```python
"""tab.txt — chord shapes, a strumming (or picking) read per section, and the vocal melody as guitar tab."""
from ..models import LOW_CONFIDENCE, Transcription
from . import guitar
from .chordsheet import label
from .timing import BeatMap

STRINGS = "EADGBe"
BARS_PER_SYSTEM = 4
SLOT = 4                                    # characters per sixteenth


def _shapes_block(t: Transcription) -> list[str]:
    out, seen = ["CHORD SHAPES (low E → high e; x = muted)"], set()
    for c in sorted(t.analysis.chords, key=lambda c: c.onset):
        if c.symbol in seen:
            continue
        seen.add(c.symbol)
        shape = guitar.shape_for(c)
        out.append(f"{label(c):<11}{shape}" if shape else
                   f"{label(c):<11}(no shape in dictionary — work it out from the chord name)")
    return out


def _rhythm_block(t: Transcription, beatmap: BeatMap) -> list[str]:
    out = ["RIGHT HAND (read from the recording; D = down, U = up, eighth-note grid)"]
    a = t.analysis
    strums = guitar.strum_onsets(a.notes)
    for s in a.sections:
        first, last = beatmap.sixteenth(s.start) // beatmap.bar16, beatmap.sixteenth(s.end) // beatmap.bar16
        patterns = [guitar.bar_pattern(strums, bar * beatmap.bar16, beatmap) for bar in range(first, max(first + 1, last))]
        pattern = guitar.common_pattern(patterns)
        picked = sum(1 for n in a.notes if n.stem == "other" and s.start <= n.onset < s.end)
        if pattern:
            out.append(f"{s.label}: strummed   | {pattern} |")
        elif picked:
            out.append(f"{s.label}: picked / arpeggiated — no full strums detected; arpeggiate the chord shapes")
        else:
            out.append(f"{s.label}: no guitar-range accompaniment detected")
    return out


def _melody_block(t: Transcription, beatmap: BeatMap) -> list[str]:
    melody = sorted((n for n in t.analysis.notes if n.stem == "vocals"), key=lambda n: n.onset)
    out = ["MELODY (vocal line in guitar range; (n) = low-confidence note)"]
    if not melody:
        return out + ["no melody notes were transcribed"]
    positions = guitar.fret_positions([n.pitch for n in melody])
    n_bars = beatmap.sixteenth(melody[-1].onset) // beatmap.bar16 + 1
    grid = [[["-" * SLOT] * beatmap.bar16 for _ in range(n_bars)] for _ in range(6)]
    for note, (string, fret) in zip(melody, positions):
        bar, slot = divmod(beatmap.sixteenth(note.onset), beatmap.bar16)
        text = f"({fret})" if note.confidence < LOW_CONFIDENCE else str(fret)
        grid[string][bar][slot] = text.ljust(SLOT, "-")[:SLOT]
    for first in range(0, n_bars, BARS_PER_SYSTEM):
        out.append(f"bar {first + 1}")
        for string in range(5, -1, -1):
            bars = ["".join(grid[string][b]) for b in range(first, min(first + BARS_PER_SYSTEM, n_bars))]
            out.append(f"{STRINGS[string]}|" + "|".join(bars) + "|")
        out.append("")
    return out


def render_tab(title: str, t: Transcription) -> str:
    a = t.analysis
    head = [title, "=" * len(title), f"Guitar tab, transcribed from take {t.take}. Standard tuning."]
    if a.global_info:
        head.append(f"Key: {a.global_info.key}   Tempo: {a.global_info.tempo_bpm:.0f} BPM   Time: {a.global_info.time_signature}")
    head.append("Inner parts of a mix transcribe approximately: treat the right-hand read as a starting point, the chords and melody as the reliable part.")
    if len(a.beats) < 2:
        return "\n".join(head + ["", "No beat grid was detected — tab cannot be laid out. See chords.txt."]) + "\n"
    beatmap = BeatMap.from_analysis(a)
    blocks = [head, _shapes_block(t), _rhythm_block(t, beatmap), _melody_block(t, beatmap)]
    return "\n\n".join("\n".join(b) for b in blocks) + "\n"
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_tab.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/render/guitar.py src/songcomposer/render/tab.py tests/test_tab.py
git commit -m "feat: tab renderer — chord shapes, right-hand pattern, melody tab with low-confidence marks"
```

---

### Task 24: MusicXML lead sheet (music21)

**Files:**
- Create: `src/songcomposer/render/leadsheet.py` (shared quantisation), `src/songcomposer/render/musicxml.py`
- Test: `tests/test_musicxml.py`

**Interfaces:**
- Produces (`render.leadsheet`), shared by Tasks 24 and 25:
  - `MelodyEvent(start16: int, len16: int, pitch: int | None, low: bool, lyric: str | None)` (dataclass; `pitch=None` is a rest)
  - `ChordEvent(start_beat: int, beats: int, chord: Chord | None)` (`None` = no chord sounding)
  - `melody_events(t: Transcription, beatmap: BeatMap) -> list[MelodyEvent]` — monophonic, gap-filled with rests, contiguous from 0 to the end of the last bar
  - `chord_events(t: Transcription, beatmap: BeatMap) -> list[ChordEvent]` — whole-beat quantised, contiguous from beat 0
- Produces (`render.musicxml`): `KIND: dict[str, str]` (Harte quality → MusicXML kind), `write_musicxml(title: str, t: Transcription, dest: Path) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_musicxml.py`:
```python
import pytest

from songcomposer.models import Analysis, Chord, GlobalInfo, LyricLine, Note, Transcription, Word
from songcomposer.render.leadsheet import chord_events, melody_events
from songcomposer.render.timing import BeatMap

BEATS = [i * 0.6 for i in range(16)]


def ch(harte, symbol, onset, dur, conf=0.9):
    root, _, q = harte.partition(":")
    return Chord(symbol=symbol, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


@pytest.fixture
def t():
    notes = [Note(pitch=64, onset=0.6, duration=0.6, confidence=0.9, stem="vocals"),
             Note(pitch=67, onset=1.2, duration=1.2, confidence=0.3, stem="vocals"),
             Note(pitch=40, onset=0.0, duration=2.0, confidence=0.9, stem="bass")]
    words = [Word(word="Glass", start=0.62, end=1.1, confidence=0.9), Word(word="hour", start=1.2, end=2.0, confidence=0.9)]
    a = Analysis(audio_sha1="a" * 40, engines={}, beats=BEATS, downbeats=BEATS[::4], notes=notes,
                 chords=[ch("C:maj", "C", 0.0, 2.4), ch("Bb:min7", "Bbm7", 4.8, 2.4, conf=0.2)],
                 global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=9.6))
    return Transcription(take=1, analysis=a, lines=[LyricLine(section="V", text="Glass hour", start=0.62, end=2.0, words=words, confidence=0.9)])


def test_melody_events_are_contiguous_with_rests_and_lyrics(t):
    ev = melody_events(t, BeatMap(BEATS, BEATS[::4], 4))
    assert [(e.start16, e.len16, e.pitch, e.low, e.lyric) for e in ev] == [
        (0, 4, None, False, None), (4, 4, 64, False, "Glass"), (8, 8, 67, True, "hour")]
    assert sum(e.len16 for e in ev) % 16 == 0


def test_chord_events_fill_gaps_with_none(t):
    ev = chord_events(t, BeatMap(BEATS, BEATS[::4], 4))
    assert [(e.start_beat, e.beats, e.chord.symbol if e.chord else None) for e in ev] == [(0, 4, "C"), (4, 4, None), (8, 4, "Bbm7")]


def test_musicxml_roundtrip(t, tmp_path):
    m21 = pytest.importorskip("music21")
    from songcomposer.render.musicxml import write_musicxml
    dest = tmp_path / "song.musicxml"
    write_musicxml("Glass Hour", t, dest)
    score = m21.converter.parse(str(dest))
    symbols = list(score.recurse().getElementsByClass(m21.harmony.ChordSymbol))
    assert [s.root().name for s in symbols] == ["C", "B-"] and symbols[1].chordKind == "minor-seventh"
    pitched = [n for n in score.recurse().notes if isinstance(n, m21.note.Note)]
    assert [n.pitch.midi for n in pitched] == [64, 67] and pitched[0].lyric == "Glass"
    assert "?" in [e.content for e in score.recurse().getElementsByClass(m21.expressions.TextExpression)]
    assert score.metadata.title == "Glass Hour"
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_musicxml.py -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/songcomposer/render/leadsheet.py`**

```python
"""Quantise the transcription onto the beat grid once, for both engraved outputs (MusicXML and LilyPond)."""
from dataclasses import dataclass

from ..models import LOW_CONFIDENCE, Chord, Transcription
from .timing import BeatMap

LYRIC_SNAP_S = 0.2


@dataclass
class MelodyEvent:
    start16: int
    len16: int
    pitch: int | None
    low: bool
    lyric: str | None


@dataclass
class ChordEvent:
    start_beat: int
    beats: int
    chord: Chord | None


def melody_events(t: Transcription, beatmap: BeatMap) -> list[MelodyEvent]:
    notes = sorted((n for n in t.analysis.notes if n.stem == "vocals"), key=lambda n: n.onset)
    words = [w for line in t.lines for w in line.words]
    placed: list[MelodyEvent] = []
    for n in notes:
        start = beatmap.sixteenth(n.onset)
        end = max(start + 1, beatmap.sixteenth(n.onset + n.duration))
        if placed and start < placed[-1].start16 + placed[-1].len16:
            if start <= placed[-1].start16:
                continue                                           # two notes on one slot: keep the first
            placed[-1].len16 = start - placed[-1].start16
        near = [w for w in words if abs(w.start - n.onset) <= LYRIC_SNAP_S]
        lyric = min(near, key=lambda w: abs(w.start - n.onset)).word if near else None
        if lyric and any(e.lyric == lyric and abs(e.start16 - start) <= 2 for e in placed):
            lyric = None
        placed.append(MelodyEvent(start, end - start, n.pitch, n.confidence < LOW_CONFIDENCE, lyric))
    out: list[MelodyEvent] = []
    cursor = 0
    for e in placed:
        if e.start16 > cursor:
            out.append(MelodyEvent(cursor, e.start16 - cursor, None, False, None))
        out.append(e)
        cursor = e.start16 + e.len16
    pad = -cursor % beatmap.bar16
    if pad:
        out.append(MelodyEvent(cursor, pad, None, False, None))
    return out


def chord_events(t: Transcription, beatmap: BeatMap) -> list[ChordEvent]:
    out: list[ChordEvent] = []
    cursor = 0
    for c in sorted(t.analysis.chords, key=lambda c: c.onset):
        start = max(cursor, beatmap.beat_index(c.onset))
        end = max(start + 1, beatmap.beat_index(c.onset + c.duration))
        if start > cursor:
            out.append(ChordEvent(cursor, start - cursor, None))
        out.append(ChordEvent(start, end - start, c))
        cursor = end
    return out
```

- [ ] **Step 4: Write `src/songcomposer/render/musicxml.py`**

```python
"""<song>.musicxml — melody, lyrics and chord symbols. Opens in MuseScore and Guitar Pro."""
from pathlib import Path

from ..models import Transcription
from .leadsheet import chord_events, melody_events
from .timing import BeatMap

KIND = {"maj": "major", "min": "minor", "dim": "diminished", "aug": "augmented", "5": "power", "1": "power",
        "7": "dominant", "maj7": "major-seventh", "min7": "minor-seventh", "minmaj7": "major-minor",
        "dim7": "diminished-seventh", "hdim7": "half-diminished", "6": "major-sixth", "maj6": "major-sixth",
        "min6": "minor-sixth", "9": "dominant-ninth", "maj9": "major-ninth", "min9": "minor-ninth",
        "11": "dominant-11th", "min11": "minor-11th", "13": "dominant-13th", "maj13": "major-13th", "min13": "minor-13th",
        "sus2": "suspended-second", "sus4": "suspended-fourth"}


def _m21_name(name: str) -> str:
    return name[0] + name[1:].replace("b", "-")                    # music21 spells flats with '-'


def write_musicxml(title: str, t: Transcription, dest: Path) -> None:
    from music21 import expressions, harmony, key, metadata, meter, note, stream, tempo
    a = t.analysis
    beatmap = BeatMap.from_analysis(a)
    part = stream.Part()
    g = a.global_info
    if g:
        part.insert(0, meter.TimeSignature(g.time_signature))
        part.insert(0, tempo.MetronomeMark(number=round(g.tempo_bpm)))
        tonic, _, mode = g.key.partition(" ")
        if mode in ("major", "minor"):
            part.insert(0, key.Key(_m21_name(tonic), mode))
    for e in melody_events(t, beatmap):
        if e.pitch is None:
            el = note.Rest(quarterLength=e.len16 / 4)
        else:
            el = note.Note(e.pitch, quarterLength=e.len16 / 4)
            if e.lyric:
                el.addLyric(e.lyric)
            if e.low:
                el.notehead = "x"                                   # visibly different: low-confidence pitch
        part.insert(e.start16 / 4, el)
    for e in chord_events(t, beatmap):
        if e.chord is None:
            continue
        kw = {"root": _m21_name(e.chord.root), "kind": KIND.get(e.chord.quality, "major")}
        if e.chord.bass:
            kw["bass"] = _m21_name(e.chord.bass)
        part.insert(e.start_beat, harmony.ChordSymbol(**kw))
        if e.chord.confidence < 0.5 or e.chord.quality not in KIND:
            part.insert(e.start_beat, expressions.TextExpression("?"))
    score = stream.Score()
    score.insert(0, metadata.Metadata(title=title, composer=f"Song Composer — transcribed from take {t.take}"))
    score.insert(0, part)
    score.makeNotation(inPlace=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(dest))
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_musicxml.py -v` → PASS.
  If the round-trip assertion on `chordKind` or `TextExpression` fails because music21 normalises differently, assert on what it actually returns (print it) — but the two pure `leadsheet` tests must pass exactly as written.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/render/leadsheet.py src/songcomposer/render/musicxml.py tests/test_musicxml.py
git commit -m "feat: MusicXML lead sheet — melody, lyrics, chord symbols, low-confidence marks"
```

---

### Task 25: Engraved `chart.pdf` (LilyPond)

LilyPond does the guitar-specific heavy lifting itself: `TabStaff` computes frets from pitches, `FretBoards` draws chord diagrams from `\chordmode`.

**Files:**
- Create: `src/songcomposer/render/lilypond.py`
- Test: `tests/test_lilypond.py`

**Interfaces:**
- Consumes: `melody_events`, `chord_events`, `split_sixteenths`, `split_at_bars`, `into_range`
- Produces: `lilypond.ly_pitch(midi: int, flats: bool) -> str`, `lilypond.ly_chord(chord: Chord, beats: int) -> str`, `lilypond.build_ly(title: str, t: Transcription) -> str`, `lilypond.write_pdf(title: str, t: Transcription, ly_path: Path, pdf_path: Path) -> bool` (False + printed reason if LilyPond is missing or fails; the `.ly` is always written)

- [ ] **Step 1: Install LilyPond**

```bash
scoop install lilypond
lilypond --version
```
Expected: `GNU LilyPond 2.26.0` (scoop `main` bucket, confirmed by the spike).

- [ ] **Step 2: Write the failing tests**

`tests/test_lilypond.py`:
```python
import shutil

import pytest

from songcomposer.render import lilypond
from test_musicxml import ch, t  # noqa: F401 — reuse the fixture and chord helper


@pytest.mark.parametrize("midi,flats,expected", [(60, False, "c'"), (48, False, "c"), (72, False, "c''"), (40, False, "e,"),
                                                 (61, False, "cis'"), (61, True, "des'"), (70, True, "bes'"), (63, True, "ees'")])
def test_ly_pitch(midi, flats, expected):
    assert lilypond.ly_pitch(midi, flats) == expected


@pytest.mark.parametrize("harte,symbol,beats,expected", [
    ("C:maj", "C", 4, "c1"), ("A:min", "Am", 2, "a2:m"), ("Bb:min7", "Bbm7", 3, "bes2.:m7"), ("F#:dim", "F#dim", 1, "fis4:dim"),
    ("D:sus4", "Dsus4", 4, "d1:sus4"), ("B:hdim7", "Bm7b5", 4, "b1:m7.5-"), ("C:maj7", "Cmaj7", 4, "c1:maj7")])
def test_ly_chord(harte, symbol, beats, expected):
    assert lilypond.ly_chord(ch(harte, symbol, 0, 1), beats) == expected


def test_ly_chord_with_bass():
    c = ch("C:maj", "C/E", 0, 1).model_copy(update={"bass": "E"})
    assert lilypond.ly_chord(c, 4) == "c1/e"


def test_build_ly_has_every_layer_and_marks_doubt(t):  # noqa: F811
    ly = lilypond.build_ly('Glass "Hour"', t)
    assert 'title = "Glass \\"Hour\\""' in ly
    for needle in ("\\new ChordNames", "\\new FretBoards", "\\new TabStaff", "\\lyricsto", "\\key c \\major",
                   "\\time 4/4", "\\tempo 4 = 100", "chordChanges = ##t"):
        assert needle in ly, needle
    assert "r4 e'4 \\parenthesize g'2" in ly                        # rest, sure note, low-confidence note
    assert '"Glass" "hour"' in ly
    assert "c1 s1 \\once \\override ChordName.color = #grey bes1:m7" in ly
    assert "c1 s1 bes1:m7" in ly                                    # FretBoards copy has no overrides


@pytest.mark.skipif(shutil.which("lilypond") is None, reason="lilypond not installed")
def test_pdf_is_produced(t, tmp_path):  # noqa: F811
    assert lilypond.write_pdf("Glass Hour", t, tmp_path / "chart.ly", tmp_path / "chart.pdf") is True
    assert (tmp_path / "chart.pdf").read_bytes().startswith(b"%PDF")


def test_missing_lilypond_is_reported_not_fatal(t, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(lilypond.shutil, "which", lambda name: None)
    assert lilypond.write_pdf("T", t, tmp_path / "chart.ly", tmp_path / "chart.pdf") is False
    assert (tmp_path / "chart.ly").exists() and "scoop install lilypond" in capsys.readouterr().out
```

- [ ] **Step 3: Run, verify failure** — `uv run pytest tests/test_lilypond.py -v` → `ImportError`.

- [ ] **Step 4: Write `src/songcomposer/render/lilypond.py`**

```python
"""chart.pdf — chord names, fretboard diagrams, melody, lyrics and tab, engraved by LilyPond from text we generate."""
import shutil
import subprocess
from pathlib import Path

from ..chordsym import pitch_class
from ..models import LOW_CONFIDENCE, Chord, Transcription
from .guitar import into_range
from .leadsheet import chord_events, melody_events
from .timing import BeatMap, split_at_bars, split_sixteenths

SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
FLAT = ["c", "des", "d", "ees", "e", "f", "ges", "g", "aes", "a", "bes", "b"]
DUR16 = {16: "1", 12: "2.", 8: "2", 6: "4.", 4: "4", 3: "8.", 2: "8", 1: "16"}
QUALITY = {"maj": "", "min": ":m", "dim": ":dim", "aug": ":aug", "5": ":1.5", "1": ":1.5", "7": ":7", "maj7": ":maj7",
           "min7": ":m7", "minmaj7": ":m7+", "dim7": ":dim7", "hdim7": ":m7.5-", "6": ":6", "maj6": ":6", "min6": ":m6",
           "9": ":9", "maj9": ":maj9", "min9": ":m9", "11": ":11", "min11": ":m11", "13": ":13", "maj13": ":maj13",
           "min13": ":m13", "sus2": ":sus2", "sus4": ":sus4"}


def _name(note_name: str) -> str:
    return (FLAT if "b" in note_name[1:] else SHARP)[pitch_class(note_name)]


def ly_pitch(midi: int, flats: bool) -> str:
    octave = midi // 12 - 4
    return (FLAT if flats else SHARP)[midi % 12] + ("'" * octave if octave > 0 else "," * -octave)


def ly_chord(chord: Chord, beats: int) -> str:
    out = f"{_name(chord.root)}{DUR16[beats * 4]}{QUALITY.get(chord.quality, '')}"
    return out + (f"/{_name(chord.bass)}" if chord.bass else "")


def _uses_flats(key_name: str) -> bool:
    tonic, _, mode = key_name.partition(" ")
    if "b" in tonic[1:]:
        return True
    return tonic == "F" if mode == "major" else tonic in ("D", "G", "C", "F")


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _melody(t: Transcription, beatmap: BeatMap, flats: bool) -> tuple[str, str]:
    music, words = [], []
    for e in melody_events(t, beatmap):
        pieces = [(s, n) for s, length in split_at_bars(e.start16, e.len16, beatmap.bar16) for n in split_sixteenths(length)]
        for i, (_, n) in enumerate(pieces):
            if e.pitch is None:
                music.append(f"r{DUR16[n]}")
                continue
            head = "\\parenthesize " if e.low and i == 0 else ""
            tie = "~" if i + 1 < len(pieces) else ""
            music.append(f"{head}{ly_pitch(into_range(e.pitch), flats)}{DUR16[n]}{tie}")
        if e.pitch is not None:
            words.append(_quote(e.lyric) if e.lyric else "_")
    return " ".join(music), " ".join(words)


def _harmonies(t: Transcription, beatmap: BeatMap, mark_doubt: bool) -> str:
    out = []
    for e in chord_events(t, beatmap):
        for i, (_, beats16) in enumerate(split_at_bars(e.start_beat * 4, e.beats * 4, beatmap.bar16)):
            for n in split_sixteenths(beats16):
                if e.chord is None:
                    out.append(f"s{DUR16[n]}")
                    continue
                if mark_doubt and e.chord.confidence < LOW_CONFIDENCE:
                    out.append("\\once \\override ChordName.color = #grey")
                out.append(ly_chord(e.chord, 1).replace("4", DUR16[n], 1))
    return " ".join(out)


def build_ly(title: str, t: Transcription) -> str:
    a = t.analysis
    beatmap = BeatMap.from_analysis(a)
    g = a.global_info
    flats = _uses_flats(g.key) if g else False
    music, words = _melody(t, beatmap, flats)
    setup = []
    if g:
        tonic, _, mode = g.key.partition(" ")
        if mode in ("major", "minor"):
            setup.append(f"\\key {_name(tonic)} \\{mode}")
        setup += [f"\\time {g.time_signature}", f"\\tempo 4 = {round(g.tempo_bpm)}"]
    return f"""\\version "2.24.0"
\\header {{
  title = {_quote(title)}
  composer = {_quote(f"Song Composer — transcribed from take {t.take}")}
  tagline = "Grey chord names and parenthesised notes are low-confidence detections. Melody is shown in guitar range."
}}
harmonies = \\chordmode {{ {_harmonies(t, beatmap, True)} }}
shapes = \\chordmode {{ {_harmonies(t, beatmap, False)} }}
melody = {{ {' '.join(setup)} {music} }}
words = \\lyricmode {{ {words} }}
\\score {{
  <<
    \\new ChordNames {{ \\set chordChanges = ##t \\harmonies }}
    \\new FretBoards {{ \\shapes }}
    \\new Staff {{ \\clef "treble_8" \\new Voice = "mel" {{ \\melody }} }}
    \\new Lyrics \\lyricsto "mel" {{ \\words }}
    \\new TabStaff {{ \\melody }}
  >>
  \\layout {{ }}
}}
"""


def write_pdf(title: str, t: Transcription, ly_path: Path, pdf_path: Path) -> bool:
    ly_path.parent.mkdir(parents=True, exist_ok=True)
    ly_path.write_text(build_ly(title, t), encoding="utf-8")
    if shutil.which("lilypond") is None:
        print(f"! lilypond not found — wrote {ly_path} but no PDF. Install with `scoop install lilypond` and re-run `chart`.")
        return False
    proc = subprocess.run(["lilypond", "-o", str(pdf_path.with_suffix("")), str(ly_path)],
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0 or not pdf_path.exists():
        print(f"! lilypond failed — {ly_path} is kept for inspection:\n{proc.stderr[-800:]}")
        return False
    return True
```

Note on `_harmonies`: `ly_chord(chord, 1)` always yields the duration `4`; the first `4` in the string is that duration (root names contain no digits), so replacing it once with the real duration is safe.

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_lilypond.py -v` → PASS, including the real PDF compile. Open the PDF once and look at it: chord names, fret diagrams, staff, lyrics and tab should all be present.

- [ ] **Step 6: Commit**

```bash
git add src/songcomposer/render/lilypond.py tests/test_lilypond.py
git commit -m "feat: LilyPond chart — chord names, fret diagrams, melody, lyrics, tab staff → PDF"
```

---

### Task 26: `lyrics.json` (video-project contract) and the `chart` stage

**Files:**
- Create: `src/songcomposer/render/lyricsjson.py`, `src/songcomposer/chart.py`
- Modify: `src/songcomposer/cli.py`
- Test: `tests/test_chart.py`

**Interfaces:**
- Produces:
  - `lyricsjson.build_lyrics_json(song: str, title: str, t: Transcription) -> dict` — the contract the video project consumes. Word objects are `{word, start, end, confidence}`: the same `{word, start, end}` shape `wordsFromAlignment()` produces in the video project, plus confidence.
  - `chart.run_chart(song: str, force: bool = False) -> dict[str, Path]`

`lyrics.json` shape:
```json
{"schema_version": 1, "song": "glass-hour", "title": "Glass Hour", "audio": "glass-hour.mp3", "duration_s": 151.2,
 "lines": [{"section": "Verse 1", "text": "…", "start": 12.4, "end": 15.9, "confidence": 0.87,
            "words": [{"word": "The", "start": 12.4, "end": 12.6, "confidence": 0.93}]}],
 "unaligned_lines": ["a line that was written but not detected in the audio"]}
```

- [ ] **Step 1: Write the failing tests**

`tests/test_chart.py`:
```python
import pytest

from songcomposer import chart
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import SongSpec, SpecSection
from songcomposer.paths import SongPaths
from songcomposer.render.lyricsjson import build_lyrics_json
from test_musicxml import t  # noqa: F401


def test_lyrics_json_contract(t):  # noqa: F811
    t.lines.append(t.lines[0].model_copy(update={"text": "never sung", "start": None, "end": None, "words": [], "confidence": 0.0}))
    out = build_lyrics_json("glass-hour", "Glass Hour", t)
    assert (out["schema_version"], out["audio"], out["duration_s"]) == (1, "glass-hour.mp3", 9.6)
    assert out["lines"][0]["words"][0] == {"word": "Glass", "start": 0.62, "end": 1.1, "confidence": 0.9}
    assert out["unaligned_lines"] == ["never sung"] and len(out["lines"]) == 1


def test_chart_writes_every_deliverable_into_out(root, t, monkeypatch):  # noqa: F811
    pytest.importorskip("music21")
    p = SongPaths("demo")
    write_model(p.spec, SongSpec(title="Glass Hour", style_prompt="s", target_duration_s=60,
                                 sections=[SpecSection(name="V", lines=["Glass hour"], duration_s=60)]))
    monkeypatch.setattr(chart, "run_transcribe", lambda song, force=False: t)
    written = chart.run_chart("demo")
    for path in (p.chords_txt, p.tab_txt, p.out_musicxml, p.lyrics_json, p.chart_ly):
        assert path.exists(), path
    assert set(written) >= {"chords", "tab", "musicxml", "lyrics"}
    assert all(root / "out" / "demo" in path.parents for path in written.values())       # deliverables stay here
    assert "Bbm7?" in p.chords_txt.read_text(encoding="utf-8")
    assert read_json(p.lyrics_json)["song"] == "demo"
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_chart.py -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/render/lyricsjson.py`**

```python
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
```

- [ ] **Step 4: Write `src/songcomposer/chart.py`**

```python
"""Stage: chart. Transcribe the chosen take, then render every guitar deliverable into out/<song>/."""
from pathlib import Path

from .jsonio import read_json, write_json
from .models import LOW_CONFIDENCE, SongSpec
from .paths import SongPaths
from .render.chordsheet import render_chordsheet
from .render.lilypond import write_pdf
from .render.lyricsjson import build_lyrics_json
from .render.musicxml import write_musicxml
from .render.tab import render_tab
from .transcribe import run_transcribe


def run_chart(song: str, force: bool = False) -> dict[str, Path]:
    paths = SongPaths(song)
    title = SongSpec(**read_json(paths.require(paths.spec, "compose"))).title
    t = run_transcribe(song, force=force)
    paths.out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    paths.chords_txt.write_text(render_chordsheet(title, t), encoding="utf-8")
    written["chords"] = paths.chords_txt
    paths.tab_txt.write_text(render_tab(title, t), encoding="utf-8")
    written["tab"] = paths.tab_txt
    write_json(paths.lyrics_json, build_lyrics_json(song, title, t))
    written["lyrics"] = paths.lyrics_json
    if len(t.analysis.beats) >= 2:
        write_musicxml(title, t, paths.out_musicxml)
        written["musicxml"] = paths.out_musicxml
        if write_pdf(title, t, paths.chart_ly, paths.chart_pdf):
            written["pdf"] = paths.chart_pdf
    else:
        print("! no beat grid detected — skipped MusicXML and PDF (they need bar positions)")

    chords = t.analysis.chords
    low = sum(1 for c in chords if c.confidence < LOW_CONFIDENCE)
    unaligned = sum(1 for l in t.lines if l.start is None)
    print(f"chart: {len(chords)} chords ({low} low-confidence, marked ?), "
          f"{len(t.lines) - unaligned}/{len(t.lines)} lyric lines timed")
    for name, path in written.items():
        print(f"  {name:<9}→ {path}")
    return written
```

- [ ] **Step 5: Add the CLI command — append to `src/songcomposer/cli.py`**

```python
@app.command()
def chart(song: str, force: bool = typer.Option(False, "--force", help="re-transcribe the chosen take")) -> None:
    """Chosen take → chords.txt, tab.txt, chart.pdf, <song>.musicxml, lyrics.json in out/<song>/."""
    from .chart import run_chart
    run_chart(song, force=force)
```

- [ ] **Step 6: Run tests** — `uv run pytest tests/test_chart.py -v` → PASS.

- [ ] **Step 7: Commit**

```bash
git add src/songcomposer/render/lyricsjson.py src/songcomposer/chart.py src/songcomposer/cli.py tests/test_chart.py
git commit -m "feat: chart stage — all guitar deliverables plus the lyrics.json video contract"
```

---

### Task 27: `run` (whole pipeline), accuracy harness, contract test, docs

**Files:**
- Create: `src/songcomposer/pipeline.py`, `src/songcomposer/accuracy.py`, `README.md`
- Modify: `src/songcomposer/cli.py`, `CLAUDE.md` (status line only)
- Test: `tests/test_pipeline.py`, `tests/test_accuracy.py`

**Interfaces:**
- Produces:
  - `pipeline.run_all(song, src, brief_file=None, brief_text=None, provider_name=None, fidelity=None, input_fn=input, provider=None) -> None` — runs every stage in order; each stage keeps its own no-op-if-done behaviour, so re-running resumes. `generate` still stops for the typed `yes`; then it asks which take.
  - `accuracy.parse_lab(text: str) -> list[tuple[float, float, str]]` (lines of `onset end label`, Harte labels — the standard MIREX `.lab` format), `accuracy.score_chords(found: list[Chord], truth: list[tuple[float, float, str]]) -> dict` with `root`, `full`, `high_confidence_wrong` (fractions of truth duration)

- [ ] **Step 1: Write the failing tests**

`tests/test_accuracy.py`:
```python
import pytest

from songcomposer.accuracy import parse_lab, score_chords
from songcomposer.models import Chord


def ch(harte, onset, dur, conf):
    root, _, q = harte.partition(":")
    return Chord(symbol=harte, harte=harte, root=root, quality=q, extensions=[], bass=None, onset=onset, duration=dur, confidence=conf)


def test_parse_lab_skips_blanks_and_no_chord():
    assert parse_lab("0.0 2.0 C:maj\n\n2.0 4.0 N\n4.0 6.0 A:min7\n") == [(0.0, 2.0, "C:maj"), (4.0, 6.0, "A:min7")]


def test_scores_are_duration_weighted():
    truth = [(0.0, 4.0, "C:maj"), (4.0, 8.0, "A:min7")]
    found = [ch("C:maj", 0, 4, 0.9), ch("A:min", 4, 2, 0.9), ch("F:maj", 6, 2, 0.2)]
    s = score_chords(found, truth)
    assert s["root"] == pytest.approx(0.75)                  # C right (4s) + A root right (2s) of 8s
    assert s["full"] == pytest.approx(0.5)                   # A:min ≠ A:min7
    assert s["high_confidence_wrong"] == pytest.approx(0.0)  # the wrong-root F was honest about its doubt
```

`tests/test_pipeline.py`:
```python
"""Contract test (BUILD-SPEC §11): run every stage with the paid/GPU parts faked and validate every stage file
against its schema. No network, no GPU, no money."""
import pytest

from songcomposer import compose, pipeline, transcribe
from songcomposer.jsonio import read_json
from songcomposer.models import (Analysis, Brief, Chord, Chosen, GlobalInfo, Note, SongSpec, SourceInfo, Subjective,
                                 TakesManifest, Transcription, Word)
from songcomposer.paths import SongPaths
from test_compose import GOOD
from test_generate import FakeProvider, mp3_bytes  # noqa: F401

GRID = [i * 0.6 for i in range(64)]


def fake_analyze(audio, cache_dir, config, ears=None, lyrics_hint=""):
    words = [Word(word=w, start=2.0 + i * 0.5, end=2.4 + i * 0.5, confidence=0.9)
             for i, w in enumerate(" ".join(GOOD.sections[0].lines).split())]
    return Analysis(
        audio_sha1="a" * 40, engines={"chords": "fake"}, beats=GRID, downbeats=GRID[::4], lyrics=words if lyrics_hint else [],
        chords=[Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None, onset=0, duration=2.4, confidence=0.9)],
        notes=[Note(pitch=64, onset=2.0, duration=0.5, confidence=0.9, stem="vocals")],
        global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=38.4),
        subjective=Subjective(genre_tags=["folk"], instrumentation=["guitar"], timbre="warm", vocal_character="breathy",
                              vocal_gender="female", production="dry", emotional_arc="", arrangement_density=[], sections=[]))


def test_whole_pipeline_contracts(root, sine_wav, mp3_bytes, monkeypatch):  # noqa: F811
    pytest.importorskip("music21")
    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(transcribe, "analyze", fake_analyze)
    monkeypatch.setattr(transcribe, "to_wav", lambda src, dst: dst.write_bytes(b"wav"))
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)
    replies = iter(["yes", "2"])
    pipeline.run_all("demo", str(sine_wav), brief_text="A song about insomnia after a breakup.",
                     provider=FakeProvider("fake", mp3_bytes), fidelity="loose", input_fn=lambda prompt: next(replies))
    p = SongPaths("demo")
    for path, model in ((p.source_json, SourceInfo), (p.analysis, Analysis), (p.brief, Brief), (p.spec, SongSpec),
                        (p.takes_json, TakesManifest), (p.chosen, Chosen), (p.transcription, Transcription)):
        model.model_validate(read_json(path))                     # raises if any stage broke its contract
    assert read_json(p.chosen)["take"] == 2
    assert p.out_mp3.exists() and p.chords_txt.exists() and p.tab_txt.exists() and p.lyrics_json.exists()
    assert read_json(p.lyrics_json)["lines"][0]["words"][0]["word"] == "The"
```

- [ ] **Step 2: Run, verify failure** — `uv run pytest tests/test_pipeline.py tests/test_accuracy.py -v` → `ImportError`.

- [ ] **Step 3: Write `src/songcomposer/accuracy.py`**

```python
"""Accuracy against a published chart in MIREX .lab format ('onset end Harte-label' per line)."""
from .chordsym import parse_harte
from .models import LOW_CONFIDENCE, Chord


def parse_lab(text: str) -> list[tuple[float, float, str]]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parse_harte(parts[2]) is not None:
            out.append((float(parts[0]), float(parts[1]), parts[2]))
    return out


def score_chords(found: list[Chord], truth: list[tuple[float, float, str]]) -> dict:
    total = sum(e - s for s, e, _ in truth) or 1.0
    root = full = confident_wrong = 0.0
    for s, e, lab in truth:
        want = parse_harte(lab)
        for c in found:
            shared = min(e, c.onset + c.duration) - max(s, c.onset)
            if shared <= 0:
                continue
            if c.root == want.root:
                root += shared
                full += shared if c.quality == want.quality else 0.0
            elif c.confidence >= LOW_CONFIDENCE:
                confident_wrong += shared
    return {"root": root / total, "full": full / total, "high_confidence_wrong": confident_wrong / total}
```

- [ ] **Step 4: Write `src/songcomposer/pipeline.py`**

```python
"""Stage: run — the whole pipeline. Every stage is a no-op if its output exists, so re-running resumes."""
from typing import Callable

from .analysis import analyze
from .brief import run_brief
from .chart import run_chart
from .compose import run_compose
from .config import load_config
from .generate import run_generate
from .ingest import run_ingest
from .jsonio import read_json, write_model
from .models import TakesManifest
from .paths import SongPaths
from .pick import run_pick


def run_all(song: str, src: str, brief_file: str | None = None, brief_text: str | None = None,
            provider_name: str | None = None, fidelity: str | None = None,
            input_fn: Callable[[str], str] = input, provider=None) -> None:
    paths = SongPaths(song)
    run_ingest(song, src)
    if not paths.analysis.exists():
        write_model(paths.analysis, analyze(paths.source_wav, paths.cache, load_config()))
    if not paths.brief.exists():
        run_brief(song, from_file=brief_file, text=brief_text)
    run_compose(song)
    run_generate(song, provider_name=provider_name, fidelity=fidelity, input_fn=input_fn, provider=provider)
    if not paths.chosen.exists():
        takes = TakesManifest(**read_json(paths.takes_json)).all_takes()
        print("Listen to the takes in", paths.takes_dir)
        for t in takes:
            print(f"  take {t.index}: {t.provider}, {t.duration_s:.0f}s")
        run_pick(song, int(input_fn("Which take? ").strip()))
    run_chart(song)
```

`pipeline.analyze` and `transcribe.analyze` are the two seams the contract test patches — keep both as module-level imports.

- [ ] **Step 5: Add the CLI commands — append to `src/songcomposer/cli.py`**

```python
@app.command()
def run(song: str,
        from_: str = typer.Option(..., "--from", help="URL, audio file or video file"),
        brief: str = typer.Option(None, "--brief", help="brief markdown file"),
        text: str = typer.Option(None, "--text", help="inline brief"),
        provider: str = typer.Option(None, "--provider")) -> None:
    """Whole pipeline. Still stops to confirm cost, and to ask which take you want."""
    from .pipeline import run_all
    run_all(song, from_, brief_file=brief, brief_text=text, provider_name=provider)


@app.command()
def accuracy(song: str,
             truth: str = typer.Option(..., "--truth", help=".lab file: 'onset end C:maj' per line"),
             which: str = typer.Option("reference", "--of", help="reference | take")) -> None:
    """Score detected chords against a published chart."""
    from pathlib import Path

    from .accuracy import parse_lab, score_chords
    from .jsonio import read_json
    from .models import Analysis, Transcription
    from .paths import SongPaths
    p = SongPaths(song)
    found = (Analysis(**read_json(p.analysis)) if which == "reference" else Transcription(**read_json(p.transcription)).analysis).chords
    s = score_chords(found, parse_lab(Path(truth).read_text(encoding="utf-8")))
    print(f"root {s['root']:.0%}   root+quality {s['full']:.0%}   wrong-but-confident {s['high_confidence_wrong']:.0%}")
```

- [ ] **Step 6: Write `README.md`** — sections: what it is (3 lines from CLAUDE.md), install (`scoop install uv lilypond`; `uv sync --extra analysis`; `.env`), the eight commands with one-line descriptions (copy from each command's docstring), the `work/` and `out/` layout (copy from BUILD-SPEC §3), "how to read confidence marks" (`?` on chords, `(n)` in tab, grey/parenthesised in the PDF, `x` noteheads and `?` text in MusicXML, `confidence: 0.0` = interpolated word in `lyrics.json`), and a pointer to `templates/first-song.md`. In `CLAUDE.md` change `Status: **design approved, not yet implemented.**` to `Status: **v1 implemented — see docs/superpowers/plans/2026-09-20-song-composer-v1.md.**`

- [ ] **Step 7: Run the full suite**

```bash
uv run pytest -m "not gpu" -v
uv run pytest -m gpu -v
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/songcomposer/pipeline.py src/songcomposer/accuracy.py src/songcomposer/cli.py tests/test_pipeline.py tests/test_accuracy.py README.md CLAUDE.md
git commit -m "feat: run pipeline, chord accuracy harness, whole-pipeline contract test, README"
```

- [ ] **Step 9: MILESTONE C — chart the real song (human step, free)**

```bash
uv run songcomposer chart <song>
```
Play along to `out/<song>/<song>.mp3` from `out/<song>/chords.txt`. The chart was derived from that exact recording, so it should not clash. For accuracy numbers on a known song (BUILD-SPEC §11), ingest it, run `analyze`, hand-write a `.lab` for its first minute from a published chart, and run `songcomposer accuracy <song> --truth file.lab`. Record results in `docs/spikes/2026-09-20-mir-stack.md` under "Real-music accuracy". This is also the evidence BUILD-SPEC §10 asks for before revisiting piano/violin parts.

---

## Spec coverage (self-review)

| BUILD-SPEC requirement | Task |
|---|---|
| §1 Ingest local audio, local video, YouTube/video links | 3 |
| §1 Brief: inline text, markdown file, one-liner | 6 |
| §1/§4 Two-ear analysis + lyrics ear, fused into `01-analysis.json` | 5, 13–20 |
| §1 Lyric + song-spec generation | 8 |
| §1 Three candidate takes, you pick one; only the chosen take is transcribed | 9–12, 21 |
| §1 Guitar output: chord sheet, tab, combined chart, PDF, MusicXML | 22, 23, 25 (combined chart = the PDF: chords + fret diagrams + melody + lyrics + tab), 24 |
| §2 #6 Fidelity prompted per run and folded into the plan | 7 (`build_request`), 9 (`ask_fidelity`) |
| §2 #8 Provider interface, Suno and ElevenLabs, `--provider` | 7, 10, 11 |
| §3 Every stage file and the full CLI surface (`ingest analyze brief compose generate pick chart run`) | 1, 3, 5, 6, 8, 9, 12, 26, 27 |
| §4 Analysis engine runs twice through one code path | 5/20 (`analyze`), 21 (second run) |
| §5 Instrument-neutral representation; confidence on every chord and note | 2 (+ test that bans guitar fields), 16, 18 |
| §5 Renderers surface low confidence | 22 (`?`), 23 (`(n)`), 24 (`x` noteheads, `?` text), 25 (grey / parenthesised), 26 (summary line) |
| §6 Python 3.11 + uv, PyTorch CUDA, ffmpeg, yt-dlp, LilyPond | 1, 3, 13, 25 |
| §7 Pre-flight gate, cost estimate, explicit confirmation, take caching, `--regen`, actual cost in `takes.json` | 7, 9 |
| §8 Reuse: `lib.mjs` client layer, `generate-music.mjs`, `generate_audiobook.py` gate/caching/sanity check | 1, 4, 10, 11 / 11 / 5, 7, 9 |
| §8 "Templates as project-local recipes" convention | 12 (`templates/first-song.md`) |
| §9 `lyrics.json` word-timed contract for the video project | 26 |
| §10 MIR library rot | Spike doc + pins in 13 |
| §10 Suno/ElevenLabs bake-off | 12 (Milestone A) |
| §11 Synthetic ground truth; schema contract per stage; accuracy vs published charts; no paid calls in tests | 13–20, 27 (`test_pipeline.py`), 27 (`accuracy`), every HTTP test uses `MockTransport` |
| §12 Open questions 1–5 | "Decisions this plan locks in" table |

### Known gaps — deliberate, revisit only if triggered

- **`GOOGLE_AI_API_KEY` fallback client is not built.** BUILD-SPEC §10 lists it as a fallback *if* OpenRouter will not pass audio to Gemini. OpenRouter documents `input_audio` support for the Gemini slugs, so this is built only if Milestone A (Task 12, step 8) shows it failing. If it does: add `clients/google_ai.py` with the same `chat_json` signature and switch on the presence of the key in `analysis/subjective.py`.
- **`subtitles.mjs` cue shaping is not ported.** `lyrics.json` carries lines and words; cue shaping for display is the video project's job on its side of the contract.
- **Picked (fingerstyle) sections get a description, not a note-for-note arpeggio tab.** BUILD-SPEC §10 is explicit that inner guitar parts transcribe only approximately from a mix; the tab says so rather than printing a confident-looking guess.
- **Live-API unknowns** (Suno `resultJson` shape, real per-request prices, Whisper accuracy on sung vocals) are resolved at Milestones A–C, with the raw-result safety net in Task 10 protecting paid output in the meantime.

## Milestones

| Milestone | After task | You can |
|---|---|---|
| **A** | 12 | Generate real songs from a reference + brief; run the Suno vs ElevenLabs bake-off |
| **B** | 20 | Get a full two-ear analysis (chords, key, tempo, structure, lyrics) of any reference |
| **C** | 27 | Get chords, tab, PDF, MusicXML and `lyrics.json` for the take you picked |
