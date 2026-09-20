# Song Composer — Build Spec

**Status:** design approved, not yet implemented
**Date:** 2026-09-20
**Sibling project:** `C:\dev\Logic8 Apps\Video Generator - Fal Remotion` (source of reusable code — see §8)

---

## 1. What this is

A **songwriter and proof-of-concept generator**. You give it a reference (audio file, video, or link) plus a brief, and it:

1. Genuinely analyses the reference — chords, key, tempo, structure, timbre, vocal character
2. Writes an original song in that vein — lyrics, structure, style
3. Generates real, human-sounding vocal audio
4. Produces the **guitar chords and tab** for what it just generated

It is a **CLI you operate yourself**, not a chat workflow and not a hosted app. Each stage is a separate command that writes an inspectable file, so any stage can be re-run without redoing the ones before it.

### v1 scope

- Reference ingestion: local audio, local video, YouTube/video links
- Brief ingestion: inline text, markdown file, or a one-liner
- Full two-ear analysis (see §4)
- Lyric + song-spec generation
- Three candidate audio takes, you pick one
- Guitar output: chord sheet, tab, combined chart, PDF, MusicXML

### Deferred (explicitly out of v1)

- **Alternate instrument parts** — piano, violin, bass, lead sheets
- Multi-instrument PDF sheet music beyond guitar
- Any UI, hosting, or multi-user concern

Deferred does **not** mean unplanned. The internal note representation is instrument-neutral from day one (§5), so adding a piano or violin renderer later is a new output module, not a rewrite. See §10 for the known quality risk that drove this deferral.

---

## 2. Locked decisions

These were settled in the design conversation. See `DECISIONS.md` for the reasoning behind each.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Separate project** from the video generator | Python + CUDA + engraving toolchain vs Node/Remotion. Different runtimes, different lifecycles. Follows the existing WhisperX sibling pattern. |
| 2 | **CLI, staged** | Each stage independently runnable and inspectable. The analysis report is valuable on its own. |
| 3 | **Audio first, then transcribe our own output** | ElevenLabs Music is prompt-driven, not score-driven — it cannot play a chart we hand it. So we generate, then run our own analyser over the result. The tab is therefore *derived from the actual recording* and is guaranteed to match it. |
| 4 | **Two ears, not one** | DSP models transcribe chords accurately but can't hear mood. Audio LLMs describe mood brilliantly and hallucinate chords. Neither alone is honest. |
| 5 | **Three takes per song** | Vocal character and arrangement vary widely between generations at the same prompt. Three gives a real choice at modest cost. Only the chosen take is transcribed. |
| 6 | **Fidelity is prompted per run** | The CLI asks how closely to track the reference at generation time and folds the answer into the composition plan. Not a config setting. |
| 7 | **Guitar default, other instruments later** | Guitar is the v1 renderer. The representation underneath is instrument-neutral. |
| 8 | **Audio generation sits behind a provider interface — Suno *and* ElevenLabs** | Decision #3 already made the generator swappable. Suno v5.5 is reachable via Kie.ai using a key and client helpers that already exist. Bake the two off on song one. See `DECISIONS.md` → "Suno reconsidered". |

---

## 3. Pipeline and data contracts

Stages write files to disk. Each is re-runnable in isolation.

```
work/<song>/
  00-source.wav          ingest: YouTube / video / audio file → normalised audio
  00-source.json         provenance: origin URL/path, duration, sample rate
  01-analysis.json       the two ears, fused (see §4)
  02-brief.json          your brief: inline text, .md file, or one-liner
  03-spec.json           lyrics, section map, style directives, fidelity choice
  04-takes/
    take-1.mp3
    take-2.mp3
    take-3.mp3
    takes.json           prompt + composition plan used, cost actually incurred
  05-chosen.json         which take you picked
  06-transcription.json  analysis re-run over the chosen take

out/<song>/
  <song>.mp3             the chosen take
  chords.txt             lyrics with chords above the line
  tab.txt                ASCII tab with picking/strumming notation
  chart.pdf              engraved chart
  <song>.musicxml        opens in MuseScore / Guitar Pro
  lyrics.json            word-level timings (feeds the video project — §9)
```

**Folder convention deliberately mirrors the video project** (`input/<project>/` → `output/<project>/`) so it feels familiar to work in.

### CLI surface

```
songcomposer ingest   <song> --from <url|path>
songcomposer analyze  <song>
songcomposer brief    <song> --from <file.md> | --text "..."
songcomposer compose  <song>              # → lyrics + spec
songcomposer generate <song>              # → 3 takes (prompts for fidelity, confirms cost)
songcomposer pick     <song> --take 2
songcomposer chart    <song>              # → chords, tab, PDF, MusicXML
songcomposer run      <song> --from <url> --brief <file.md>   # whole pipeline
```

---

## 4. The analysis engine — the core

This is the heart of the project and **it runs twice**: once over the reference on the way in, once over our own generated audio on the way out. Same code path both times. That symmetry is what makes decision #3 work.

### Objective ear — local, GPU, free

Runs on the RTX 4070 SUPER (12GB).

| Component | Purpose |
|---|---|
| **Demucs** | Stem separation — run *first*. Splitting out drums and vocals is the single biggest accuracy gain available; chord detection on a full mix is fighting percussion. |
| **Chord recognition** | Timestamped chord sequence, run over the harmonic stems |
| **Beat / downbeat / structure** | Tempo, time signature, section map (intro/verse/chorus/bridge) |
| **basic-pitch** | Polyphonic note transcription → MIDI |
| **librosa / essentia** | Key estimation, loudness, spectral and timbral descriptors |

> **Note on librosa:** it is table stakes, not the engine. It gives loading, chroma, tempo and a rough key guess. Asked for a chord chart on its own it returns a smeared triad-only approximation that cannot distinguish Cmaj7 from Am and will not hear a suspension. It rides along underneath the components above rather than replacing any of them.

### Subjective ear — Gemini audio via OpenRouter

Sends the actual audio and returns structured JSON:

- Timbre and instrumentation
- Vocal character and delivery (grain, register, phrasing, restraint vs belt)
- Production style (space, compression, era)
- Arrangement density over time
- Emotional arc

### Lyrics ear — Whisper, local

Words with timings, from the reference and later from our own output.

All three fuse into a single `01-analysis.json`.

---

## 5. Internal representation

One instrument-neutral structure sits between analysis and output:

- Notes (pitch, onset, duration, confidence)
- Chords (symbol, root, quality, extensions, onset, duration, confidence)
- Sections (label, start, end)
- Lyrics (word, start, end)
- Global (key, tempo, time signature, loudness)

Guitar tab, chord sheets, and later piano/violin parts are **renderers over this structure**. Nothing instrument-specific leaks upstream of the renderers.

**Confidence is carried on every chord and note.** Renderers surface low-confidence material rather than silently presenting a guess as fact.

---

## 6. Tech stack

- **Python 3.11** — *not* 3.13. The music-analysis ecosystem lags badly and several key libraries break on 3.13. Managed with `uv`.
- **PyTorch + CUDA** — Demucs, basic-pitch, Whisper all GPU-accelerated
- **ffmpeg** — already installed (scoop), handles video→audio extraction
- **yt-dlp** — link ingestion
- **LilyPond or MuseScore CLI** — PDF engraving (selection is a plan-stage spike)
- **Audio generation, behind a provider interface:**
  - **Suno v5.5 via Kie.ai** — `POST /api/v1/generate`. Custom mode takes our own lyrics, style and title; returns multiple variations per request.
  - **ElevenLabs Music** — `music_v2`, `composition_plan` for per-section control, supports inpainting.

### Credentials

| Key | Status | Used for |
|---|---|---|
| `ELEVENLABS_API_KEY` | ✅ exists in video project `.env` | Music generation, STT |
| `OPENROUTER_API_KEY` | ✅ exists | Gemini audio analysis, lyric writing |
| `KIE_API_KEY` | ✅ exists | Suno v5.5 music generation |
| `GOOGLE_AI_API_KEY` | ⚠️ may be needed | Fallback if OpenRouter does not pass audio through to Gemini |

**At most one new credential.** Everything else — Demucs, chord models, basic-pitch, Whisper, LilyPond — is free, local, and runs on hardware already owned. The capability gap here was never connectors.

---

## 7. Cost and safety gates

**Pre-flight validation before any paid call is mandatory.** This is an established standing rule from the audiobook work and applies unchanged here: the spec must pass validation before stage `generate` is permitted to call ElevenLabs.

- `generate` prints estimated cost for three takes and waits for explicit confirmation
- Generated takes are cached; re-running `generate` without `--regen` is a no-op
- Actual cost incurred is recorded in `04-takes/takes.json`

---

## 8. Reuse from the video project

Do **not** build these from scratch. Source: `C:\dev\Logic8 Apps\Video Generator - Fal Remotion`.

### High value — adapt directly

| File | Lines | What to take |
|---|---|---|
| `scripts/ai-video/lib.mjs` | 610 | The whole client layer. `loadEnv()` / `requireEnv()` for `.env` parsing, `readJson`/`writeJson`/`exists`, `downloadTo`, `projectPath`, `parseArgs`, `sleep`, `elevenPost()` for ElevenLabs, OpenRouter auth + chat helpers. **Also `kiePost()` / `kieGet()` / `pollUnifiedJob()` already pointed at `https://api.kie.ai/api/v1` — that is the Suno route, auth and polling already solved.** The single biggest head start. |
| `scripts/ai-video/generate-music.mjs` | 90 | **A working ElevenLabs Music call already exists.** Request body shape, length derivation, `--regen` caching flag, manifest patching. Adapt for three takes and upgrade `model_id` from `music_v1` to `music_v2`. |
| `scripts/audiobook/generate_audiobook.py` | 218 | **The Python patterns we want, in the language this project uses.** `validate_chunk()` / `preflight()` — the mandatory pre-paid-call gate. `chunk_key()` sha1 content-hash caching so edits only regenerate what changed. `sanity_check()` post-generation verification. Port the structure wholesale. |
| `scripts/logic8-intro/voice.mjs` | 291 | `wordsFromAlignment()` — converts ElevenLabs character-level alignment into word timings. Directly reusable for lyric timing. Also the `/text-to-speech/{id}/with-timestamps` call shape and its own `preflight()` gate. |

### Useful — study the pattern

| File | What to take |
|---|---|
| `scripts/subtitles.mjs` | Word timings → cue shaping (line length, reading speed, break at sentence then clause). Applies to lyric sheets and any karaoke output. |
| `scripts/transcribe-calls.mjs` | ElevenLabs speech-to-text usage pattern. |
| `templates/*.md` | The "templates as project-local recipes" convention — reusable task recipes kept as markdown and treated as authoritative. Worth adopting here. |
| `CLAUDE.md` | Structure and tone for the project instruction file. |

### Also available

`.env` in the video project already holds `ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`, `FAL_API_KEY`, `KIE_API_KEY`. Copy the two needed rather than re-provisioning.

---

## 9. Relationship to the video project

Same shape as the existing WhisperX sibling relationship: **separate project, JSON contract.**

Song Composer emits `out/<song>/<song>.mp3` plus `lyrics.json` with word-level timings. The video project can consume those to render a lyric video through its existing caption compositions. No code coupling in either direction.

Note: `CLAUDE.md` in the video project references WhisperX at `C:\Dev\WhisperX Subtitles\`, which **does not exist at that path**. Do not assume it is available.

---

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **MIR library rot** — several chord/structure libraries are semi-abandoned and break on modern Python/numpy | High | Python 3.11. Spike the analysis stack and confirm each library works *before* committing to it in the plan. |
| **Chord accuracy on dense mixes** | Medium | Demucs stem separation first. Report per-chord confidence rather than presenting guesses as fact. |
| **Transcription quality for non-guitar parts** | High | **This is why multi-instrument output is deferred.** From a generated mix: melody reliable, bass reliable, chord chart reliable; piano and guitar parts approximate; anything buried in the mix poor. Multi-instrument PDFs are achievable as lead sheets and playable parts, not faithful transcriptions. Revisit once the analyser exists and real accuracy is measurable. |
| **Generated vocals are a lottery** | Medium | Three takes, user picks. |
| **OpenRouter may not pass audio through to Gemini** | Medium | Fallback to a direct Google AI Studio key. |
| **Harmonic drift in generated audio** | Medium | Accepted consequence of decision #3 — we transcribe what was played rather than dictating it. The chart is always *truthful*, even when the harmony isn't what we'd have chosen. |

---

## 11. Testing strategy

**Manufacture ground truth.** Synthesise audio from a MIDI progression we wrote ourselves, in a known key at a known tempo, then assert the analyser recovers exactly that. Real songs have no answer key; synthetic ones do.

- Unit: each analysis component against synthetic fixtures with known answers
- Accuracy: measure against published charts for a handful of well-known songs
- Contract: every stage's JSON output validated against a schema
- No paid API calls in the test suite — ElevenLabs responses fixtured

---

## 12. Open questions for the implementation plan

1. Which chord-recognition library survives the spike (Chordino vs a PyTorch model vs madmom)
2. LilyPond vs MuseScore CLI for engraving
3. Whether to port `lib.mjs` to Python or run a thin Node layer for the API calls only
4. Whether analysis caching should be content-hashed like the audiobook chunk cache (probably yes)
5. **Suno vs ElevenLabs bake-off** — same lyrics, same style brief, listened side by side on the first real song. Decides the default provider. Licensing may override quality if songs ever enter client deliverables.

---

## 13. Next step

Implementation plan. This spec is the input to it.
