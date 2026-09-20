# Song Composer — Design Conversation & Decision Record

**Date:** 2026-09-20
**Context:** Design session held in the `Video Generator - Fal Remotion` project before this project existed. This file preserves that conversation so nothing is lost in the move.

---

## How this started

The opening question was: *"How good is this app we built? I want to write a song, and it needs to sound absolutely beautiful… I will provide context and an example video/song, and you will need to understand the tone, chords etc. and replicate something similar with guitar chords and tab. Are we capable, or do we need more connectors like Suno?"*

It grew from there into a full project.

---

## Findings from the capability audit

Checked rather than assumed. Results:

### ElevenLabs Music is already connected

Creator tier, active, `music_v2` available. `compose_music` generates full songs **with vocals** from either a prompt or a structured `composition_plan` giving section-by-section control over text, duration, and positive/negative style tags. Also available: `isolate_audio` (stem separation), `speech_to_text`, `video_to_music`.

**Suno is not needed.** The Suno-equivalent was already wired up.

### The machine has a GPU

**NVIDIA GeForce RTX 4070 SUPER, 12GB.**

The video project's `CLAUDE.md` states *"no NVIDIA GPU. CPU rendering only."* **That is wrong** and has likely been costing time on every Remotion render in that repo. It also means local audio analysis — Demucs, Whisper, chord models — runs in seconds rather than minutes. This fact underpins the whole "do the analysis locally and free" approach.

### Credentials already provisioned

`.env` in the video project holds `ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`, `FAL_API_KEY`, `KIE_API_KEY`.

### Not available

`C:\Dev\WhisperX Subtitles\` — referenced by the video project's `CLAUDE.md` as the upstream transcript source — **does not exist at that path.** Don't plan around it.

---

## The central constraint

**Claude has no audio input.** Images reach the model; audio does not. So "Claude listens to the reference song" is never the mechanism. Something else listens and reports back in text, and *what we choose as the ears determines the quality of everything downstream.*

### The trap that shaped the architecture

The obvious move — hand the song to an audio-native LLM — is wrong on its own.

- **Audio LLMs** (Gemini, GPT-4o-audio) genuinely hear. On tone, instrumentation, vocal character, production, emotional arc they are excellent. Ask one *"what are the chords?"* and it returns a plausible, confident, **wrong** progression. It pattern-matches vibes into chord names rather than transcribing. For a tool whose entire output is a chart you will play, that failure is fatal and invisible.
- **DSP / MIR models** are the reverse: a timestamped chord sequence genuinely derived from the spectrum, plus key, BPM, downbeats, sections — but no ability to tell you the song feels like regret at 2am.

Hence **two ears**, fused. This is the most important idea in the design.

---

## Questions asked and answered

| Question | Answer | Consequence |
|---|---|---|
| Who operates it? | **A CLI you run yourself** | Staged commands, each independently runnable; no chat-in-the-loop requirement |
| How should audio and tab relate? | **Generate audio first, then transcribe our own output** | The analysis engine runs twice. Chart is derived from the recording, so they can never disagree. |
| Where does content come from? | **All of it** — a brief, a YouTube link, an uploaded video/audio file, and an MD file | Ingestion accepts four source types |
| Guitar assumptions? | **Everything** — a chords version, a tab version, and a mix. Plus eventually PDF sheet music for other instruments. | Multiple renderers over one neutral representation |
| How many takes? | **Three, then pick** | Generation is a lottery; only the chosen take gets transcribed |
| Fidelity to the reference? | **Ask on each generation, answer in the prompt** | Interactive prompt at generate time, not a config setting |
| Build order? | **"Do everything at once"** | One project, one spec, complete pipeline — internally staged, not drip-fed over weeks |
| Non-guitar parts? | **Defer** | Guitar-only v1; representation stays instrument-neutral so it's a renderer later, not a rewrite |
| Own project or same repo? | **New project** (asked three times, answered the same three times) | This repo |

---

## Why a separate project

Settled decisively. Four reasons:

1. **Runtime mismatch.** The video project is Node/React/Remotion. This is Python + CUDA + a music-engraving toolchain. Bolting a multi-gigabyte GPU inference stack and a score engraver onto a video renderer's dependency tree makes both harder to reason about.
2. **The instruction files would fight each other.** The video project's `CLAUDE.md` is tightly scoped to compositions, render workflows and portrait-layout rules. A songwriting pipeline's rules have nothing to say to any of that.
3. **Lifecycle.** The video project was mid-flight on `feature/logic8-intro-video` with a large dirty working tree. This wants its own history and pace.
4. **The pattern already exists.** WhisperX is a separate project that hands the video project a JSON contract. Song Composer is the same shape.

---

## The scope-growth moment

Partway through, the ask expanded from "write a song with chords" to a pipeline spanning ingestion, MIR analysis, generation, transcription, **and music engraving** — four distinct toolchains, none of them Node. That expansion is what made the separate-project decision unambiguous rather than merely sensible.

It also triggered a decomposition flag. Resolution: keep it as **one project and one spec covering the complete pipeline**, but stage it internally and cut multi-instrument output from v1. Those two choices together make "everything at once" genuinely achievable.

---

## Standing rules carried over

From accumulated project memory. These apply here unchanged.

### Pre-flight validation before any paid generation call — MANDATORY

Validate content in-script *before* spending money on a generation API. Established during the audiobook work and non-negotiable. In this project it gates the `generate` stage.

### Deliverables stay in the project's own output folder

Renders and artefacts live in this repo's `out/<song>/`. Never copied into another project's tree.

### Never round-trip UTF-8 files through PowerShell 5.1

`Get-Content` / `Set-Content` mangle encoding. Use Node, or an editor tool, for any file with non-ASCII content. Relevant here — lyrics will contain smart quotes, accents and em-dashes.

### ElevenLabs quirks

- A **partial** `voice_settings` object is silently ignored — send all fields or none
- The API default already equals speed 1.08

### OpenRouter

Billing is per-character for TTS; watch for SSML handling and chat-model drift when using it for generation rather than chat.

---

## Deliberately rejected

| Option | Why not |
|---|---|
| ~~**Suno connector**~~ | **Reversed — see "Suno reconsidered" below.** The original reasoning ("ElevenLabs already does this") was too quick. |
| **Write the chart first, then make audio match it** | ElevenLabs Music is prompt-driven, not score-driven. It will drift. You'd get a beautiful track and an accurate tab that are *different songs* — you'd play along and it would clash. |
| **Render MIDI from our own chart for exact match** | Matches the tab perfectly but sounds like MIDI. Fails the "absolutely beautiful" requirement outright. |
| **librosa as the chord engine** | Smeared triad-only output. Cannot distinguish Cmaj7 from Am, cannot hear suspensions. It's a supporting library, not the answer. |
| **Audio LLM alone for analysis** | Hallucinates chords with total confidence. See "the trap" above. |
| **Python 3.13** | The MIR ecosystem lags badly. Several key libraries break. Use 3.11. |
| **Spotify Audio Features API** | Deprecated for new applications. |

---

## Suno reconsidered — and it's available now

The initial answer to *"do we need Suno?"* was "no, ElevenLabs Music already does this." That was correct that there's no **gap**, but wrong to treat the question as closed. Quality is a separate question from capability.

### What the check turned up

- **Suno has no official public API.** Access is via resellers.
- **Kie.ai resells Suno**, currently at **v5.5**, endpoint `POST /api/v1/generate`. Supports a *custom mode* (you supply your own lyrics, style and title) and a *simple mode* (prompt only), and **returns multiple variations per request**.
- **`KIE_API_KEY` already exists** in the video project's `.env`, and `lib.mjs` already has `kiePost()`, `kieGet()` and `pollUnifiedJob()` pointed at `https://api.kie.ai/api/v1`.

So Suno is reachable **today, through a connector already wired, with auth and polling helpers already written.** Zero new credentials.

### Why this doesn't disturb the design

It doesn't, and that's the point. Decision #3 — generate first, then transcribe our own output — already made the audio generator a **swappable module**. Whichever service produces the take, the analyser transcribes it identically. Suno is a provider choice, not an architecture change.

Suno's *custom mode* accepting our own lyrics matters: the songwriting stays ours either way. And its multiple-variations-per-request behaviour lines up neatly with the three-takes decision.

### The real trade-off

| | Suno v5.5 (via Kie.ai) | ElevenLabs Music |
|---|---|---|
| Song/vocal quality | Generally considered the leader | Good, strong production |
| Access | Third-party reseller, no official API | First-party, already wired |
| Section-level control | Custom mode: lyrics, style, title | `composition_plan` — per-chunk text, duration, style tags |
| Commercial licensing | Murkier, especially via a reseller | Trained on licensed catalogue; cleaner for client work |
| Inpainting / revision | — | Supported |

**Licensing is the one that deserves real thought.** These are PoCs today, but IDM2/Logic8 produce client deliverables. If a generated song ever ends up in client work, the provenance of the model that made it becomes a live question. ElevenLabs has the cleaner story there; Suno via a reseller has the better sound.

### Resolution

Build a **provider interface** with both behind it. Run a bake-off on the first real song — same lyrics, same style brief, listen side by side. Let the CLI take `--provider suno|elevenlabs`, defaulting to whichever wins, and keep the other available for when licensing matters more than polish.

---

## The bottom line on connectors

The original question was *"what other connectors do we need?"*

**At most one, possibly zero.** A direct Google AI Studio key, and only if OpenRouter turns out not to pass audio through to Gemini. Everything else — Demucs, chord recognition, basic-pitch, Whisper, LilyPond — is free, local, and runs on hardware already owned.

The capability gap was never connectors. It was that nobody had pointed the right models at the problem.
