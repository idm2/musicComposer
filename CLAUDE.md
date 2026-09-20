# Song Composer

A **songwriter and proof-of-concept generator**. Takes a reference song (audio file, video, or link) plus a brief, genuinely analyses the reference, writes an original song in that vein, generates human-sounding vocal audio, and produces the guitar chords and tab for what it generated.

**Read `docs/BUILD-SPEC.md` first** — it is the authoritative design. `docs/DECISIONS.md` records why each choice was made and what was deliberately rejected.

Status: **design approved, not yet implemented.**

---

## The one idea that explains the architecture

The analysis engine runs **twice** — once over the reference on the way in, once over our own generated audio on the way out. Same code path both times.

This exists because song-generation APIs are prompt-driven, not score-driven: you cannot hand one a chord chart and have it play that. So rather than writing a chart and hoping the audio matches, we generate the audio and then transcribe it. The chart is *derived from the actual recording*, so the two can never disagree.

Consequence worth internalising: **the audio generator is a swappable module.** Whichever service produces the take, the analyser transcribes it identically.

That's why generation sits behind a **provider interface** with two implementations: **Suno v5.5** (via Kie.ai — `KIE_API_KEY` and the client helpers already exist) and **ElevenLabs Music**. Suno generally sounds better; ElevenLabs has the cleaner commercial-licensing story. Bake them off on the first real song. Neither choice touches anything upstream of the generator.

## The second idea: two ears

Audio-native LLMs hear tone, timbre, vocal character and mood brilliantly — and hallucinate chords with total confidence. DSP/MIR models transcribe chords accurately from the spectrum — and cannot hear mood at all.

Never use one alone. `analysis.json` fuses both, plus Whisper for lyrics. See BUILD-SPEC §4.

**librosa is table stakes, not the engine.** It cannot distinguish Cmaj7 from Am and will not hear a suspension. It rides underneath Demucs, the chord model and basic-pitch — it does not replace them.

---

## Hard rules

### Pre-flight validation before any paid generation call — MANDATORY

Validate the spec in-script *before* calling a paid generation API. Carried over from the audiobook work and non-negotiable. The `generate` stage must refuse to spend money on unvalidated input, must print estimated cost, and must wait for explicit confirmation.

### Confidence travels with the data

Every chord and note carries a confidence value. Renderers surface low-confidence material rather than presenting a guess as fact. A chart that is quietly wrong is worse than one that admits doubt.

### Python 3.11, not 3.13

The music-analysis ecosystem lags badly and several key libraries break on 3.13. Managed with `uv`.

### Never round-trip UTF-8 files through PowerShell 5.1

`Get-Content` / `Set-Content` mangle encoding. Use Node or an editor tool. Lyrics contain smart quotes, accents and em-dashes — this will bite.

### Deliverables stay here

Output goes to this repo's `out/<song>/`. Never copied into another project's tree.

`work/` and `out/` are git-ignored (large, regenerable). The small permanent record of each song — brief, analysis,
spec, takes manifest, song sheet, charts — is archived into the versioned **`songs/<song>/`** by
`uv run python tools/archive-song.py <song>`. Do that when a song is finished; see `songs/README.md`.

---

## Environment

- **Windows 11.** Git Bash and PowerShell both available; use Unix paths in bash.
- **GPU: NVIDIA RTX 4070 SUPER, 12GB.** Demucs, Whisper, basic-pitch and chord models all run locally and fast. (Note: the sibling video project's CLAUDE.md wrongly claims no GPU exists — ignore that.)
- **ffmpeg** installed via scoop.
- **Python 3.13** is on PATH — do not use it for this project. Create a 3.11 venv.

---

## Don't build these from scratch

The sibling project `C:\dev\Logic8 Apps\Video Generator - Fal Remotion` has working code to adapt. Full inventory in BUILD-SPEC §8. The highlights:

| Source file | Why it matters |
|---|---|
| `scripts/ai-video/lib.mjs` (610 lines) | Whole client layer — `.env` loading, JSON helpers, `elevenPost()`, OpenRouter auth, job polling, arg parsing |
| `scripts/ai-video/generate-music.mjs` (90 lines) | **A working ElevenLabs Music call already exists.** Adapt for three takes; upgrade `music_v1` → `music_v2`. |
| `scripts/audiobook/generate_audiobook.py` (218 lines) | The pre-flight gate, sha1 content-hash caching and post-generation sanity check — already in Python |
| `scripts/logic8-intro/voice.mjs` (291 lines) | `wordsFromAlignment()` converts ElevenLabs character alignment into word timings — directly reusable for lyrics |

Credentials exist in that project's `.env`: `ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`, `FAL_API_KEY`, `KIE_API_KEY`. Copy what's needed rather than re-provisioning.

**Do not modify the video project from here.** It consumes this project's output (audio + word-timed `lyrics.json`) to render lyric videos, via a JSON contract — same pattern as its WhisperX relationship. No code coupling either way.

---

## Scope discipline

**v1 is guitar-only.** Chord sheet, tab, combined chart, PDF, MusicXML.

Piano, violin and other instrument parts are **deferred, not unplanned**. The internal representation is instrument-neutral, so they arrive as new renderers rather than a rewrite. The deferral is driven by a real quality risk: transcribing a generated mix gives reliable melody, bass and chords, but only approximate inner parts. Revisit once the analyser exists and accuracy is measurable — see BUILD-SPEC §10.
