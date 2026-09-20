# Song Composer

A songwriter and proof-of-concept generator. Takes a reference song (audio file, video, or link) plus a
brief, genuinely analyses the reference, writes an original song in that vein, generates human-sounding
vocal audio, and produces the guitar chords and tab for what it generated. The analysis engine runs twice
— once over the reference, once over our own generated audio — so the chart it produces is derived from
the actual recording and can never disagree with it.

See `docs/BUILD-SPEC.md` for the full design and `docs/DECISIONS.md` for why each choice was made.

## Install

```bash
scoop install uv lilypond
uv sync --extra analysis
```

`uv sync --extra analysis` pulls the ~6 GB GPU analysis stack (PyTorch/CUDA, Demucs, basic-pitch,
faster-whisper, lv-chordia, librosa). Never run a bare `uv sync` afterwards — it is exact and would
uninstall that stack; add dependencies with `uv add --optional analysis <pkg>` instead.

Copy `.env.example` to `.env` and fill in the keys (`ELEVENLABS_API_KEY`, `OPENROUTER_API_KEY`,
`KIE_API_KEY` — these already exist in the sibling video project's `.env` if you have that repo; copy
them across rather than re-provisioning). `.env` is never printed, logged or committed.

Requires Python 3.11 (not 3.13, which several analysis libraries break on) and ffmpeg on PATH (also via
scoop). `songcomposer.toml` holds user-editable defaults (default provider, model slugs).

## Commands

Every stage writes an inspectable file to `work/<song>/` or `out/<song>/`, and re-running a stage is a
no-op once its output exists (see "Resuming" below).

| Command | What it does |
|---|---|
| `songcomposer ingest <song> --from <url\|path>` | Reference -> `work/<song>/00-source.wav` + `00-source.json`. |
| `songcomposer analyze <song> [--ears ...]` | Two-ear analysis of the reference -> `work/<song>/01-analysis.json`. |
| `songcomposer brief <song> --from <file.md> \| --text "..."` | Your brief -> `work/<song>/02-brief.json`. |
| `songcomposer compose <song> [--force]` | Brief + analysis -> lyrics and song spec (`work/<song>/03-spec.json`). Edit the file freely afterwards. |
| `songcomposer generate <song> [--provider] [--fidelity] [--regen]` | Spec -> takes. Validates, prints estimated cost, and waits for an explicit `yes` before spending. |
| `songcomposer pick <song> --take N` | Choose a take -> `05-chosen.json` and `out/<song>/<song>.mp3`. |
| `songcomposer chart <song> [--force]` | Chosen take -> `chords.txt`, `tab.txt`, `chart.pdf`, `<song>.musicxml`, `lyrics.json` in `out/<song>/`. |
| `songcomposer run <song> --from <url> [--brief file \| --text ...] [--provider]` | Whole pipeline. Still stops to confirm cost, and to ask which take you want. |
| `songcomposer accuracy <song> --truth <file.lab> [--of reference\|take]` | Score detected chords against a published chart (MIREX `.lab` format). |

Run any command with `--help` for its full option list.

### Resuming (`run` and re-running stages)

Every stage keeps its own no-op-if-done behaviour, so `run` — and re-running an individual command —
resumes rather than repeating expensive work:

- `ingest` skips re-downloading/re-encoding if `00-source.wav`/`.json` already exist (`--force` overrides).
- `run` itself skips the objective+subjective `analyze` call if `01-analysis.json` exists, and skips
  `brief` if `02-brief.json` exists.
- `compose` refuses to overwrite an existing `03-spec.json` unless `--force` (so your hand-edits to the
  lyrics survive a re-run).
- `generate` is a no-op for a request it already generated takes for (same spec + fidelity + provider);
  `--regen` forces fresh, paid takes. **This is the one stage that still stops for the cost estimate and
  a typed `yes` even under `run` — there is no bypass flag.**
- `pick` is cheap (a copy + a JSON write) and simply redoes it if asked again.
- `chart` re-runs the (fast, deterministic) renderers every time, but skips the expensive part — the
  `analyze` re-run over the chosen take — if `06-transcription.json` already matches the chosen take.

## Layout

```
work/<song>/
  00-source.wav          ingest: YouTube / video / audio file → normalised audio
  00-source.json         provenance: origin URL/path, duration, sample rate
  01-analysis.json       the two ears, fused
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
  lyrics.json            word-level timings (feeds the video project)
```

`work/` and `out/` are git-ignored — they hold big, regenerable audio and caches. The permanent,
versioned record of a finished song lives in `songs/<song>/`; see `songs/README.md` for what goes there
and how to write a brief that works, and run `uv run python tools/archive-song.py <song>` to copy a
song's small artefacts out of `work/`/`out/` once you're done with it. `templates/first-song.md` is the
step-by-step recipe for producing your first song and running the Suno/ElevenLabs provider bake-off.

## How to read confidence marks

Every chord and note carries a confidence value (0.0–1.0); anything below `LOW_CONFIDENCE` (0.5) is
surfaced, never silently presented as fact. What that looks like in each deliverable:

- **`chords.txt`** — a low-confidence chord gets a trailing `?` on its symbol (e.g. `Am7?`). The header
  line states how many of the chart's chords are low-confidence. If no lyric line could be aligned at
  all, the chords are printed under `[Chords — no lyric timing detected, not placed against words]`
  instead of being silently dropped; a line with no timing prints as `<line text>   (timing not
  detected — chords not placed)`.
- **`tab.txt`** — a low-confidence melody note is wrapped in parentheses, e.g. `(3)` instead of `3`, on
  the tab grid. The melody section header states how many transcribed notes are shown versus how many
  were dropped (outside a sung line's span, or below the note-confidence floor). Sections state plainly
  when nothing could be read from the audio: `not detected in the audio` (section never sung), `strums
  detected but no consistent pattern — listen and choose your own`, `no guitar-range accompaniment
  detected`, or (if no beat grid was found at all) `No beat grid was detected — tab cannot be laid out.
  See chords.txt.`
- **`chart.pdf`** (engraved by LilyPond) — a low-confidence chord name is printed in grey; a
  low-confidence melody note is parenthesised. The PDF's tagline says so explicitly: "Grey chord names
  and parenthesised notes are low-confidence detections."
- **`<song>.musicxml`** — a low-confidence melody note gets an `x` notehead instead of a normal one; a
  low-confidence (or unrecognised-quality) chord symbol has a `?` text expression placed above it. Both
  are visible in MuseScore/Guitar Pro without needing to inspect the XML.
- **`lyrics.json`** — each word carries its own `confidence`; a word that could not be matched to the
  audio and was only interpolated between its aligned neighbours gets `confidence: 0.0`. A line that
  could not be aligned to the recording at all is omitted from `lines` and listed instead under
  `unaligned_lines`.
- **`songcomposer accuracy`** — reports `root` (root-note-only accuracy), `full` (root+quality accuracy)
  and `high_confidence_wrong` (fraction of the chart's duration where a *confident* detection was simply
  wrong) against a hand-written `.lab` chart, all as durations-weighted fractions.

A chart that admits doubt is the design intent — never trust a chord name or note that isn't marked, but
also never assume the marked ones are worthless; they are usually right, just not certain.
