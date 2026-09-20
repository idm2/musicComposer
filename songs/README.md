# songs/ — the permanent record of every song made with this tool

`work/<song>/` and `out/<song>/` are git-ignored: they hold big regenerable audio and caches. Everything in
**this** folder is versioned, so a song's inputs, measurements and deliverables survive a `git clean`, a disk wipe,
or a year of not thinking about it.

One folder per song. What goes in it:

| File | What it is |
|---|---|
| `brief.source.md` | the brief in your own words — the thing worth keeping |
| `00-source.json` | provenance of the reference: URL/path, title, duration, sha1 |
| `01-analysis.json` | the two-ear analysis of the reference |
| `adhoc-objective.json` | (early songs only) measurements taken before the analysis stage existed |
| `02-brief.json`, `03-spec.json` | the pipeline's brief and the song spec: lyrics, sections, style prompt |
| `takes.json` | what was actually sent to the generator, and what each take cost |
| `<song>.md` | **the deliverable** — the full song sheet: tone, harmony, voicings, tab, lyrics, generator blocks |
| `chords.txt`, `tab.txt`, `<song>.musicxml`, `lyrics.json` | (from Task 26 onward) transcribed from the take you picked |

Audio is not versioned — takes live in `work/<song>/04-takes/` and the chosen take in `out/<song>/<song>.mp3`.
`takes.json` records each take's provider, id, duration and cost so a lost take can be traced or regenerated.

## Making a new song

The recipe is `templates/first-song.md`. The short version, with a reference:

```bash
uv run songcomposer ingest   <song> --from "<youtube url | audio file | video file>"
uv run songcomposer analyze  <song>
uv run songcomposer brief    <song> --from songs/<song>/brief.source.md
uv run songcomposer compose  <song>          # then EDIT work/<song>/03-spec.json — the lyrics are yours
uv run songcomposer generate <song> --provider suno --fidelity loose --takes 3   # prints cost, waits for a typed "yes"
uv run songcomposer pick     <song> --take N
uv run songcomposer chart    <song>          # available from Task 26
```

Or, with no reference — a song from a brief alone — skip `ingest`/`analyze`, omit `--fidelity`
(it's meaningless with nothing to track; `generate` fixes it to `loose` and says so), and `run` will
compose from the brief on its own:

```bash
uv run songcomposer run <song> --brief songs/<song>/brief.source.md --provider suno --takes 3
```

`--takes` (1-6, default 3) controls how many takes any provider makes; if you omit it, `generate`/`run`
asks. ElevenLabs makes exactly that many; Suno returns tracks in pairs, so an odd count still pays for
the next even one (the cost confirmation says so before you spend anything).

Then copy the artefacts here:

```bash
uv run python tools/archive-song.py <song>
```

## Writing a brief that works

`songs/in-the-flow/brief.source.md` is the worked example. What made it work:

- **Say what it is about in plain words**, including the feeling, not just the topic.
- **Name what to borrow and what must differ.** ("Same sound-world; completely different words, melody and chord movement.")
- **Give the measured facts about the reference** (key, tempo, the actual chord loop) — the writer uses them and
  will not invent them.
- **Say which ideas are primary and which are seasoning.** The in-the-flow brief says: love song first, universe
  and flow imagery second, never preachy. That single sentence did more than any other.
- **Name the instrument you want to play.** Asking for fingerpicked acoustic guitar is what makes the output playable.
- **Ask for structure** if you care about it (intro → hook → verse → pre-chorus → chorus → bridge → final chorus → outro).

## Useful things learned so far

- Generators follow **key, tempo, mood and structure** well; they do **not** play a chord chart you hand them.
  That is why the tool transcribes the take you pick. The chords in a `<song>.md` are the song *as written*.
- `--fidelity loose` is usually right when the reference is a **live** recording: `medium`/`close` fold the heard
  instrumentation into the prompt, and for a live clip that includes "sampled crowd noise".
- Suno returns **2 tracks per request**, so 3 takes costs 2 requests and yields 4 tracks. ~$0.12 total.
- Ask for a tempo a little faster than you want: the four in-the-flow takes came back 73–86 BPM against 88 requested.
