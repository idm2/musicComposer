# tools/

Small operator scripts that sit beside the `songcomposer` CLI. Run them with `uv run python tools/<name>.py`.

| Script | What it does |
|---|---|
| `archive-song.py <song>` | Copies a song's small permanent artefacts out of the git-ignored `work/`/`out/` into the versioned `songs/<song>/`. Scrubs callback URLs from `takes.json`. Run it after finishing a song. |
| `measure-reference.py` | Measures a reference: Demucs stems, then beat/downbeat tracking, Krumhansl key estimation, and chord recognition on both the harmonic stem and the full mix. Writes `adhoc-objective.json`. |
| `measure-takes.py` | Measures every generated take: tempo, key, and which chords it actually used, by seconds. Useful for choosing a take on evidence as well as by ear. |

## Why the two `measure-*` scripts exist

They were written before the analysis pipeline stages existed, to get real numbers for the first song. They call the
installed models directly and report **no confidence values**.

Once the pipeline is complete, prefer the real stages, which do carry confidence:

```bash
uv run songcomposer analyze <song>      # the reference, both ears, cached
uv run songcomposer chart  <song>       # the take you picked, transcribed
```

Keep the scripts for quick one-off questions ("what tempo did that take actually come back at?") and for measuring
audio that is not part of a song project. Both are hard-coded to a path near the top — edit it before running.
