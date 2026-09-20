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
download the URLs by hand — do not regenerate. If generation fails part-way, `takes.json` records the
full estimate as an upper-bound cost and says so in `warnings` — check your Kie.ai / ElevenLabs dashboard
for the real figure.
