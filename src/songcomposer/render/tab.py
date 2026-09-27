"""tab.txt — chord shapes, a strumming (or picking) read per section, and the vocal melody as guitar tab."""
from ..models import LOW_CONFIDENCE, Transcription
from . import guitar
from .chordsheet import label
from .leadsheet import MELODY_MIN_CONFIDENCE, melody_selection_note, selected_melody
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
        frets = guitar.voicing(c)
        if not frets:
            out.append(f"{label(c):<11}(no shape in dictionary — work it out from the chord name)")
            continue
        out.append(f"{label(c):<11}{('.' if any(len(f) > 1 for f in frets) else '').join(frets)}")
    return out


def _rhythm_block(t: Transcription, beatmap: BeatMap) -> list[str]:
    out = ["RIGHT HAND (read from the recording; D = down, U = up, eighth-note grid)"]
    a = t.analysis
    strums = guitar.strum_onsets(a.notes)
    for s in a.sections:
        if s.end <= s.start:
            # zero-length: the generator never sang this section. Never let a strum in a
            # neighbouring bar be misread as "detected" for a section that doesn't exist in the audio.
            out.append(f"{s.label}: not detected in the audio")
            continue
        first = beatmap.sixteenth(s.start) // beatmap.bar16
        end16 = beatmap.sixteenth(s.end)
        last = (end16 - 1) // beatmap.bar16 + 1                    # ceil(end16 / bar16): include the bar s.end falls in
        patterns = [guitar.bar_pattern(strums, bar * beatmap.bar16, beatmap) for bar in range(first, max(first + 1, last))]
        pattern = guitar.common_pattern(patterns, min_hits=guitar.MIN_PATTERN_HITS, min_coverage=guitar.MIN_PATTERN_COVERAGE)
        strums_here = any(p.strip("-") for p in patterns)
        picked = sum(1 for n in a.notes if n.stem == "other" and s.start <= n.onset < s.end)
        if pattern:
            out.append(f"{s.label}: strummed   | {pattern} |")
        elif strums_here:
            out.append(f"{s.label}: strums detected but no consistent pattern — listen and choose your own")
        elif picked:
            out.append(f"{s.label}: picked / arpeggiated — no full strums detected; arpeggiate the chord shapes")
        else:
            out.append(f"{s.label}: no guitar-range accompaniment detected")
    return out


def _melody_block(t: Transcription, beatmap: BeatMap) -> list[str]:
    melody, kept, total = selected_melody(t)
    if any(line.start is not None for line in t.lines):
        header = "MELODY (vocal line in guitar range; (n) = low-confidence note)"
    else:
        # no lyric line has timing at all: we cannot clip to sung spans, so show every note that
        # clears the confidence floor and say plainly that instrumental-section artefacts may be included.
        header = ("MELODY (vocal line in guitar range; (n) = low-confidence note; "
                   "no lyric timing detected — notes not clipped to sung spans)")
    dropped = total - kept
    out = [header, f"{melody_selection_note(kept, total)} "
                    f"({dropped} outside sung spans or below confidence {MELODY_MIN_CONFIDENCE})."]
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
    g = a.global_info
    line = []
    if g and g.key is not None:
        line.append(f"Key: {g.key}")
    if g and g.tempo_bpm is not None:
        line.append(f"Tempo: {g.tempo_bpm:.0f} BPM")
    if g and g.time_signature is not None:
        line.append(f"Time: {g.time_signature}")
    if line:
        head.append("   ".join(line))
    head.append("Inner parts of a mix transcribe approximately: treat the right-hand read as a starting point, the chords and melody as the reliable part.")
    if len(a.beats) < 2:
        return "\n".join(head + ["", "No beat grid was detected — tab cannot be laid out. See chords.txt."]) + "\n"
    beatmap = BeatMap.from_analysis(a)
    blocks = [head, _shapes_block(t), _rhythm_block(t, beatmap), _melody_block(t, beatmap)]
    return "\n\n".join("\n".join(b) for b in blocks) + "\n"
