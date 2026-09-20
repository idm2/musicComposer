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
        if s.end <= s.start:
            # zero-length: the generator never sang this section. Never let a strum in a
            # neighbouring bar be misread as "detected" for a section that doesn't exist in the audio.
            out.append(f"{s.label}: not detected in the audio")
            continue
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
