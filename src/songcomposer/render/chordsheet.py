"""chords.txt — lyrics with chords above the line, placed from the transcription of the actual take."""
from ..models import LOW_CONFIDENCE, Chord, LyricLine, Transcription
from .guitar import chord_charts, suggest_capo

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


def render_chordsheet(title: str, t: Transcription, capo: bool = False) -> str:
    a = t.analysis
    chords = sorted(a.chords, key=lambda c: c.onset)
    low = sum(1 for c in chords if c.confidence < LOW_CONFIDENCE)
    out = [title, "=" * len(title)]
    g = a.global_info
    line = []
    if g and g.key is not None:
        line.append(f"Key: {g.key} (confidence {g.key_confidence:.2f})")
    if g and g.tempo_bpm is not None:
        line.append(f"Tempo: {g.tempo_bpm:.0f} BPM")
    if g and g.time_signature is not None:
        line.append(f"Time: {g.time_signature}")
    if line:
        out.append("   ".join(line))
    # Off by default: Sam plays open shapes and a wrong capo line makes every chord on the sheet wrong.
    fret, shapes = suggest_capo(chords) if capo else (0, {})
    if fret:
        out.append(f"Capo suggestion: fret {fret} — play " + ", ".join(f"{k}→{v}" for k, v in shapes.items()))
    out += [f"Transcribed from take {t.take}. ? = low-confidence chord (below {LOW_CONFIDENCE}): {low} of {len(chords)} chords.", ""]
    if chords:
        out += chord_charts(chords) + [""]

    timed = [l for l in t.lines if l.start is not None]
    if timed:
        intro = [c for c in chords if c.onset < timed[0].start - LEAD_S]
        if intro:
            out += ["[Intro]", "  ".join(label(c) for c in intro), ""]
    elif chords:
        # no lyric line could be aligned at all (e.g. the take sang something quite different from
        # the spec) — never silently drop the chords just because they have nowhere to sit above a word
        out += ["[Chords — no lyric timing detected, not placed against words]",
                "  ".join(label(c) for c in chords), ""]
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
            row = "  ".join(label(c) for c in outro)
            # the last sung section may already BE "Outro" — don't print a second, duplicate header
            out += ["", row] if section and section.lower() == "outro" else ["", "[Outro]", row]
    return "\n".join(out) + "\n"
