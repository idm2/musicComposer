"""The engine's SECOND run: analyse our own chosen take, then align the lyric lines we wrote to what was sung.
The chart is derived from the actual recording, so the two can never disagree."""
import re
from difflib import SequenceMatcher

from .analysis import OBJECTIVE_EARS, analyze
from .audio import to_wav
from .config import load_config
from .jsonio import read_json, write_model
from .models import Chosen, LyricLine, Section, SongSpec, Transcription, Word
from .paths import SongPaths

GUESS_WORD_S = 0.3
MAX_LINE_SPAN_S = 60.0


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def _fill(times: list[tuple[float, float] | None]) -> list[tuple[float, float]]:
    """Interpolate timings for unmatched words between their matched neighbours."""
    out = list(times)
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        n = j - i
        left = out[i - 1][1] if i > 0 else out[j][0] - GUESS_WORD_S * n
        right = out[j][0] if j < len(out) else left + GUESS_WORD_S * n
        step = max(0.0, right - left) / n
        for k in range(n):
            out[i + k] = (round(left + k * step, 3), round(left + (k + 1) * step, 3))
        i = j
    return out


def align_lines(spec: SongSpec, heard: list[Word]) -> list[LyricLine]:
    tokens = [(s.name, li, tok) for s in spec.sections for li, line in enumerate(s.lines) for tok in line.split()]
    a, b = [_norm(t[2]) for t in tokens], [_norm(w.word) for w in heard]
    match: dict[int, int] = {}
    for blk in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            if a[blk.a + k]:
                match[blk.a + k] = blk.b + k

    lines: list[LyricLine] = []
    cursor = 0
    for s in spec.sections:
        for text in s.lines:
            n = len(text.split())
            idxs = list(range(cursor, cursor + n))
            cursor += n
            hits = [i for i in idxs if i in match]
            if hits:
                span = heard[match[hits[-1]]].end - heard[match[hits[0]]].start
            if not hits or span > MAX_LINE_SPAN_S or span < 0:
                lines.append(LyricLine(section=s.name, text=text, start=None, end=None, words=[], confidence=0.0))
                continue
            times = _fill([(heard[match[i]].start, heard[match[i]].end) if i in match else None for i in idxs])
            words = [Word(word=tokens[i][2], start=t[0], end=t[1],
                          confidence=heard[match[i]].confidence if i in match else 0.0) for i, t in zip(idxs, times)]
            mean = sum(heard[match[i]].confidence for i in hits) / len(hits)
            lines.append(LyricLine(section=s.name, text=text, start=words[0].start, end=words[-1].end, words=words,
                                   confidence=round(mean * len(hits) / n, 3)))
    return lines


def sections_from_lines(lines: list[LyricLine], duration_s: float) -> list[Section]:
    """Emit every spec section, in spec order — never drop one just because none of its lines
    aligned. A section with no aligned line is a zero-length marker (start == end) at the point
    where it would have sat: the end of whatever section was emitted before it, or the first
    aligned start (0.0 if nothing ever aligned) when it precedes everything. Zero length says
    "we don't know where this ran"; it never borrows or inflates a neighbour's span."""
    groups: list[tuple[str, list[LyricLine]]] = []
    for line in lines:
        if groups and groups[-1][0] == line.section:
            groups[-1][1].append(line)
        else:
            groups.append((line.section, [line]))

    aligned: dict[int, tuple[float, list[float]]] = {}
    for idx, (_, glines) in enumerate(groups):
        hits = [l for l in glines if l.start is not None]
        if hits:
            aligned[idx] = (hits[0].start, [l.confidence for l in hits])

    aligned_order = sorted(aligned)
    ends = {idx: (aligned[aligned_order[pos + 1]][0] if pos + 1 < len(aligned_order) else duration_s)
           for pos, idx in enumerate(aligned_order)}
    first_aligned_start = aligned[aligned_order[0]][0] if aligned_order else 0.0

    sections: list[Section] = []
    prev_end: float | None = None
    for idx, (name, _) in enumerate(groups):
        if idx in aligned:
            start, confs = aligned[idx]
            end = ends[idx]
            confidence = round(sum(confs) / len(confs), 3)
        else:
            start = end = first_aligned_start if prev_end is None else prev_end
            confidence = 0.0
        sections.append(Section(label=name, start=start, end=end, confidence=confidence))
        prev_end = end
    return sections


def run_transcribe(song: str, force: bool = False) -> Transcription:
    paths = SongPaths(song)
    chosen = Chosen(**read_json(paths.require(paths.chosen, "pick")))
    if paths.transcription.exists() and not force:
        existing = Transcription(**read_json(paths.transcription))
        if existing.take == chosen.take:
            print(f"- transcription of take {chosen.take} exists (use --force to redo)")
            return existing
    spec = SongSpec(**read_json(paths.require(paths.spec, "compose")))
    wav = paths.cache / f"chosen-{chosen.sha1[:16]}.wav"
    if not wav.exists():
        to_wav(paths.takes_dir / chosen.file, wav)
    print(f"> transcribing take {chosen.take} — same engine that analysed the reference")
    result = analyze(wav, paths.cache, load_config(), ears=set(OBJECTIVE_EARS), lyrics_hint=" ".join(spec.all_lines())[:900])
    lines = align_lines(spec, result.lyrics)
    aligned = [l for l in lines if l.start is not None]
    if len(aligned) * 2 >= len(lines) and result.global_info:          # our own form beats a DSP guess at it
        result.sections = sections_from_lines(lines, result.global_info.duration_s)
    out = Transcription(take=chosen.take, analysis=result, lines=lines)
    write_model(paths.transcription, out)
    print(f"  {len(aligned)}/{len(lines)} lyric lines aligned, {len(result.chords)} chords → {paths.transcription}")
    return out
