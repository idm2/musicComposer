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
    firsts: list[tuple[str, float, list[float]]] = []
    for line in lines:
        if line.start is None:
            continue
        if firsts and firsts[-1][0] == line.section:
            firsts[-1][2].append(line.confidence)
        else:
            firsts.append((line.section, line.start, [line.confidence]))
    return [Section(label=name, start=start, end=firsts[i + 1][1] if i + 1 < len(firsts) else duration_s,
                    confidence=round(sum(confs) / len(confs), 3)) for i, (name, start, confs) in enumerate(firsts)]


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
