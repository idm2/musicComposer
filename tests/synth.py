"""Manufactured ground truth (BUILD-SPEC §11). Real songs have no answer key; this one does.

16 bars of 4/4 at 100 BPM in C major: C | C | Am | Am | F | F | G | G, twice.
Parts: saw-wave triads, a bass root, a kick/hat pattern, and a sine melody of chord tones in quarter notes.
"""
from pathlib import Path

import numpy as np

SR = 44100
BPM = 100
BEAT_S = 60 / BPM
BAR_S = 4 * BEAT_S
PROGRESSION = [("C:maj", [60, 64, 67]), ("A:min", [57, 60, 64]), ("F:maj", [53, 57, 60]), ("G:maj", [55, 59, 62])]
BARS_PER_CHORD = 2
REPEATS = 2
N_BARS = len(PROGRESSION) * BARS_PER_CHORD * REPEATS
DURATION_S = N_BARS * BAR_S
ALL_PARTS = frozenset({"harmony", "bass", "drums", "melody"})


def _bar_chords() -> list[tuple[str, list[int]]]:
    return [chord for _ in range(REPEATS) for chord in PROGRESSION for _ in range(BARS_PER_CHORD)]


def _melody_pitches(triad: list[int]) -> list[int]:
    root, third, fifth = (p + 12 for p in triad)
    return [root, third, fifth, third]


TRUTH = {
    "tempo_bpm": BPM, "key": "C major", "time_signature": "4/4",
    "chords": [(i * BARS_PER_CHORD * BAR_S, (i + 1) * BARS_PER_CHORD * BAR_S, PROGRESSION[i % 4][0])
               for i in range(len(PROGRESSION) * REPEATS)],
    "melody": [(round(bar * BAR_S + beat * BEAT_S, 6), pitch)
               for bar, (_, triad) in enumerate(_bar_chords())
               for beat, pitch in enumerate(_melody_pitches(triad))],
}


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _saw(freq: float, t: np.ndarray, harmonics: int = 10) -> np.ndarray:
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))


def _place(out: np.ndarray, start_s: float, sig: np.ndarray) -> None:
    a = int(round(start_s * SR))
    b = min(len(out), a + len(sig))
    out[a:b] += sig[: b - a]


def render(parts=ALL_PARTS) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(int(round(DURATION_S * SR)))
    bar_t = np.arange(int(BAR_S * SR)) / SR
    beat_t = np.arange(int(BEAT_S * SR)) / SR
    for bar, (_, triad) in enumerate(_bar_chords()):
        start = bar * BAR_S
        if "harmony" in parts:
            env = np.minimum(1, bar_t / 0.02) * np.exp(-bar_t * 0.5)
            _place(out, start, 0.25 * env * sum(_saw(_hz(p), bar_t) for p in triad))
        if "bass" in parts:
            env = np.minimum(1, bar_t / 0.01) * np.exp(-bar_t * 0.8)
            _place(out, start, 0.5 * env * _saw(_hz(triad[0] - 24), bar_t, 4))
        for beat in range(4):
            at = start + beat * BEAT_S
            if "drums" in parts:
                kick = np.sin(2 * np.pi * 55 * beat_t) * np.exp(-beat_t * 30)
                _place(out, at, (0.9 if beat == 0 else 0.5) * kick)
                for eighth in (0, 0.5):
                    hat = rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300)
                    _place(out, at + eighth * BEAT_S, 0.08 * hat)
            if "melody" in parts:
                env = np.minimum(1, beat_t / 0.015) * np.minimum(1, (BEAT_S - beat_t) / 0.05)
                _place(out, at, 0.35 * env * np.sin(2 * np.pi * _hz(_melody_pitches(triad)[beat]) * beat_t))
    return (0.7 * out / np.abs(out).max()).astype(np.float32)


def write(path: Path, parts=ALL_PARTS) -> Path:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = render(parts)
    sf.write(str(path), np.stack([mono, mono], axis=1), SR, subtype="PCM_16")
    return path
