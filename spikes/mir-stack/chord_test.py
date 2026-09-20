"""Spike: synthesize C - Am - F - G (4 s each) and run lv-chordia on it."""
import sys
import time

import numpy as np
import soundfile as sf

SR = 44100
WAV = "test_C_Am_F_G.wav"


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(f, t, n_harm=12):
    # band-limited-ish sawtooth: sum of harmonics
    out = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        out += np.sin(2 * np.pi * f * k * t) / k
    return out


def triad(notes, dur=4.0):
    t = np.arange(int(SR * dur)) / SR
    y = sum(saw(midi_hz(n), t) for n in notes)
    # root an octave below as a bass note
    y += 1.5 * saw(midi_hz(notes[0] - 12), t, 6)
    env = np.minimum(1, t / 0.02) * np.exp(-t * 0.4)
    return y * env


prog = [
    ("C", [60, 64, 67]),
    ("Am", [57, 60, 64]),
    ("F", [53, 57, 60]),
    ("G", [55, 59, 62]),
]
y = np.concatenate([triad(n) for _, n in prog])
y = 0.5 * y / np.abs(y).max()
sf.write(WAV, y.astype(np.float32), SR)
print("wrote", WAV, len(y) / SR, "s")

if "--lv" in sys.argv:
    from lv_chordia.chord_recognition import chord_recognition

    t0 = time.time()
    import os

    # NOTE: lv-chordia resolves relative paths against its own package dir -> pass absolute
    res = chord_recognition(os.path.abspath(WAV), chord_dict_name="submission")
    print("lv-chordia took %.1fs" % (time.time() - t0))
    for r in res:
        print(r)
