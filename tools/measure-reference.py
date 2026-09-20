"""AD-HOC objective listen of the reference — NOT the pipeline (Tasks 15-20 are unbuilt).
Uses the built stems stage plus the installed models directly. No confidence scores yet."""
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

root = Path(r"C:\dev\IDM2 Apps\Song Composer")
work = root / "work" / "in-the-flow"
wav = work / "00-source.wav"

from songcomposer.analysis.stems import separate

stems = separate(wav, work / ".cache")
print("stems ok:", stems.harmonic, file=sys.stderr)

import librosa
import numpy as np
import torch
from beat_this.inference import File2Beats
from lv_chordia.chord_recognition import chord_recognition

beats, downbeats = File2Beats(checkpoint_path="final0", device="cuda" if torch.cuda.is_available() else "cpu", dbn=False)(str(wav))
beats = [float(b) for b in beats]
downbeats = [float(d) for d in downbeats]
tempo = 60.0 / median(b - a for a, b in zip(beats, beats[1:]))
per_bar = Counter(sum(1 for b in beats if lo - 0.02 <= b < hi - 0.02) for lo, hi in zip(downbeats, downbeats[1:])).most_common(3)

chords_h = chord_recognition(str(stems.harmonic.resolve()), chord_dict_name="submission")
chords_m = chord_recognition(str(wav.resolve()), chord_dict_name="submission")

MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
y, sr = librosa.load(str(stems.harmonic), sr=22050, mono=True)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
scored = sorted(
    [(float(np.corrcoef(chroma, np.roll(MAJOR, t))[0, 1]), f"{NAMES[t]} major") for t in range(12)]
    + [(float(np.corrcoef(chroma, np.roll(MINOR, t))[0, 1]), f"{NAMES[t]} minor") for t in range(12)], reverse=True)

out = {
    "tempo_bpm": round(tempo, 1), "beats_per_bar_counts": per_bar, "n_beats": len(beats),
    "first_downbeats": [round(d, 2) for d in downbeats[:6]],
    "key_candidates": [(n, round(s, 3)) for s, n in scored[:4]],
    "chords_harmonic_stem": [(c["start_time"], c["end_time"], c["chord"]) for c in chords_h],
    "chords_full_mix": [(c["start_time"], c["end_time"], c["chord"]) for c in chords_m],
}
(work / "adhoc-objective.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
