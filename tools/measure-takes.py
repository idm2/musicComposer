"""AD-HOC: measure key / tempo / chords of each generated take (full mix, no stems — quick look)."""
import json, subprocess, sys
from collections import Counter
from pathlib import Path
from statistics import median
import librosa, numpy as np, torch
from beat_this.inference import File2Beats
from lv_chordia.chord_recognition import chord_recognition

takes = Path(r"C:\dev\IDM2 Apps\Song Composer\work\in-the-flow\04-takes")
MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
f2b = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
out = {}
for i in (1, 2, 3, 4):
    mp3 = takes / f"take-{i}.mp3"; wav = takes / f".tmp-take-{i}.wav"
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(mp3),"-ac","2","-ar","44100",str(wav)], check=True)
    beats, downs = f2b(str(wav)); beats = [float(b) for b in beats]
    tempo = 60.0 / median(b - a for a, b in zip(beats, beats[1:]))
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    sc = sorted([(float(np.corrcoef(ch, np.roll(MAJOR,t))[0,1]), f"{NAMES[t]} major") for t in range(12)] +
                [(float(np.corrcoef(ch, np.roll(MINOR,t))[0,1]), f"{NAMES[t]} minor") for t in range(12)], reverse=True)
    chords = chord_recognition(str(wav.resolve()), chord_dict_name="submission")
    dur = Counter()
    for c in chords:
        if c["chord"] != "N": dur[c["chord"]] += c["end_time"] - c["start_time"]
    out[i] = {"tempo": round(tempo,1), "key": [(n, round(s,2)) for s, n in sc[:2]],
              "top_chords_by_seconds": [(k, round(v)) for k, v in dur.most_common(8)],
              "first_16": [c["chord"] for c in chords if c["chord"] != "N"][:16]}
    wav.unlink()
print(json.dumps(out, indent=1))
