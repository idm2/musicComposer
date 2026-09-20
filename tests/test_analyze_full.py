import subprocess
import sys
from pathlib import Path

import pytest

from songcomposer import analysis
from songcomposer.analysis import beats, chords, key, lyrics, notes, stems, structure, subjective
from songcomposer.config import Config
from songcomposer.models import Chord, Note, Subjective, Word

HEARD = Subjective(genre_tags=[], instrumentation=[], timbre="", vocal_character="", vocal_gender="none",
                   production="", emotional_arc="", arrangement_density=[], sections=[])


def _fake_transcribe_words(seen):
    """Records the hint it was called with, and always returns a single fake word.
    (Deviation from the brief: replaces its `seen.setdefault(...) and [] or [...]` lambda,
    which relied on a falsy-empty-list trick, with a small named function — same behaviour,
    readable.)"""
    def fn(v, c, m, hint=""):
        seen["hint"] = hint
        return [Word(word="hi", start=0, end=1, confidence=0.9)]
    return fn


@pytest.fixture
def patched(monkeypatch, tmp_path):
    seen = {}
    fake = stems.Stems(vocals=Path("v.wav"), drums=Path("d.wav"), bass=Path("b.wav"), other=Path("o.wav"),
                       harmonic=Path("h.wav"))
    monkeypatch.setattr(analysis, "objective_installed", lambda: True)
    monkeypatch.setattr(subjective, "listen", lambda a, c, m: HEARD)
    monkeypatch.setattr(stems, "separate", lambda a, c: fake)
    monkeypatch.setattr(beats, "track_beats", lambda a, c: beats.summarise_beats([i * 0.6 for i in range(16)], [0.0, 2.4, 4.8, 7.2]))
    monkeypatch.setattr(chords, "detect_chords", lambda h, m, c: [
        Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None, onset=0.04, duration=2.36, confidence=0.9)])
    monkeypatch.setattr(key, "measure", lambda h, m, c: {"key": "C major", "key_confidence": 0.8, "loudness_lufs": -14.0, "duration_s": 9.6})
    monkeypatch.setattr(notes, "transcribe_notes", lambda s, c: [Note(pitch=60, onset=0, duration=1, confidence=0.7, stem="vocals")])
    monkeypatch.setattr(lyrics, "transcribe_words", _fake_transcribe_words(seen))
    monkeypatch.setattr(structure, "boundaries", lambda h, d, dur: [0.0, dur])
    return seen


def test_full_analysis_fuses_every_ear(patched, tmp_path, sine_wav):
    a = analysis.analyze(sine_wav, tmp_path, Config(), lyrics_hint="our words")
    assert set(a.engines) == {"subjective", "stems", "beats", "chords", "key", "notes", "lyrics", "structure"}
    assert a.global_info.key == "C major" and a.global_info.tempo_bpm == 100.0 and a.global_info.time_signature == "4/4"
    assert a.chords[0].onset == 0.0                                  # snapped to the beat grid
    assert a.notes and a.lyrics and a.sections and a.subjective == HEARD
    assert patched["hint"] == "our words"


def test_subjective_only_still_works_and_warns(patched, tmp_path, sine_wav, capsys):
    a = analysis.analyze(sine_wav, tmp_path, Config(), ears={"subjective"})
    assert a.chords == [] and a.global_info is None
    assert "PARTIAL ANALYSIS" in capsys.readouterr().out


def test_importing_analysis_package_does_not_import_the_heavy_stack():
    """The top-level package imports every component module lazily, inside analyze() itself,
    so `import songcomposer.analysis` alone must never pull in the GPU stack."""
    heavy = "torch", "librosa", "lv_chordia", "faster_whisper", "basic_pitch", "demucs", "beat_this"
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis; "
         f"loaded = [m for m in {heavy!r} if m in sys.modules]; "
         "assert not loaded, f'songcomposer.analysis imported {loaded} eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.gpu
def test_end_to_end_on_the_synth_recovers_the_truth(tmp_path, synth_song):
    a = analysis.analyze(synth_song.mix, tmp_path, Config(), ears=set(analysis.OBJECTIVE_EARS) - {"lyrics"})
    t = synth_song.truth
    assert a.global_info.key == t["key"] and abs(a.global_info.tempo_bpm - 100) < 2
    right = sum(min(e, c.onset + c.duration) - max(s, c.onset)
                for s, e, lab in t["chords"] for c in a.chords
                if c.harte == lab and min(e, c.onset + c.duration) > max(s, c.onset))
    assert right / 38.4 >= 0.75, f"only {right / 38.4:.0%} of the timeline has the right chord"
