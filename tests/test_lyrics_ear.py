from types import SimpleNamespace as NS

import pytest

from songcomposer.analysis import lyrics


def test_words_from_segments_strips_and_carries_probability():
    segs = [NS(words=[NS(word=" Glass", start=1.0, end=1.4, probability=0.93), NS(word=" hour,", start=1.4, end=2.0, probability=0.41)]),
            NS(words=None), NS(words=[NS(word="  ", start=3, end=3.1, probability=0.9)])]
    got = lyrics.words_from_segments(segs)
    assert [(w.word, w.start, w.end, w.confidence) for w in got] == [("Glass", 1.0, 1.4, 0.93), ("hour,", 1.4, 2.0, 0.41)]


def test_hint_changes_the_cache_key(tmp_path, sine_wav, monkeypatch):
    calls = []
    monkeypatch.setattr(lyrics, "_run_whisper", lambda path, model, hint: calls.append(hint) or [])
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3")
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3")
    lyrics.transcribe_words(sine_wav, tmp_path, "large-v3", hint="glass hour hold me still")
    assert calls == ["", "glass hour hold me still"]


def test_importing_lyrics_does_not_import_heavy_deps():
    """Architectural pin: analyze() will import every component module, including lyrics,
    even for an LLM-only --ears subjective run that has no business touching these. Run in
    a subprocess so the current session's already-imported modules cannot mask a regression."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.analysis.lyrics; "
         "assert 'torch' not in sys.modules, 'lyrics imported torch eagerly'; "
         "assert 'faster_whisper' not in sys.modules, 'lyrics imported faster_whisper eagerly'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.gpu
def test_whisper_loads_on_cuda_and_returns_a_list(tmp_path, sine_wav):
    assert isinstance(lyrics.transcribe_words(sine_wav, tmp_path, "tiny"), list)
