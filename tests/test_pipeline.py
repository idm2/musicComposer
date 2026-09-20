"""Contract test (BUILD-SPEC §11): run every stage with the paid/GPU parts faked and validate every stage file
against its schema. No network, no GPU, no money."""
import pytest

from songcomposer import compose, pipeline, transcribe
from songcomposer.jsonio import read_json
from songcomposer.models import (Analysis, Brief, Chord, Chosen, GlobalInfo, Note, SongSpec, SourceInfo, Subjective,
                                 TakesManifest, Transcription, Word)
from songcomposer.paths import SongPaths
from test_compose import GOOD
from test_generate import FakeProvider, mp3_bytes  # noqa: F401

GRID = [i * 0.6 for i in range(64)]


def fake_analyze(audio, cache_dir, config, ears=None, lyrics_hint=""):
    words = [Word(word=w, start=2.0 + i * 0.5, end=2.4 + i * 0.5, confidence=0.9)
             for i, w in enumerate(" ".join(GOOD.sections[0].lines).split())]
    return Analysis(
        audio_sha1="a" * 40, engines={"chords": "fake"}, beats=GRID, downbeats=GRID[::4], lyrics=words if lyrics_hint else [],
        chords=[Chord(symbol="C", harte="C:maj", root="C", quality="maj", extensions=[], bass=None, onset=0, duration=2.4, confidence=0.9)],
        notes=[Note(pitch=64, onset=2.0, duration=0.5, confidence=0.9, stem="vocals")],
        global_info=GlobalInfo(key="C major", key_confidence=0.9, tempo_bpm=100, time_signature="4/4", loudness_lufs=-14, duration_s=38.4),
        subjective=Subjective(genre_tags=["folk"], instrumentation=["guitar"], timbre="warm", vocal_character="breathy",
                              vocal_gender="female", production="dry", emotional_arc="", arrangement_density=[], sections=[]))


@pytest.mark.parametrize("with_reference", [True, False], ids=["with-reference", "brief-only"])
def test_whole_pipeline_contracts(root, sine_wav, mp3_bytes, monkeypatch, with_reference):  # noqa: F811
    pytest.importorskip("music21")
    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(transcribe, "analyze", fake_analyze)

    def fake_to_wav(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"wav")

    monkeypatch.setattr(transcribe, "to_wav", fake_to_wav)
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)

    if with_reference:
        src, fidelity, chosen_take, replies = str(sine_wav), "loose", 2, iter(["yes", "2"])
    else:
        # No --from at all (Task 29): ingest and analyze never run. fidelity is left unset — with no
        # analysis, run_generate must not ask for it (and must not ask to spend before the fake 'yes').
        src, fidelity, chosen_take, replies = None, None, 1, iter(["yes", "1"])

    pipeline.run_all("demo", src, brief_text="A song about insomnia after a breakup.",
                     provider=FakeProvider("fake", mp3_bytes), fidelity=fidelity,
                     input_fn=lambda prompt: next(replies))
    p = SongPaths("demo")
    expected = [(p.brief, Brief), (p.spec, SongSpec), (p.takes_json, TakesManifest), (p.chosen, Chosen),
                (p.transcription, Transcription)]
    if with_reference:
        expected = [(p.source_json, SourceInfo), (p.analysis, Analysis)] + expected
    for path, model in expected:
        model.model_validate(read_json(path))                     # raises if any stage broke its contract
    if not with_reference:
        assert not p.source_wav.exists() and not p.source_json.exists() and not p.analysis.exists()
        assert read_json(p.spec)["fidelity"] == "loose"
    assert read_json(p.chosen)["take"] == chosen_take
    assert p.out_mp3.exists() and p.chords_txt.exists() and p.tab_txt.exists() and p.lyrics_json.exists()
    assert read_json(p.lyrics_json)["lines"][0]["words"][0]["word"] == "The"


def test_run_all_threads_the_take_count_through_to_generate(root, sine_wav, mp3_bytes, monkeypatch):  # noqa: F811
    """`--takes` must reach run_generate — not just be accepted and dropped."""
    pytest.importorskip("music21")
    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(transcribe, "analyze", fake_analyze)
    monkeypatch.setattr(transcribe, "to_wav", lambda src, dst: (dst.parent.mkdir(parents=True, exist_ok=True),
                                                                dst.write_bytes(b"wav")))
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)
    replies = iter(["yes", "1"])
    pipeline.run_all("demo", str(sine_wav), brief_text="A song about insomnia after a breakup.",
                     provider=FakeProvider("fake", mp3_bytes, n=2), fidelity="loose", takes=2,
                     input_fn=lambda prompt: next(replies))
    p = SongPaths("demo")
    assert [t.index for t in TakesManifest(**read_json(p.takes_json)).all_takes()] == [1, 2]
