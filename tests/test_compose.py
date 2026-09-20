import pytest

from songcomposer import compose
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import Analysis, Brief, Chord, GlobalInfo, SourceInfo, SpecSection, Subjective, Word
from songcomposer.paths import SongPaths

ANALYSIS = Analysis(
    audio_sha1="a" * 40, engines={"subjective": "m"},
    global_info=GlobalInfo(key="A minor", key_confidence=0.8, tempo_bpm=92, time_signature="4/4",
                           loudness_lufs=-14, duration_s=200),
    chords=[Chord(symbol="Am", harte="A:min", root="A", quality="min", extensions=[], bass=None,
                  onset=0, duration=2, confidence=0.9),
            Chord(symbol="F", harte="F:maj", root="F", quality="maj", extensions=[], bass=None,
                  onset=2, duration=2, confidence=0.3)],
    lyrics=[Word(word="harbour", start=1, end=1.5, confidence=0.9)],
    subjective=Subjective(genre_tags=["folk"], instrumentation=["acoustic guitar"], timbre="warm",
                          vocal_character="breathy", vocal_gender="female", production="dry",
                          emotional_arc="sad to hopeful", arrangement_density=[], sections=[]))

GOOD = compose.ComposedSong(
    title="Glass Hour", style_prompt="sparse indie folk", negative_style="edm", vocal_gender="female",
    target_duration_s=120,
    sections=[SpecSection(name="Verse 1", duration_s=60, lines=["The kettle clicks off in the dark",
                          "Your coat still hangs behind the door", "I count the streetlights to the park",
                          "And lose my place at twenty-four"]),
              SpecSection(name="Chorus", duration_s=60, lines=["Glass hour, hold me still",
                          "Glass hour, against my will", "Turn the morning down", "Until you come around"])])


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    write_model(p.analysis, ANALYSIS)
    write_model(p.brief, Brief(text="A song about insomnia after a breakup.", origin="inline"))
    write_model(p.source_json, SourceInfo(origin="u", kind="url", title="Some Artist - Some Song",
                uploader="Some Artist", duration_s=200, sample_rate=44100, sha1="a" * 40, ingested_at="now"))
    return p


@pytest.fixture
def song_without_reference(root):
    """No 00-source.json, no 01-analysis.json — a brief-only song (Task 29)."""
    p = SongPaths("demo")
    write_model(p.brief, Brief(text="A song about insomnia after a breakup.", origin="inline"))
    return p


def test_summary_includes_measured_facts_and_marks_low_confidence_chords():
    s = compose.summarise_analysis(ANALYSIS)
    assert "A minor" in s and "92" in s and "breathy" in s
    vocab_line = s.split("Chord vocabulary")[1].split("\n")[0]
    vocab_symbols = [sym.strip() for sym in vocab_line.split(":", 1)[1].split(",")]
    assert "Am" in vocab_symbols and "F" not in vocab_symbols   # 0.3-confidence F excluded


def test_summary_with_partial_analysis_says_so():
    s = compose.summarise_analysis(Analysis(audio_sha1="a" * 40, engines={}))
    assert "not measured" in s


def test_compose_writes_spec_and_passes_brief_and_analysis_to_writer(song, monkeypatch):
    seen = {}

    def fake(messages, model, out_type, **kw):
        seen["text"] = messages[-1]["content"]
        return GOOD

    monkeypatch.setattr(compose, "chat_json", fake)
    spec = compose.run_compose("demo")
    assert "insomnia" in seen["text"] and "A minor" in seen["text"]
    assert "Some Artist" not in seen["text"]                      # provenance never reaches the writer
    on_disk = read_json(song.spec)
    assert on_disk["title"] == "Glass Hour" and on_disk["fidelity"] is None
    assert spec.writer_model == "google/gemini-2.5-pro"


def test_compose_feeds_preflight_problems_back_once(song, monkeypatch):
    bad = GOOD.model_copy(deep=True)
    bad.sections[0].lines[0] = "TODO write this"
    replies = iter([bad, GOOD])
    prompts = []

    def fake(messages, model, out_type, **kw):
        prompts.append(messages[-1]["content"])
        return next(replies)

    monkeypatch.setattr(compose, "chat_json", fake)
    compose.run_compose("demo")
    assert len(prompts) == 2 and "placeholder" in prompts[1]


def test_compose_is_noop_when_spec_exists(song, monkeypatch):
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)
    compose.run_compose("demo")
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: pytest.fail("should not be called"))
    compose.run_compose("demo")


def test_system_prompts_share_the_songwriting_rules():
    """The reference-specific framing paragraph differs, but the actual songwriting rules are one
    block shared by both prompts — not two near-identical copies."""
    shared_snippet = "10-25 words describing genre, mood, instrumentation and vocal delivery"
    assert shared_snippet in compose.SYSTEM
    assert shared_snippet in compose.SYSTEM_NO_REFERENCE
    assert compose.SYSTEM != compose.SYSTEM_NO_REFERENCE


def test_compose_without_reference_writes_spec_and_says_so(song_without_reference, monkeypatch, capsys):
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: GOOD)
    spec = compose.run_compose("demo")
    assert spec.title == "Glass Hour"
    assert read_json(song_without_reference.spec)["title"] == "Glass Hour"
    assert "without a reference" in capsys.readouterr().out


def test_compose_without_reference_prompt_never_mentions_a_reference(song_without_reference, monkeypatch):
    seen = {}

    def fake(messages, model, out_type, **kw):
        seen["messages"] = messages
        return GOOD

    monkeypatch.setattr(compose, "chat_json", fake)
    compose.run_compose("demo")
    combined = " ".join(m["content"] for m in seen["messages"]).lower()
    assert "reference" not in combined
    assert "insomnia" in combined                             # the brief still reaches the writer


def test_compose_without_reference_still_feeds_preflight_problems_back(song_without_reference, monkeypatch):
    bad = GOOD.model_copy(deep=True)
    bad.sections[0].lines[0] = "TODO write this"
    replies = iter([bad, GOOD])
    monkeypatch.setattr(compose, "chat_json", lambda *a, **k: next(replies))
    spec = compose.run_compose("demo")
    assert spec.title == "Glass Hour"
