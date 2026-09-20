import pytest

from songcomposer.jsonio import read_json, write_model
from songcomposer.models import GenerationRun, TakeRecord, TakesManifest
from songcomposer.paths import SongPaths
from songcomposer.pick import run_pick


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    p.takes_dir.mkdir(parents=True)
    takes = []
    for i, provider in ((1, "suno"), (2, "suno"), (3, "elevenlabs")):
        p.take(i).write_bytes(f"audio-{i}".encode())
        takes.append(TakeRecord(index=i, file=f"take-{i}.mp3", provider=provider, provider_ref="r",
                                duration_s=150, bytes=7))
    write_model(p.takes_json, TakesManifest(runs=[GenerationRun(
        provider="mixed", request_hash="h", fidelity="loose", fidelity_note="", style_prompt="s", payload={},
        cost_estimate_usd=0, cost_actual_usd=0, created_at="now", takes=takes)]))
    return p


def test_pick_records_choice_and_copies_deliverable(song):
    chosen = run_pick("demo", 3)
    assert (chosen.take, chosen.provider, chosen.file) == (3, "elevenlabs", "take-3.mp3")
    assert song.out_mp3.read_bytes() == b"audio-3"
    assert read_json(song.chosen)["sha1"] == chosen.sha1


def test_pick_unknown_take_lists_the_valid_ones(song):
    with pytest.raises(ValueError, match="1, 2, 3"):
        run_pick("demo", 9)


def test_changing_the_pick_invalidates_the_old_transcription(song):
    run_pick("demo", 1)
    song.transcription.write_text("{}", encoding="utf-8")
    run_pick("demo", 1)
    assert song.transcription.exists()            # same take: keep
    run_pick("demo", 2)
    assert not song.transcription.exists()        # different take: the old chart would describe the wrong audio
