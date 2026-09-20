import subprocess

import pytest

from songcomposer import generate
from songcomposer.config import Config
from songcomposer.generate.provider import GENERIC_LIMITS, CostEstimate, TakeResult
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import SongSpec, SpecSection
from songcomposer.paths import SongPaths

SPEC = SongSpec(
    title="Glass Hour", style_prompt="sparse indie folk", target_duration_s=120,
    sections=[SpecSection(name="Verse 1", duration_s=60, lines=[
                  "The kettle clicks off in the dark", "Your coat still hangs behind the door",
                  "I count the streetlights to the park", "And lose my place at twenty-four"]),
              SpecSection(name="Chorus", duration_s=60, lines=[
                  "Glass hour, hold me still", "Glass hour, against my will",
                  "Turn the morning down", "Until you come around"])])


@pytest.fixture(scope="session")
def mp3_bytes(tmp_path_factory):
    out = tmp_path_factory.mktemp("mp3") / "t.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=90",
                    "-b:a", "64k", str(out)], check=True)
    return out.read_bytes()


class FakeProvider:
    limits = GENERIC_LIMITS

    def __init__(self, name, mp3, n=3, fail_after=None):
        self.name, self.mp3, self.n, self.fail_after, self.calls = name, mp3, n, fail_after, 0

    def payload(self, req):
        return {"style": req.style_prompt, "lyrics": req.lyrics_text()}

    def estimate(self, req):
        return CostEstimate(usd=0.12, basis="fake")

    def generate(self, req, dest_dir, start_index, on_take):
        self.calls += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.n):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("provider blew up")
            path = dest_dir / f"take-{start_index + i}.mp3"
            path.write_bytes(self.mp3)
            on_take(TakeResult(provider_ref=f"ref-{i}", path=path, duration_s=90.0))
        return 0.11


@pytest.fixture
def song(root):
    p = SongPaths("demo")
    write_model(p.spec, SPEC)
    return p


def answers(*values):
    it = iter(values)
    return lambda prompt: next(it)


def test_preflight_failure_never_reaches_the_provider(song, mp3_bytes):
    bad = SPEC.model_copy(deep=True)
    bad.sections[0].lines[0] = "TODO write this"
    write_model(song.spec, bad)
    fake = FakeProvider("fake", mp3_bytes)
    with pytest.raises(SystemExit):
        generate.run_generate("demo", fidelity="loose", provider=fake, input_fn=answers("yes"))
    assert fake.calls == 0 and not song.takes_json.exists()


@pytest.mark.parametrize("reply", ["y", "", "no", "Yes please"])
def test_anything_but_exact_yes_aborts_without_spending(song, mp3_bytes, reply, capsys):
    fake = FakeProvider("fake", mp3_bytes)
    with pytest.raises(SystemExit):
        generate.run_generate("demo", fidelity="loose", provider=fake, input_fn=answers(reply))
    assert fake.calls == 0
    assert "nothing was spent" in capsys.readouterr().out


def test_cost_is_printed_before_confirmation(song, mp3_bytes, capsys):
    prompts = []

    def input_fn(prompt):
        prompts.append((prompt, capsys.readouterr().out))
        return "yes"

    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes), input_fn=input_fn)
    assert "$0.12" in prompts[0][1] and "pre-flight passed" in prompts[0][1]


def test_happy_path_writes_three_takes_and_manifest(song, mp3_bytes):
    m = generate.run_generate("demo", fidelity="medium", note="more space",
                              provider=FakeProvider("fake", mp3_bytes), input_fn=answers("yes"))
    assert [t.file for t in m.all_takes()] == ["take-1.mp3", "take-2.mp3", "take-3.mp3"]
    assert all(song.take(i).exists() for i in (1, 2, 3))
    run = read_json(song.takes_json)["runs"][0]
    assert (run["provider"], run["cost_estimate_usd"], run["cost_actual_usd"]) == ("fake", 0.12, 0.11)
    assert run["fidelity"] == "medium" and "more space" in run["style_prompt"]
    assert run["payload"]["lyrics"].startswith("[Verse 1]")
    assert read_json(song.spec)["fidelity"] == "medium"


def test_rerun_is_a_noop_without_regen(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes), input_fn=answers("yes"))
    again = FakeProvider("fake", mp3_bytes)
    generate.run_generate("demo", fidelity="loose", provider=again,
                          input_fn=lambda p: pytest.fail("must not ask to spend again"))
    assert again.calls == 0


def test_second_provider_appends_takes_for_the_bakeoff(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("suno", mp3_bytes, n=4), input_fn=answers("yes"))
    m = generate.run_generate("demo", fidelity="loose", provider=FakeProvider("elevenlabs", mp3_bytes),
                              input_fn=answers("yes"))
    assert [(t.index, t.provider) for t in m.all_takes()] == [
        (1, "suno"), (2, "suno"), (3, "suno"), (4, "suno"), (5, "elevenlabs"), (6, "elevenlabs"), (7, "elevenlabs")]


def test_regen_replaces_only_that_providers_takes(song, mp3_bytes):
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("suno", mp3_bytes, n=2), input_fn=answers("yes"))
    generate.run_generate("demo", fidelity="loose", provider=FakeProvider("elevenlabs", mp3_bytes, n=1), input_fn=answers("yes"))
    m = generate.run_generate("demo", fidelity="loose", regen=True, provider=FakeProvider("suno", mp3_bytes, n=2),
                              input_fn=answers("yes"))
    assert [(t.index, t.provider) for t in m.all_takes()] == [(3, "elevenlabs"), (4, "suno"), (5, "suno")]
    assert not song.take(1).exists() and song.take(4).exists()


def test_paid_takes_survive_a_mid_run_failure(song, mp3_bytes):
    with pytest.raises(RuntimeError, match="blew up"):
        generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes, fail_after=1),
                              input_fn=answers("yes"))
    run = read_json(song.takes_json)["runs"][0]
    assert len(run["takes"]) == 1 and "stopped early" in run["warnings"][0]
    assert run["cost_actual_usd"] == 0.12
    assert any("upper bound" in w for w in run["warnings"])


def test_cost_recorded_as_upper_bound_when_nothing_lands(song, mp3_bytes):
    with pytest.raises(RuntimeError, match="blew up"):
        generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes, fail_after=0),
                              input_fn=answers("yes"))
    run = read_json(song.takes_json)["runs"][0]
    assert run["takes"] == []
    assert run["cost_actual_usd"] == 0.12
    assert any("stopped early" in w for w in run["warnings"])
    assert any("upper bound" in w for w in run["warnings"])

    # a run with zero takes is not treated as cached — retrying must still gate on pre-flight/confirmation
    # and actually call the provider again.
    m = generate.run_generate("demo", fidelity="loose", provider=FakeProvider("fake", mp3_bytes),
                              input_fn=answers("yes"))
    assert [t.file for t in m.all_takes()] == ["take-1.mp3", "take-2.mp3", "take-3.mp3"]


def test_get_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="suno"):
        generate.get_provider("nope", Config())


def test_sanity_check_flags_short_and_tiny(tmp_path, mp3_bytes):
    f = tmp_path / "a.mp3"
    f.write_bytes(mp3_bytes)
    assert generate.sanity_check(f, 90, 120) is None
    assert "duration" in generate.sanity_check(f, 20, 120)
    f.write_bytes(b"x" * 500)
    assert "small" in generate.sanity_check(f, 90, 120)


def test_ask_fidelity():
    assert generate.ask_fidelity(answers("2", "slower")) == ("medium", "slower")
    assert generate.ask_fidelity(answers("9", "close", "")) == ("close", "")
