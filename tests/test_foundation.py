import os

import pytest

from songcomposer import env, hashing, jsonio
from songcomposer.config import load_config
from songcomposer.paths import SongPaths


def test_load_env_parses_quotes_and_does_not_override(root, monkeypatch):
    (root / ".env").write_text('A_KEY="quoted"\n# comment\nB_KEY=plain\nC_KEY=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.setenv("C_KEY", "fromshell")
    env._loaded = False
    env.load_env()
    assert os.environ["A_KEY"] == "quoted"
    assert os.environ["B_KEY"] == "plain"
    assert os.environ["C_KEY"] == "fromshell"


def test_require_env_missing_raises_with_name(root, monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    env._loaded = False
    with pytest.raises(RuntimeError, match="NOPE_KEY"):
        env.require_env("NOPE_KEY")


def test_json_roundtrip_preserves_smart_quotes_and_accents(root):
    p = root / "deep" / "x.json"
    data = {"line": "She said “café” — déjà vu’s"}
    jsonio.write_json(p, data)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")          # no BOM
    assert "“café”".encode("utf-8") in raw               # not \u-escaped
    assert raw.endswith(b"\n") and b"\r\n" not in raw
    assert jsonio.read_json(p) == data


def test_song_paths_layout(root):
    p = SongPaths("my-song")
    assert p.work == root / "work" / "my-song"
    assert p.source_wav.name == "00-source.wav"
    assert p.source_json.name == "00-source.json"
    assert p.analysis.name == "01-analysis.json"
    assert p.brief.name == "02-brief.json"
    assert p.spec.name == "03-spec.json"
    assert p.takes_dir == p.work / "04-takes"
    assert p.takes_json == p.takes_dir / "takes.json"
    assert p.chosen.name == "05-chosen.json"
    assert p.transcription.name == "06-transcription.json"
    assert p.cache == p.work / ".cache"
    assert p.out == root / "out" / "my-song"
    assert p.out_mp3 == p.out / "my-song.mp3"
    assert p.out_musicxml == p.out / "my-song.musicxml"
    assert {p.chords_txt.name, p.tab_txt.name, p.chart_pdf.name, p.lyrics_json.name} == {
        "chords.txt", "tab.txt", "chart.pdf", "lyrics.json"}


@pytest.mark.parametrize("bad", ["My Song", "../x", "", "-lead", "a_b"])
def test_song_name_rejected(root, bad):
    with pytest.raises(ValueError):
        SongPaths(bad)


def test_hashing(sine_wav):
    h = hashing.file_sha1(sine_wav)
    assert len(h) == 40 and h == hashing.file_sha1(sine_wav)
    assert len(hashing.content_key("a", "b")) == 16
    assert hashing.content_key("a", "b") != hashing.content_key("ab", "")


def test_config_defaults_when_file_missing(root):
    c = load_config()
    assert c.default_provider == "suno"
    assert c.whisper_model == "large-v3"
