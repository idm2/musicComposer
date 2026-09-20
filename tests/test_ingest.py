import subprocess

import pytest

from songcomposer import ingest
from songcomposer.audio import probe_duration
from songcomposer.jsonio import read_json
from songcomposer.paths import SongPaths


@pytest.mark.parametrize("src,kind", [
    ("https://www.youtube.com/watch?v=abc", "url"),
    ("http://example.com/x.mp3", "url"),
    ("C:/music/ref.mp3", "audio"), ("ref.WAV", "audio"), ("a.flac", "audio"), ("a.m4a", "audio"),
    ("clip.mp4", "video"), ("clip.MOV", "video"), ("clip.mkv", "video"), ("clip.webm", "video"),
])
def test_classify_source(src, kind):
    assert ingest.classify_source(src) == kind


def test_classify_unknown_extension_raises():
    with pytest.raises(ValueError, match="unsupported"):
        ingest.classify_source("notes.txt")


def test_ingest_local_audio_normalises_to_44k_stereo_wav(root, sine_wav):
    info = ingest.run_ingest("demo", str(sine_wav))
    p = SongPaths("demo")
    assert p.source_wav.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels,codec_name", "-of", "csv=p=0", str(p.source_wav)],
        capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "pcm_s16le,44100,2"
    assert info.kind == "audio" and info.sample_rate == 44100
    assert abs(info.duration_s - 2.0) < 0.1
    assert abs(probe_duration(p.source_wav) - 2.0) < 0.1
    on_disk = read_json(p.source_json)
    assert on_disk["sha1"] == info.sha1 and len(info.sha1) == 40
    assert on_disk["origin"] == str(sine_wav)


def test_ingest_is_noop_when_source_exists_unless_forced(root, sine_wav):
    first = ingest.run_ingest("demo", str(sine_wav))
    again = ingest.run_ingest("demo", "does-not-exist.wav")     # would fail if it actually ran
    assert again.sha1 == first.sha1
    with pytest.raises(FileNotFoundError):
        ingest.run_ingest("demo", "does-not-exist.wav", force=True)


def test_ingest_url_uses_ytdlp_then_normalises(root, sine_wav, monkeypatch):
    calls = []

    def fake_download(url, dest_dir):
        calls.append(url)
        target = dest_dir / "download.wav"
        target.write_bytes(sine_wav.read_bytes())
        return target, {"title": "Some Title", "uploader": "Some Channel"}

    monkeypatch.setattr(ingest, "_download", fake_download)
    info = ingest.run_ingest("demo", "https://youtu.be/xyz")
    assert calls == ["https://youtu.be/xyz"]
    assert (info.kind, info.title, info.uploader) == ("url", "Some Title", "Some Channel")
