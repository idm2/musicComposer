import pytest

from songcomposer import chart
from songcomposer.jsonio import read_json, write_model
from songcomposer.models import SongSpec, SpecSection
from songcomposer.paths import SongPaths
from songcomposer.render.lyricsjson import build_lyrics_json
from test_musicxml import t  # noqa: F401


def test_lyrics_json_contract(t):  # noqa: F811
    t.lines.append(t.lines[0].model_copy(update={"text": "never sung", "start": None, "end": None, "words": [], "confidence": 0.0}))
    out = build_lyrics_json("glass-hour", "Glass Hour", t)
    assert (out["schema_version"], out["audio"], out["duration_s"]) == (1, "glass-hour.mp3", 9.6)
    assert out["lines"][0]["words"][0] == {"word": "Glass", "start": 0.62, "end": 1.1, "confidence": 0.9}
    assert out["unaligned_lines"] == ["never sung"] and len(out["lines"]) == 1


def test_chart_writes_every_deliverable_into_out(root, t, monkeypatch):  # noqa: F811
    pytest.importorskip("music21")
    p = SongPaths("demo")
    write_model(p.spec, SongSpec(title="Glass Hour", style_prompt="s", target_duration_s=60,
                                 sections=[SpecSection(name="V", lines=["Glass hour"], duration_s=60)]))
    monkeypatch.setattr(chart, "run_transcribe", lambda song, force=False: t)
    written = chart.run_chart("demo")
    for path in (p.chords_txt, p.tab_txt, p.out_musicxml, p.lyrics_json, p.chart_ly):
        assert path.exists(), path
    assert set(written) >= {"chords", "tab", "musicxml", "lyrics"}
    assert all(root / "out" / "demo" in path.parents for path in written.values())       # deliverables stay here
    assert "Bbm7?" in p.chords_txt.read_text(encoding="utf-8")
    assert read_json(p.lyrics_json)["song"] == "demo"


def test_import_chart_does_not_import_heavy_libs():
    """Architectural pin: chart.py wires up renderers but every heavy dependency they use
    (music21, torch, librosa, demucs, faster_whisper, basic_pitch, lv_chordia) must stay lazy."""
    import subprocess
    import sys

    heavy = ["music21", "torch", "librosa", "demucs", "faster_whisper", "basic_pitch", "lv_chordia"]
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import songcomposer.chart; "
         f"loaded = [m for m in {heavy!r} if m in sys.modules]; "
         "assert not loaded, f'chart.py imported heavy libs eagerly: {loaded}'"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
