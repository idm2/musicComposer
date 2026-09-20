import io
import json
import os
import subprocess
import sys

from songcomposer.cli import _force_utf8_console


def _run_cli(args, cwd):
    """Invoke the real console-script entry point as a subprocess, with stdout/stderr
    forced to cp1252 the way a legacy Windows console (or a redirected/piped stream)
    would — this is what exposed the UnicodeEncodeError."""
    return subprocess.run(
        [sys.executable, "-c", "from songcomposer.cli import main; main()", *args],
        cwd=cwd,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
        capture_output=True,
    )


def test_help_renders_under_cp1252_and_lists_stages(tmp_path):
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    stdout = result.stdout.decode("utf-8")
    assert "ingest" in stdout
    assert "pick" in stdout


def test_missing_upstream_file_prints_friendly_error_not_a_traceback(tmp_path):
    result = _run_cli(["pick", "demo", "--take", "1"], cwd=tmp_path)
    assert result.returncode == 1
    stderr = result.stderr.decode("utf-8")
    assert "error:" in stderr
    assert "songcomposer generate demo" in stderr
    assert "Traceback" not in stderr


def test_invalid_song_name_prints_friendly_error_not_a_traceback(tmp_path):
    result = _run_cli(["pick", "Bad Name", "--take", "1"], cwd=tmp_path)
    assert result.returncode == 1
    stderr = result.stderr.decode("utf-8")
    assert "error:" in stderr
    assert "kebab-case" in stderr
    assert "Traceback" not in stderr


def test_run_without_from_or_brief_prints_friendly_error_not_a_traceback(tmp_path):
    result = _run_cli(["run", "demo"], cwd=tmp_path)
    assert result.returncode == 1
    stderr = result.stderr.decode("utf-8")
    assert "error:" in stderr
    assert "--from" in stderr and "--brief" in stderr
    assert "Traceback" not in stderr


def test_run_help_says_from_is_optional(tmp_path):
    result = _run_cli(["run", "--help"], cwd=tmp_path)
    assert result.returncode == 0
    stdout = result.stdout.decode("utf-8")
    assert "optional" in stdout.lower()


def test_generate_with_fidelity_but_no_reference_prints_friendly_error(tmp_path):
    spec_dir = tmp_path / "work" / "demo"
    spec_dir.mkdir(parents=True)
    (spec_dir / "03-spec.json").write_text(json.dumps({
        "title": "T", "style_prompt": "folk", "target_duration_s": 120,
        "sections": [{"name": "Verse 1", "lines": ["a"], "duration_s": 60},
                     {"name": "Chorus", "lines": ["b"], "duration_s": 60}]}), encoding="utf-8")
    result = _run_cli(["generate", "demo", "--fidelity", "medium"], cwd=tmp_path)
    assert result.returncode == 1
    stderr = result.stderr.decode("utf-8")
    assert "error:" in stderr and "reference" in stderr
    assert "Traceback" not in stderr


def test_force_utf8_console_survives_cp1252_environment(monkeypatch):
    """On Windows, PYTHONIOENCODING (or the console codepage) can leave sys.stdout
    bound to cp1252, which cannot encode characters like -> " " -- x ... that later
    stages print, raising UnicodeEncodeError. _force_utf8_console() must reconfigure
    the stream to UTF-8 regardless of what encoding it started with."""
    buffer = io.BytesIO()
    wrapper = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", wrapper)

    _force_utf8_console()

    text = "→ “café” —"
    print(text)
    wrapper.flush()

    assert buffer.getvalue().decode("utf-8").rstrip("\r\n") == text
