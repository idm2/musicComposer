import io

from songcomposer.cli import _force_utf8_console


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
