import re
from pathlib import Path

_SONG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SongPaths:
    """Every file location for one song. Mirrors BUILD-SPEC §3."""

    def __init__(self, song: str, root: Path | None = None):
        if not _SONG_RE.match(song or ""):
            raise ValueError(f"song name must be lowercase kebab-case (a-z, 0-9, -): {song!r}")
        root = Path(root) if root else Path.cwd()
        self.song = song
        self.work = root / "work" / song
        self.out = root / "out" / song
        self.cache = self.work / ".cache"
        self.source_wav = self.work / "00-source.wav"
        self.source_json = self.work / "00-source.json"
        self.analysis = self.work / "01-analysis.json"
        self.brief = self.work / "02-brief.json"
        self.spec = self.work / "03-spec.json"
        self.takes_dir = self.work / "04-takes"
        self.takes_json = self.takes_dir / "takes.json"
        self.chosen = self.work / "05-chosen.json"
        self.transcription = self.work / "06-transcription.json"
        self.out_mp3 = self.out / f"{song}.mp3"
        self.out_musicxml = self.out / f"{song}.musicxml"
        self.chords_txt = self.out / "chords.txt"
        self.tab_txt = self.out / "tab.txt"
        self.chart_ly = self.out / "chart.ly"
        self.chart_pdf = self.out / "chart.pdf"
        self.lyrics_json = self.out / "lyrics.json"

    def take(self, index: int) -> Path:
        return self.takes_dir / f"take-{index}.mp3"

    def require(self, path: Path, made_by: str) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run `songcomposer {made_by} {self.song}` first.")
        return path
