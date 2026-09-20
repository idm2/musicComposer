"""Stage: ingest. YouTube / video / audio file → work/<song>/00-source.wav + 00-source.json.

Format-normalises only (44.1 kHz stereo PCM — what Demucs wants). Deliberately does NOT
loudness-normalise: loudness is something the analyser measures.
"""
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .audio import probe_duration, to_wav
from .hashing import file_sha1
from .jsonio import read_json, write_model
from .models import SourceInfo
from .paths import SongPaths

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".wma"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def classify_source(src: str) -> Literal["url", "audio", "video"]:
    if src.lower().startswith(("http://", "https://")):
        return "url"
    ext = Path(src).suffix.lower()
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    raise ValueError(f"unsupported source type {ext!r}: {src}")


def _download(url: str, dest_dir: Path) -> tuple[Path, dict]:
    """yt-dlp best audio → dest_dir. Returns (file, metadata)."""
    proc = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--no-playlist", "-f", "bestaudio/best",
         "-o", str(dest_dir / "download.%(ext)s"), "-j", "--no-simulate", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {proc.stderr[-600:]}")
    files = [f for f in dest_dir.glob("download.*") if f.suffix != ".part"]
    if not files:
        raise RuntimeError("yt-dlp reported success but wrote no file")
    meta = json.loads(proc.stdout.strip().splitlines()[-1])
    return files[0], {"title": meta.get("title"), "uploader": meta.get("uploader") or meta.get("channel")}


def run_ingest(song: str, src: str, force: bool = False) -> SourceInfo:
    paths = SongPaths(song)
    if paths.source_wav.exists() and paths.source_json.exists() and not force:
        print(f"- source exists at {paths.source_wav} (use --force to re-ingest)")
        return SourceInfo(**read_json(paths.source_json))

    kind = classify_source(src)
    meta: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "url":
            print(f"> downloading {src}")
            local, meta = _download(src, Path(tmp))
        else:
            local = Path(src)
            if not local.exists():
                raise FileNotFoundError(local)
        print(f"> normalising -> {paths.source_wav}")
        to_wav(local, paths.source_wav)

    info = SourceInfo(
        origin=src, kind=kind, title=meta.get("title"), uploader=meta.get("uploader"),
        duration_s=round(probe_duration(paths.source_wav), 3), sample_rate=44100,
        sha1=file_sha1(paths.source_wav), ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    write_model(paths.source_json, info)
    print(f"  {info.duration_s:.1f}s  sha1={info.sha1[:12]}  -> {paths.source_json}")
    return info
