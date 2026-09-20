"""Stage: pick. Only the chosen take is ever transcribed."""
import shutil
from datetime import datetime, timezone

from .hashing import file_sha1
from .jsonio import read_json, write_model
from .models import Chosen, TakesManifest
from .paths import SongPaths


def run_pick(song: str, take: int) -> Chosen:
    paths = SongPaths(song)
    manifest = TakesManifest(**read_json(paths.require(paths.takes_json, "generate")))
    by_index = {t.index: t for t in manifest.all_takes()}
    if take not in by_index or not (paths.takes_dir / by_index[take].file).exists():
        raise ValueError(f"no take {take} — available: {', '.join(str(i) for i in sorted(by_index))}")
    record = by_index[take]
    src = paths.takes_dir / record.file
    chosen = Chosen(take=take, file=record.file, provider=record.provider, sha1=file_sha1(src),
                    chosen_at=datetime.now(timezone.utc).isoformat())
    if paths.chosen.exists() and read_json(paths.chosen).get("sha1") != chosen.sha1:
        paths.transcription.unlink(missing_ok=True)
    write_model(paths.chosen, chosen)
    paths.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, paths.out_mp3)
    print(f"  take {take} ({record.provider}) → {paths.out_mp3}")
    return chosen
