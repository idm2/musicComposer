"""Accuracy against a published chart in MIREX .lab format ('onset end Harte-label' per line)."""
from .chordsym import parse_harte
from .models import LOW_CONFIDENCE, Chord


def parse_lab(text: str) -> list[tuple[float, float, str]]:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parse_harte(parts[2]) is not None:
            out.append((float(parts[0]), float(parts[1]), parts[2]))
    return out


def score_chords(found: list[Chord], truth: list[tuple[float, float, str]]) -> dict:
    total = sum(e - s for s, e, _ in truth) or 1.0
    root = full = confident_wrong = 0.0
    for s, e, lab in truth:
        want = parse_harte(lab)
        for c in found:
            shared = min(e, c.onset + c.duration) - max(s, c.onset)
            if shared <= 0:
                continue
            if c.root == want.root:
                root += shared
                full += shared if c.quality == want.quality else 0.0
            elif c.confidence >= LOW_CONFIDENCE:
                confident_wrong += shared
    return {"root": root / total, "full": full / total, "high_confidence_wrong": confident_wrong / total}
