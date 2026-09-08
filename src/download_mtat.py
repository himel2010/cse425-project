"""P1.1 — Download MagnaTagATune (audio zips + CSVs), extract, verify.

Usage: python src/download_mtat.py
Idempotent: skips files already present; re-run safe.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

from utils import load_config, ensure_dir, update_metrics

# Mirror order (PLAN.md P1.1). Each entry: base URL for the file set.
MIRRORS = [
    "https://mi.soi.city.ac.uk/datasets/magnatagatune/",
    "https://mirg.city.ac.uk/datasets/magnatagatune/",
]
AUDIO_PARTS = ["mp3.zip.001", "mp3.zip.002", "mp3.zip.003"]
CSVS = ["annotations_final.csv", "clip_info_final.csv"]


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> bool:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total:
                    pct = 100 * got / total
                    print(f"\r  {dest.name}: {got>>20}/{total>>20} MB ({pct:4.1f}%)",
                          end="", flush=True)
        print()
        return dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        print(f"\n  FAILED {url}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def fetch(name: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {name}")
        return True
    for base in MIRRORS:
        print(f"  trying {base}{name}")
        if _download(base + name, dest):
            return True
    return False


def main() -> int:
    cfg = load_config()
    root = ensure_dir(cfg["datasets"]["mtat"]["root"])
    audio_dir = ensure_dir(cfg["datasets"]["mtat"]["audio_dir"])

    # CSVs
    for csv in CSVS:
        if not fetch(csv, root / csv):
            print(f"ERROR: could not download {csv} from any mirror.")
            return 1

    # Audio parts
    for part in AUDIO_PARTS:
        if not fetch(part, root / part):
            print(f"ERROR: could not download {part} from any mirror.")
            return 1

    # Concatenate the 3 split-zip parts into one, then extract.
    merged = root / "mp3_all.zip"
    if not merged.exists():
        print("  merging zip parts...")
        with open(merged, "wb") as out:
            for part in AUDIO_PARTS:
                with open(root / part, "rb") as p:
                    while True:
                        b = p.read(1 << 20)
                        if not b:
                            break
                        out.write(b)

    # Extract if audio folders not already present
    have_folders = any((audio_dir / f).exists() for f in "0123456789abcdef")
    if not have_folders:
        print("  extracting audio (this takes a while)...")
        with zipfile.ZipFile(merged) as z:
            z.extractall(audio_dir)

    # Verify
    mp3s = list(audio_dir.rglob("*.mp3"))
    n = len(mp3s)
    print(f"\nmp3 count: {n} (expected ~25863)")
    update_metrics("dataset.mtat", {
        "mp3_count": n,
        "expected": 25863,
        "audio_dir": str(audio_dir),
    })
    if n < 20000:
        print("WARNING: mp3 count low; extraction may be incomplete.")
        return 1
    print("MTAT download OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
