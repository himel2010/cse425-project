"""P1.2 — Fetch MusicCaps audio from the HuggingFace mirror CLAPv2/MusicCaps.

YouTube blocks yt-dlp with bot-detection ("Sign in to confirm you're not a bot")
and Chromium cookie decryption fails (DPAPI app-bound encryption). The CLAPv2
mirror ships the actual 10 s clips as WAV inside parquet, so we bypass YouTube
entirely. Captions still come from musiccaps-public.csv (keyed by ytid).

Usage: python src/download_musiccaps.py [--limit N]
Idempotent: skips ytids whose wav already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import soundfile as sf

from utils import load_config, ensure_dir, update_metrics

HF_REPO = "CLAPv2/MusicCaps"
CSV_SOURCES = [
    "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv",
    "https://raw.githubusercontent.com/google-research/google-research/master/musiccaps/musiccaps-public.csv",
]


def _fetch_csv(dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for url in CSV_SOURCES:
        try:
            print(f"  csv try: {url}")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if dest.stat().st_size > 0:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"    failed: {e}")
    return False


def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    import librosa
    return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    mc = cfg["datasets"]["musiccaps"]
    root = ensure_dir(mc["root"])
    wav_dir = ensure_dir(mc["wav_dir"])
    sr_out = mc["sample_rate"]

    if not _fetch_csv(Path(mc["csv"])):
        print("WARNING: could not fetch musiccaps-public.csv (captions).")

    from datasets import load_dataset, Audio
    print(f"Streaming {HF_REPO} (audio) from HuggingFace ...")
    ds = load_dataset(HF_REPO, split="train", streaming=True)
    # decode=False -> raw file bytes; we decode with soundfile (no torchcodec dep)
    ds = ds.cast_column("audio", Audio(decode=False))

    ok, fail = 0, 0
    for i, ex in enumerate(ds):
        if args.limit and i >= args.limit:
            break
        # CLAPv2 mirror keys the clip by 'index' = "{ytid}.wav".
        audio = ex.get("audio", {})
        ytid = ex.get("ytid") or ex.get("id")
        if not ytid:
            key = ex.get("index") or (audio.get("path") if isinstance(audio, dict) else "")
            ytid = Path(key or "").stem
        if not ytid:
            fail += 1
            continue
        out = wav_dir / f"{ytid}.wav"
        if out.exists() and out.stat().st_size > 0:
            ok += 1
            continue
        try:
            import io
            arr, sr_in = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if arr.ndim > 1:            # stereo -> mono
                arr = arr.mean(axis=1)
            arr = _resample(arr, int(sr_in), sr_out)
            sf.write(out, arr, sr_out)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  write fail {ytid}: {e}")
        if (i + 1) % 200 == 0:
            print(f"\r  {i+1}  ok={ok} fail={fail}", end="", flush=True)
    print()

    update_metrics("dataset.musiccaps", {
        "source": HF_REPO,
        "downloaded": ok,
        "failed": fail,
        "wav_dir": str(wav_dir),
    })
    print(f"MusicCaps: {ok} wavs written, {fail} failed (source={HF_REPO})")
    if ok < 2000:
        print("ERROR: < 2000 clips — escalate per PLAN.md.")
        return 1
    if ok < 4000:
        print("WARNING: < 4000 clips — proceeding, note in report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
