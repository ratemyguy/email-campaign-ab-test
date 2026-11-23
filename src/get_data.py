"""Download the MineThatData e-mail challenge CSV (if missing) and verify it.

The raw file is treated as untrusted data: it is only ever read as CSV text,
never executed, and its SHA-256 must match the copy this project was built on.

    python -I src/get_data.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` skips the script's folder

import requests  # noqa: E402

import config as cfg  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> Path:
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not cfg.RAW_CSV.exists():
        print(f"Downloading {cfg.DATA_URL}")
        resp = requests.get(cfg.DATA_URL, timeout=120)
        resp.raise_for_status()
        tmp = cfg.RAW_CSV.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(cfg.RAW_CSV)
    digest = sha256(cfg.RAW_CSV)
    if digest != cfg.DATA_SHA256:
        raise SystemExit(
            f"Checksum mismatch for {cfg.RAW_CSV}: got {digest}, expected {cfg.DATA_SHA256}. "
            "The source file has changed; inspect it before re-running."
        )
    print(f"Raw data OK: {cfg.RAW_CSV.relative_to(cfg.ROOT)} ({cfg.RAW_CSV.stat().st_size:,} bytes, sha256 verified)")
    return cfg.RAW_CSV


if __name__ == "__main__":
    main()
