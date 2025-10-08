from __future__ import annotations
import csv
import hashlib
import json
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import psutil


def log_memory(prefix: str = "") -> None:
    """Log current memory usage (RSS) in MB for debugging."""
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[MEM] {prefix} – RSS: {rss:.1f} MB")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a succinct format.

    Prefer logging over print to make the pipeline verifiable and quietable.
    """
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{minutes:02d}:{secs:02d}:{millis:03d}"


def load_existing_md5s(jsonl_path: str = "metadata.jsonl") -> set:
    if not os.path.exists(jsonl_path):
        return set()
    md5s = set()
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                entry = json.loads(line)
                md5s.add(entry.get("md5"))
            except Exception:
                continue
    return md5s


def unique_path(path: str) -> Tuple[str, str]:
    """Return a unique file path (path, basename) without overwriting.

    If *path* exists, append (2), (3), … before the extension.
    """
    base, ext = os.path.splitext(path)
    candidate = path
    counter = 2
    while os.path.exists(candidate):
        candidate = f"{base}({counter}){ext}"
        counter += 1
    return candidate, os.path.basename(candidate)


def append_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_input_csv(path: str) -> Dict[str, Dict]:
    """Read the input mapping CSV into a nested dict keyed by song_label.

    Expected columns: filepath, song_label, song_title, version_disc, version_track
    """
    songs: Dict[str, Dict] = {}
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            songs.setdefault(
                row["song_label"], {"title": row["song_title"], "versions": []}
            )
            songs[row["song_label"]]["versions"].append(row)
    return songs
