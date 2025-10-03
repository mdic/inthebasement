from __future__ import annotations
import json
import os
from typing import Dict, Any

# Default in-code mappings. These can be overridden at runtime by placing
# a JSON file at CONFIG_PATH (env) or utils/basement/config.json.
LABEL_COLORS: Dict[str, str] = {
    "HHST": "skyblue",
    "TGBT": "orange",
    "CAR": "green",
    "TCB": "purple",
    "MUTM": "red",
    "ATWR": "blue",
}

VERSION_LABEL_MAP: Dict[str, str] = {
    "A Tree With Roots": "ATWR",
    "The Basement Tapes - Captain Acid 2020 restoration": "CAR",
    "Basement Tapes Anthology [FLAC]": "CB",
    "Complete Basement Safety Tape": "CBST",
    "Down In The Basement": "DITB",
    "From The Reels": "FTR",
    "Mixin Up The Medicine": "MUTM",
    "The Basement Tape RSD (24 Bit Vinyl FLAC)": "RSD",
    "Sweet Bourbon Daddy": "SBD",
    "The Genuine Basement Tapes": "TGBT",
    "The Bootleg Series Vol. 11": "BS11",
}

RUNTIME_CONFIG: Dict[str, Any] = {
    "LABEL_COLORS": LABEL_COLORS,
    "VERSION_LABEL_MAP": VERSION_LABEL_MAP,
}


def load_runtime_config() -> Dict[str, Any]:
    """Load optional JSON config to override defaults.

    If the file exists at ENV CONFIG_PATH or alongside this module as
    ``config.json``, merge it shallowly over the defaults. This keeps the
    code-based defaults but allows quick edits without touching Python.
    """
    cfg_path = os.environ.get("BASEMENT_CONFIG_PATH") or os.path.join(
        os.path.dirname(__file__), "config.json"
    )
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                user = json.load(fh)
            for k, v in user.items():
                if isinstance(v, dict) and k in RUNTIME_CONFIG:
                    RUNTIME_CONFIG[k].update(v)
                else:
                    RUNTIME_CONFIG[k] = v
        except Exception:
            # Soft-fail; callers still get defaults
            pass
    return RUNTIME_CONFIG
