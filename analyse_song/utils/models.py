from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class SongVersion:
    """Typed container for a single version of a song.

    Using a dataclass instead of ad-hoc dicts increases clarity and safety.
    """

    label: str
    file: str
    disc: int
    track: int
    song_title: Optional[str] = None
    # Filled during processing
    md5: Optional[str] = None
    wav_path: Optional[str] = None
    sr: Optional[int] = None
    signal_mono: Optional[np.ndarray] = None
    signal_left: Optional[np.ndarray] = None
    signal_right: Optional[np.ndarray] = None
    duration_sec: Optional[float] = None
    duration_fmt: Optional[str] = None
    loudness: Optional[float] = None
    loudness_left: Optional[float] = None
    loudness_right: Optional[float] = None
    loudness_balance: Optional[float] = None
    rms: Optional[float] = None
    rms_left: Optional[float] = None
    rms_right: Optional[float] = None
    rms_balance: Optional[float] = None
    lr_corr: Optional[float] = None
    spectral_centroid: Optional[float] = None
    mfcc_mean: Optional[np.ndarray] = None
    tuning_cents: Optional[float] = None
    chroma_mean: Optional[np.ndarray] = None
    # Plot filenames (assigned by plotting functions)
    plots: Dict[str, str] = field(default_factory=dict)


@dataclass
class PlotBundle:
    """Collects global plot artefacts shared across versions for a song."""

    similarity_plot: Optional[str] = None
    radar_plot: Optional[str] = None
    pitch_plot: Optional[str] = None


@dataclass
class SongAnalysis:
    """Aggregated analysis result for a song, suitable for reporting."""

    song_label: str
    song_title: str
    versions: List[SongVersion]
    outdir: str
    plots: PlotBundle = field(default_factory=PlotBundle)
