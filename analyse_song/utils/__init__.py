"""Basement: core building blocks for the song analysis pipeline."""

from .config import LABEL_COLORS, VERSION_LABEL_MAP, load_runtime_config
from .models import SongVersion, SongAnalysis, PlotBundle
from .io_utils import (
    compute_md5,
    format_duration,
    load_existing_md5s,
    unique_path,
    append_jsonl,
    setup_logging,
    read_input_csv,
)
from .audio import (
    convert_to_wav,
    extract_features,
    attach_pitch_features,
)
from .plotting import (
    plot_waveforms,
    plot_spectrograms,
    plot_mel_spectrograms,
    plot_similarity_matrix,
    plot_radar_chart,
    plot_lr_balance_bars,
)
from .reporting import generate_markdown, smoke_check_outputs
from .pipeline import SongAnalyser, pick_first_valid

__all__ = [
    "LABEL_COLORS",
    "VERSION_LABEL_MAP",
    "load_runtime_config",
    "SongVersion",
    "SongAnalysis",
    "PlotBundle",
    "compute_md5",
    "format_duration",
    "load_existing_md5s",
    "unique_path",
    "append_jsonl",
    "setup_logging",
    "read_input_csv",
    "convert_to_wav",
    "extract_features",
    "attach_pitch_features",
    "plot_waveforms",
    "plot_spectrograms",
    "plot_mel_spectrograms",
    "plot_similarity_matrix",
    "plot_radar_chart",
    "plot_lr_balance_bars",
    "generate_markdown",
    "smoke_check_outputs",
    "SongAnalyser",
    "pick_first_valid",
]
