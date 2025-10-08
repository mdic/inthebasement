from __future__ import annotations
import os
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.metrics.pairwise import cosine_similarity
import gc

from .config import LABEL_COLORS
from .io_utils import unique_path
from .models import SongVersion


def _downsample_for_plot(y: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """Downsample a signal to at most max_points for efficient plotting."""
    if y is None:
        return np.array([])
    if len(y) <= max_points:
        return y
    step = max(1, len(y) // max_points)
    return y[::step]


def plot_waveforms(versions: List[SongVersion], outdir: str, song_label: str) -> None:
    # MONO
    plt.figure(figsize=(10, 6))
    for v in versions:
        color = LABEL_COLORS.get(v.label)
        y_plot = _downsample_for_plot(v.signal_mono)
        t = np.linspace(0, v.duration_sec, num=len(y_plot))
        plt.plot(t, y_plot.astype(np.float32), alpha=0.6, label=v.label, color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Mono)")
    plt.tight_layout()
    path, name = unique_path(os.path.join(outdir, f"{song_label}-waveforms_Mono.png"))
    plt.savefig(path, dpi=150)
    for v in versions:
        v.plots["waveform_mono"] = name
    plt.close("all")

    # LEFT
    plt.figure(figsize=(10, 6))
    for v in versions:
        color = LABEL_COLORS.get(v.label)
        y_plot = _downsample_for_plot(v.signal_left)
        t = np.linspace(0, v.duration_sec, num=len(y_plot))
        plt.plot(t, y_plot.astype(np.float32), alpha=0.6, label=v.label, color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Left)")
    plt.tight_layout()
    path, name = unique_path(os.path.join(outdir, f"{song_label}-waveforms_L.png"))
    plt.savefig(path, dpi=150)
    for v in versions:
        v.plots["waveform_L"] = name
    plt.close("all")

    # RIGHT
    plt.figure(figsize=(10, 6))
    for v in versions:
        color = LABEL_COLORS.get(v.label)
        y_plot = _downsample_for_plot(v.signal_right)
        t = np.linspace(0, v.duration_sec, num=len(y_plot))
        plt.plot(t, y_plot.astype(np.float32), alpha=0.6, label=v.label, color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Right)")
    plt.tight_layout()
    path, name = unique_path(os.path.join(outdir, f"{song_label}-waveforms_R.png"))
    plt.savefig(path, dpi=150)
    for v in versions:
        v.plots["waveform_R"] = name
    plt.close("all")


def plot_spectrograms(
    versions: List[SongVersion], outdir: str, song_title: str, song_label: str
) -> None:
    """Plot full-length spectrograms efficiently with lower FFT size and float32 precision."""
    for v in versions:
        for sig, name_suffix, label in [
            (v.signal_mono, "Mono", "Mono"),
            (v.signal_left, "L", "Left"),
            (v.signal_right, "R", "Right"),
        ]:
            # Compute spectrogram with smaller FFT and 32-bit precision
            D = np.abs(librosa.stft(sig, n_fft=1024, hop_length=512)).astype(np.float32)
            plt.figure(figsize=(8, 5))
            librosa.display.specshow(
                librosa.amplitude_to_db(D, ref=np.max),
                sr=v.sr,
                x_axis="time",
                y_axis="log",
                cmap="magma",
            )
            plt.colorbar(format="%+2.0f dB")
            plt.title(
                f"STFT Spec. ({name_suffix}) | {v.label} - {os.path.basename(v.file)} - {song_label}"
            )
            plt.tight_layout()
            path, name = unique_path(
                os.path.join(
                    outdir, f"{song_label}-{v.label}_spectrogram_{name_suffix}.png"
                )
            )
            plt.savefig(path, dpi=150)
            v.plots[f"spectrogram_{name_suffix}"] = name
            plt.close("all")
            del D
            gc.collect()


def plot_mel_spectrograms(
    versions: List[SongVersion], outdir: str, song_title: str, song_label: str
) -> None:
    for v in versions:
        # MONO
        Sm = librosa.feature.melspectrogram(y=v.signal_mono, sr=v.sr, n_mels=128)
        Sm_dB = librosa.power_to_db(Sm, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sm_dB, sr=v.sr, x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        path, name = unique_path(
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_Mono.png")
        )
        plt.title(
            f"Mel Spec. (Mono) | {v.label} - {os.path.basename(v.file)} - {song_label}"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        v.plots["melspec_Mono"] = name
        plt.close("all")
        del Sm, Sm_dB
        gc.collect()

        # LEFT
        Sl = librosa.feature.melspectrogram(y=v.signal_left, sr=v.sr, n_mels=128)
        Sl_dB = librosa.power_to_db(Sl, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sl_dB, sr=v.sr, x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        path, name = unique_path(
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_L.png")
        )
        plt.title(
            f"Mel Spec. (L) | {v.label} - {os.path.basename(v.file)} - {song_label}"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        v.plots["melspec_L"] = name
        plt.close("all")
        del Sl, Sl_dB
        gc.collect()

        # RIGHT
        Sr = librosa.feature.melspectrogram(y=v.signal_right, sr=v.sr, n_mels=128)
        Sr_dB = librosa.power_to_db(Sr, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sr_dB, sr=v.sr, x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        path, name = unique_path(
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_R.png")
        )
        plt.title(
            f"Mel Spec. (R) | {v.label} - {os.path.basename(v.file)} - {song_label}"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        v.plots["melspec_R"] = name
        plt.close("all")
        del Sr, Sr_dB
        gc.collect()


def plot_similarity_matrix(
    versions: List[SongVersion], outdir: str, song_label: str
) -> str:
    mfcc_matrix = np.array([v.mfcc_mean for v in versions])
    sim = cosine_similarity(mfcc_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(
        range(len(versions)), [v.label for v in versions], rotation=45, ha="right"
    )
    plt.yticks(range(len(versions)), [v.label for v in versions])
    plt.title("MFCC Similarity Matrix")
    plt.tight_layout()
    path, name = unique_path(
        os.path.join(outdir, f"{song_label}-similarity_matrix.png")
    )
    plt.savefig(path, dpi=150)
    plt.close()

    # CSV alongside
    df = pd.DataFrame(
        sim, index=[v.label for v in versions], columns=[v.label for v in versions]
    )
    df.to_csv(os.path.join(outdir, f"{song_label}-similarity_matrix.csv"))

    # stash on the first version for markdown convenience
    versions[0].plots["similarity_plot"] = name
    return name


def plot_radar_chart(versions: List[SongVersion], outdir: str, song_label: str) -> str:
    metrics = ["duration_sec", "loudness", "rms", "spectral_centroid"]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    eps = 1e-9
    percentiles = {
        m: (
            np.percentile([float(getattr(v, m)) for v in versions], 5),
            np.percentile([float(getattr(v, m)) for v in versions], 95),
        )
        for m in metrics
    }

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    markers = ["o", "x", "^", "s", "D", "v", "*", "P", "X", ">", "<"]
    marker_idx = 0
    seen_vals = {}

    def normalise(v, m):
        pmin, pmax = percentiles[m]
        x = float(getattr(v, m))
        x = min(max(x, pmin), pmax)
        return (x - pmin) / (pmax - pmin + eps)

    for v in versions:
        vals = np.array([normalise(v, m) for m in metrics])
        vals = np.concatenate([vals, vals[:1]])
        key = tuple(np.round(vals, 6))
        if key in seen_vals:
            linestyle = "--"
            marker = markers[marker_idx % len(markers)]
            marker_idx += 1
            fill_alpha = 0.08
            vals = vals + np.random.uniform(-0.01, 0.01, size=vals.shape)
        else:
            linestyle = "-"
            marker = None
            fill_alpha = 0.15
            seen_vals[key] = v.label

        color = LABEL_COLORS.get(v.label)
        ax.fill(angles, vals, alpha=fill_alpha, color=color, zorder=1)
        ax.plot(
            angles,
            vals,
            label=v.label,
            color=color,
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=3,
            solid_capstyle="round",
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Radar Plot of Main Features (5–95 percentile scaling)")
    plt.tight_layout()
    path, name = unique_path(os.path.join(outdir, f"{song_label}-radar_plot.png"))
    plt.savefig(path, dpi=150)

    # Overlap diagnostics
    dup_notes = []
    for key, label in seen_vals.items():
        overlaps = [
            v.label
            for v in versions
            if tuple(
                np.round(np.concatenate([[normalise(v, m) for m in metrics], [0]]), 6)
            )
            in seen_vals
            and seen_vals[key] != v.label
        ]
        if overlaps:
            dup_notes.append(
                f"{label} overlaps with {', '.join(sorted(set(overlaps)))} in radar plot"
            )
    if dup_notes:
        with open(
            os.path.join(outdir, f"{song_label}-radar_duplicates.txt"),
            "w",
            encoding="utf-8",
        ) as fh:
            fh.write("\n".join(dup_notes))

    versions[0].plots["radar_plot"] = name
    plt.close()
    return name


def plot_lr_balance_bars(
    versions: List[SongVersion], outdir: str, song_label: str
) -> None:
    for v in versions:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(2)
        ax1.bar(
            x - 0.2,
            [v.rms_left, v.rms_right],
            width=0.4,
            color="blue",
            alpha=0.7,
            label="RMS",
        )
        ax1.set_ylabel("RMS (avg)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.bar(
            x + 0.2,
            [v.loudness_left, v.loudness_right],
            width=0.4,
            color="orange",
            alpha=0.7,
            label="Loudness (LUFS)",
        )
        ax2.set_ylabel("Loudness (LUFS)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        ax1.set_xticks(x)
        ax1.set_xticklabels(["L", "R"])
        plt.title(
            f"Stereo Balance | {v.label} - {os.path.basename(v.file)} - {song_label}"
        )
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")
        fig.tight_layout()
        path, name = unique_path(
            os.path.join(outdir, f"{song_label}-{v.label}_balance.png")
        )
        plt.savefig(path, dpi=150)
        v.plots["balance_plot"] = name
        plt.close()
