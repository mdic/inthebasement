import os
import csv
import subprocess
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import hashlib
import argparse
import sys

# --------------------------------------------
# Colour mapping for version labels
# --------------------------------------------
LABEL_COLORS = {
    "HHST": "skyblue",
    "TGBT": "orange",
    "CAR": "green",
    "TCB": "purple",
    "MUTM": "red",
    "ATWR": "blue",
}

# --------------------------------------------
# Version label mapping from filepath
# (customise as needed)
# --------------------------------------------
VERSION_LABEL_MAP = {
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


# --------------------------------------------
# Utilities
# --------------------------------------------
def derive_version_label(filepath: str) -> str:
    for key, val in VERSION_LABEL_MAP.items():
        if key in filepath:
            return val
    return "UNKNOWN"


def convert_to_wav(infile, outdir="converted", force=False):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(infile))[0]
    outfile = os.path.join(outdir, base + ".wav")

    if force or not os.path.exists(outfile):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                infile,
                "-ar",
                "44100",
                "-ac",
                "2",
                "-sample_fmt",
                "s16",
                outfile,
            ],
            check=True,
        )
    return outfile


def compute_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{secs:02d}:{millis:03d}"


def load_existing_md5s(jsonl_path="metadata.jsonl"):
    if not os.path.exists(jsonl_path):
        return set()
    md5s = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                md5s.add(entry["md5"])
            except Exception:
                continue
    return md5s


def unique_path(path):
    """
    Ensure that the file path does not overwrite an existing file.
    If path exists, append (2), (3), ... before the extension.
    Returns both the final path and the basename (for markdown).
    """
    base, ext = os.path.splitext(path)
    counter = 2
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}({counter}){ext}"
        counter += 1
    return new_path, os.path.basename(new_path)


# --------------------------------------------
# Feature extraction  (UPDATED: stereo-aware)
# --------------------------------------------
def extract_features(wavfile, label, orig_file, version_disc, version_track, md5):
    # Load stereo (do NOT collapse to mono)
    y, sr = librosa.load(wavfile, sr=44100, mono=False)

    # Ensure two channels (duplicate mono to pseudo-stereo)
    if y.ndim == 1:
        y = np.vstack([y, y])

    left, right = y[0], y[1]
    mono = (left + right) / 2

    duration = librosa.get_duration(y=mono, sr=sr)

    # Loudness per channel + mono
    meter = pyln.Meter(sr)
    loudness_left = meter.integrated_loudness(left)
    loudness_right = meter.integrated_loudness(right)
    loudness = meter.integrated_loudness(mono)

    # RMS per channel + mono
    rms_left = float(np.mean(librosa.feature.rms(y=left)))
    rms_right = float(np.mean(librosa.feature.rms(y=right)))
    rms = float(np.mean(librosa.feature.rms(y=mono)))

    # Optional correlation (Pearson) between L and R
    if len(left) == len(right) and len(left) > 1:
        lr_corr = float(np.corrcoef(left, right)[0, 1])
    else:
        lr_corr = np.nan

    # Spectral centroid on MONO (coerenza storica con i confronti)
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

    # MFCC on MONO (per similarità)
    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Balance metrics (diff L-R)
    rms_balance = rms_left - rms_right
    loudness_balance = loudness_left - loudness_right

    return {
        "label": label,
        "file": os.path.basename(wavfile),
        "orig_file": orig_file,
        "disc": int(version_disc),
        "track": int(version_track),
        "duration_sec": duration,
        "duration_fmt": format_duration(duration),
        "loudness": loudness,
        "loudness_left": loudness_left,
        "loudness_right": loudness_right,
        "loudness_balance": loudness_balance,
        "rms": rms,
        "rms_left": rms_left,
        "rms_right": rms_right,
        "rms_balance": rms_balance,
        "lr_corr": lr_corr,
        "spectral_centroid": spectral_centroid,
        "mfcc": mfcc_mean,
        # store signals for plotting
        "signal": mono,
        "signal_left": left,
        "signal_right": right,
        "sr": sr,
        "md5": md5,
    }


# --------------------------------------------
# Plotting  (UPDATED: L/R/Mono variants)
# --------------------------------------------
def plot_waveforms(features, outdir, song_label):
    # Generate three comparative waveform plots: Mono, Left, Right
    # MONO
    plt.figure(figsize=(10, 6))
    for f in features:
        color = LABEL_COLORS.get(f["label"], None)
        t = np.linspace(0, f["duration_sec"], num=len(f["signal"]))
        plt.plot(t, f["signal"], alpha=0.6, label=f["label"], color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Mono)")
    plt.tight_layout()
    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-waveforms_Mono.png")
    )
    plt.savefig(out_path, dpi=150)
    for f in features:
        f["waveform_mono"] = out_name
    plt.close()

    # LEFT
    plt.figure(figsize=(10, 6))
    for f in features:
        color = LABEL_COLORS.get(f["label"], None)
        sig = f["signal_left"]
        t = np.linspace(0, f["duration_sec"], num=len(sig))
        plt.plot(t, sig, alpha=0.6, label=f["label"], color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Left)")
    plt.tight_layout()
    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-waveforms_L.png")
    )
    plt.savefig(out_path, dpi=150)
    for f in features:
        f["waveform_L"] = out_name
    plt.close()

    # RIGHT
    plt.figure(figsize=(10, 6))
    for f in features:
        color = LABEL_COLORS.get(f["label"], None)
        sig = f["signal_right"]
        t = np.linspace(0, f["duration_sec"], num=len(sig))
        plt.plot(t, sig, alpha=0.6, label=f["label"], color=color)
    plt.legend()
    plt.title("Comparative Waveforms (Right)")
    plt.tight_layout()
    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-waveforms_R.png")
    )
    plt.savefig(out_path, dpi=150)
    for f in features:
        f["waveform_R"] = out_name
    plt.close()


def plot_spectrograms(features, outdir, song_title, song_label):
    # For each version: STFT spectrograms for Mono, Left, Right
    for f in features:
        # MONO
        Dm = np.abs(librosa.stft(f["signal"]))
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            librosa.amplitude_to_db(Dm, ref=np.max),
            sr=f["sr"],
            x_axis="time",
            y_axis="log",
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"STFT Spectrogram (Mono) - {f['label']} {song_title}")
        plt.tight_layout()
        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_Mono.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["spectrogram_Mono"] = out_name
        plt.close()

        # LEFT
        Dl = np.abs(librosa.stft(f["signal_left"]))
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            librosa.amplitude_to_db(Dl, ref=np.max),
            sr=f["sr"],
            x_axis="time",
            y_axis="log",
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"STFT Spectrogram (Left) - {f['label']} {song_title}")
        plt.tight_layout()

        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_L.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["spectrogram_L"] = out_name
        plt.close()

        # RIGHT
        Dr = np.abs(librosa.stft(f["signal_right"]))
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            librosa.amplitude_to_db(Dr, ref=np.max),
            sr=f["sr"],
            x_axis="time",
            y_axis="log",
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"STFT Spectrogram (Right) - {f['label']} {song_title}")
        plt.tight_layout()

        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_R.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["spectrogram_R"] = out_name
        plt.close()


def plot_mel_spectrograms(features, outdir, song_title, song_label):
    # For each version: Mel spectrograms for Mono, Left, Right
    for f in features:
        # MONO
        Sm = librosa.feature.melspectrogram(y=f["signal"], sr=f["sr"], n_mels=128)
        Sm_dB = librosa.power_to_db(Sm, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sm_dB, sr=f["sr"], x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(
            f"Mel Spectrogram (Mono) - {f['label']} {os.path.basename(f['orig_file'])} - {song_label}"
        )
        plt.tight_layout()

        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_Mono.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["melspec_Mono"] = out_name
        plt.close()

        # LEFT
        Sl = librosa.feature.melspectrogram(y=f["signal_left"], sr=f["sr"], n_mels=128)
        Sl_dB = librosa.power_to_db(Sl, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sl_dB, sr=f["sr"], x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram (Left) - {f['label']} {song_title}")
        plt.tight_layout()

        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_L.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["melspec_L"] = out_name
        plt.close()

        # RIGHT
        Sr = librosa.feature.melspectrogram(y=f["signal_right"], sr=f["sr"], n_mels=128)
        Sr_dB = librosa.power_to_db(Sr, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            Sr_dB, sr=f["sr"], x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram (Right) - {f['label']} {song_title}")
        plt.tight_layout()
        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_R.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["melspec_R"] = out_name
        plt.close()


def plot_similarity_matrix(features, outdir, song_label):
    # Same as before: use MFCC (mono) for similarity
    mfcc_matrix = np.array([f["mfcc"] for f in features])
    sim = cosine_similarity(mfcc_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(
        range(len(features)), [f["label"] for f in features], rotation=45, ha="right"
    )
    plt.yticks(range(len(features)), [f["label"] for f in features])
    plt.title("MFCC Similarity Matrix")
    plt.tight_layout()

    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-similarity_matrix.png")
    )
    plt.savefig(out_path, dpi=150)
    features[0]["similarity_plot"] = out_name
    plt.close()

    # Save also CSV (unchanged prefix rule)
    sim_df = pd.DataFrame(
        sim,
        index=[f["label"] for f in features],
        columns=[f["label"] for f in features],
    )
    sim_df.to_csv(os.path.join(outdir, f"{song_label}-similarity_matrix.csv"))


def plot_radar_chart(features, outdir, song_label):
    # Metrics da visualizzare
    metrics = ["duration_sec", "loudness", "rms", "spectral_centroid"]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    eps = 1e-9

    # Calcola 5° e 95° percentile per ogni metrica
    percentiles = {
        m: (
            np.percentile([float(f[m]) for f in features], 5),
            np.percentile([float(f[m]) for f in features], 95),
        )
        for m in metrics
    }

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # NEW: marker pool e stato per duplicati
    markers = ["o", "x", "^", "s", "D", "v", "*", "P", "X", ">", "<"]
    marker_idx = 0
    seen_vals = {}

    for f in features:
        vals = []
        for m in metrics:
            pmin, pmax = percentiles[m]
            v = float(f[m])
            # Trimming → clamp valori fuori dal range
            if v < pmin:
                v = pmin
            if v > pmax:
                v = pmax
            # Normalizzazione su [0,1]
            norm = (v - pmin) / (pmax - pmin + eps)
            vals.append(norm)

        vals = np.array(vals)
        vals = np.concatenate([vals, vals[:1]])

        key = tuple(np.round(vals, 6))
        if key in seen_vals:
            linestyle = "--"
            marker = markers[marker_idx % len(markers)]
            marker_idx += 1
            fill_alpha = 0.08
            # NEW: aggiungi jitter minimo per distinguere duplicati
            jitter = np.random.uniform(-0.01, 0.01, size=vals.shape)
            vals = vals + jitter
        else:
            linestyle = "-"
            marker = None
            fill_alpha = 0.15
            seen_vals[key] = f["label"]

        color = LABEL_COLORS.get(f["label"], None)

        # IMPORTANTE: fill prima, line dopo
        ax.fill(angles, vals, alpha=fill_alpha, color=color, zorder=1)

        ax.plot(
            angles,
            vals,
            label=f["label"],
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
    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-radar_plot.png")
    )
    plt.savefig(out_path, dpi=150)
    # salvalo in un campo globale (non per-versione ma per tutta la canzone)
    features[0]["radar_plot"] = out_name

    # here, if lines overlap in radar plot, a note is added to the .md file;
    # here's where the overlap is detected and saved
    dup_notes = []
    for vals, label in seen_vals.items():
        # cerca chi altro ha la stessa chiave
        overlaps = [
            f["label"]
            for f in features
            if tuple(
                np.round(
                    np.concatenate(
                        [
                            [
                                (
                                    float(f[m])
                                    - np.percentile(
                                        [float(ff[m]) for ff in features], 5
                                    )
                                )
                                / (
                                    np.percentile([float(ff[m]) for ff in features], 95)
                                    - np.percentile(
                                        [float(ff[m]) for ff in features], 5
                                    )
                                    + eps
                                )
                                for m in metrics
                            ],
                            [0],
                        ]
                    ),
                    6,
                )
            )
            in seen_vals
            and seen_vals[vals] != f["label"]
        ]
        if overlaps:
            dup_notes.append(
                f"{label} overlaps with {', '.join(set(overlaps))} in radar plot"
            )
    if dup_notes:
        with open(
            os.path.join(outdir, f"{song_label}-radar_duplicates.txt"), "w"
        ) as df:
            for note in dup_notes:
                df.write(note + "\n")
    plt.close()


def plot_lr_balance_bars(features, outdir, song_label):
    for f in features:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        x = np.arange(2)  # L, R

        # RMS bars (blu) su ax1
        rms_vals = [f["rms_left"], f["rms_right"]]
        ax1.bar(x - 0.2, rms_vals, width=0.4, color="blue", alpha=0.7, label="RMS")
        ax1.set_ylabel("RMS (avg)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Loudness (LUFS) su ax2 (arancione)
        ax2 = ax1.twinx()
        lufs_vals = [f["loudness_left"], f["loudness_right"]]
        ax2.bar(
            x + 0.2,
            lufs_vals,
            width=0.4,
            color="orange",
            alpha=0.7,
            label="Loudness (LUFS)",
        )
        ax2.set_ylabel("Loudness (LUFS)", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        ax1.set_xticks(x)
        ax1.set_xticklabels(["L", "R"])
        plt.title(f"Stereo Balance — {song_label} {f['label']}")

        # Legenda combinata
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        fig.tight_layout()
        out_path, out_name = unique_path(
            os.path.join(outdir, f"{song_label}-{f['label']}_balance.png")
        )
        plt.savefig(out_path, dpi=150)
        # for f in features:
        f["balance_plot"] = out_name
        plt.close()


# --- PITCH & SPEED ANALYSIS ----------------------------------------------


def _compute_tuning_and_chroma(y, sr):
    """
    Compute global tuning (in cents) and time-averaged chroma vector (12-dim).
    Uses CQT chroma, robust to timbre; average across time for global profile.
    """
    # Librosa's estimate_tuning returns number of bins (1/12 tone) offset.
    # Convert to cents (100 cents per semitone).
    tuning_bins = librosa.estimate_tuning(y=y, sr=sr)  # bins of 1/12 tone
    tuning_cents = float(tuning_bins * 100.0)

    # Beat-synchronous or plain mean chroma: plain mean is simpler and fast
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # shape (12,)
    # Normalize to unit sum to be correlation-friendly
    if np.sum(chroma_mean) > 0:
        chroma_mean = chroma_mean / np.maximum(np.linalg.norm(chroma_mean), 1e-9)
    return tuning_cents, chroma_mean


def _best_semitone_shift(chroma_ref, chroma_cmp):
    """
    Find the circular chroma shift (in semitones) that maximizes cosine similarity.
    Returns an integer shift in [-6, +5] with sign convention:
    Positive shift means 'cmp' is higher than 'ref' by that many semitones.
    """
    best_shift = 0
    best_sim = -1e9
    for k in range(12):
        # roll cmp upwards by k semitones
        rolled = np.roll(chroma_cmp, k)
        sim = float(np.dot(chroma_ref, rolled))
        if sim > best_sim:
            best_sim = sim
            best_shift = k
    # map 0..11 to signed neighborhood (-6..+5) for readability
    if best_shift > 6:
        best_shift = best_shift - 12
    return int(best_shift), float(best_sim)


def attach_pitch_features(features):
    """
    For each features dict (already containing mono 'signal' and 'sr'),
    attach: 'tuning_cents' and 'chroma_mean'.
    """
    for f in features:
        tc, ch = _compute_tuning_and_chroma(f["signal"], f["sr"])
        f["tuning_cents"] = tc
        f["chroma_mean"] = ch


def pick_reference(features, candidates):
    """
    Given a list of feature dicts and a list of candidate labels,
    return the first matching feature. If none match, return None.
    """
    if not candidates:
        return None
    for cand in candidates:
        for f in features:
            if f["label"] == cand:
                return f
    return None


def pitch_speed_analysis(features, outdir, song_label, ref_label=None):
    """
    Build CSV, summary, and plot comparing pitch/tempo between versions.
    If ref_label is None, uses the first version as reference.
    """
    if not features:
        return

    # Ensure pitch descriptors are attached
    need = any(("tuning_cents" not in f or "chroma_mean" not in f) for f in features)
    if need:
        attach_pitch_features(features)

    # Choose ref version
    ref = None
    if ref_label:
        ref = pick_reference(features, ref_label)
    if ref is None:
        ref = features[0]  # absolute fallback

    ref_label_used = ref["label"]
    ref_chroma = ref["chroma_mean"]
    ref_dur = float(ref["duration_sec"])
    ref_tuning = float(ref["tuning_cents"])

    rows = []
    for f in features:
        shift_semitones, sim = _best_semitone_shift(ref_chroma, f["chroma_mean"])
        speed_from_pitch = float(2.0 ** (shift_semitones / 12.0))
        dur_ratio = ref_dur / max(float(f["duration_sec"]), 1e-9)

        rows.append(
            {
                "song_label": song_label,
                "ref_label": ref_label_used,
                "cmp_label": f["label"],
                "tuning_cents_cmp": float(f["tuning_cents"]),
                "tuning_cents_ref": ref_tuning,
                "delta_tuning_cents": float(f["tuning_cents"] - ref_tuning),
                "semitone_shift_vs_ref": int(shift_semitones),
                "chroma_similarity": sim,
                "speed_factor_from_pitch": speed_from_pitch,
                "duration_ratio_ref_over_cmp": dur_ratio,
            }
        )

    # Salva CSV
    pitch_csv = os.path.join(outdir, f"{song_label}-pitch_report.csv")
    pd.DataFrame(rows).to_csv(pitch_csv, index=False)

    # Salva Summary
    summary_path = os.path.join(outdir, f"{song_label}-pitch_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write(f"Pitch/Speed analysis (reference = {ref_label_used})\n")
        fh.write("=" * 60 + "\n\n")
        for r in rows:
            fh.write(
                f"{r['cmp_label']}: shift={r['semitone_shift_vs_ref']} st ; "
                f"Δtuning={r['delta_tuning_cents']:.1f} cents ; "
                f"speed_from_pitch={r['speed_factor_from_pitch']:.4f} ; "
                f"duration_ratio(ref/cmp)={r['duration_ratio_ref_over_cmp']:.4f}\n"
            )

    # Plot offsets (in cents)
    labels = [r["cmp_label"] for r in rows]
    shifts_cents = [r["delta_tuning_cents"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, shifts_cents, color="orange")
    plt.axhline(0, color="k", linewidth=1)
    plt.ylabel("Δ Tuning (cents vs reference)")
    plt.title(f"Pitch offsets (cents) — {song_label} (ref={ref_label_used})")
    plt.tight_layout()
    out_path, out_name = unique_path(
        os.path.join(outdir, f"{song_label}-pitch_offsets.png")
    )
    out_png = os.path.join(out_path, out_name)
    plt.savefig(out_path, dpi=150)
    features[0]["pitch_plot"] = out_name
    plt.close()

    return pitch_csv, summary_path, out_png


# --------------------------------------------
# Markdown generation  (UPDATED: Stereo Balance section)
# --------------------------------------------


def generate_markdown(
    song_label, song_title, features, outdir, ref_label=None, ref_title=None
):
    md_path = os.path.join(outdir, f"{song_label}.md")
    with open(md_path, "w") as fmd:
        # Front matter per MkDocs Material

        fmd.write("---\n")
        fmd.write(f'title: "{song_title}"\n')
        fmd.write(f"song_label: {song_label}\n")
        if ref_title is not None:
            fmd.write(f"ref_title_version: {ref_title}\n")
        fmd.write("---\n\n")

        if ref_title:
            fmd.write(f"# {song_title} (title taken from {ref_title})\n\n")
        else:
            fmd.write(f"# {song_title}\n\n")

        # Notes (se presenti in results/<song_label>/notes.md)
        notes_path = os.path.join(outdir, "notes.md")
        if os.path.exists(notes_path):
            with open(notes_path) as nf:
                fmd.write("## Notes\n\n")
                fmd.write(nf.read() + "\n\n")

        # Tabella dettagli principali
        df = pd.read_csv(os.path.join(outdir, f"{song_label}-features.csv"))
        fmd.write("## Details\n\n")
        fmd.write(df.to_markdown(index=False))
        fmd.write("\n\n")

        # Plots principali
        fmd.write("## Plots\n")
        fmd.write(f"![Waveforms (Mono)]({features[0]['waveform_mono']})\n\n")
        fmd.write(f"![Waveforms (Left)]({features[0]['waveform_L']})\n\n")
        fmd.write(f"![Waveforms (Right)]({features[0]['waveform_R']})\n\n")
        fmd.write(f"![Radar Plot]({features[0]['radar_plot']})\n\n")
        # Note about overlapping radar plot lines
        dup_file = os.path.join(outdir, f"{song_label}-radar_duplicates.txt")
        if os.path.exists(dup_file):
            fmd.write("\n**Note:** Some versions overlap in the radar plot:\n\n")
            with open(dup_file) as df:
                for line in df:
                    fmd.write(f"- {line.strip()}\n")
            fmd.write("\n")

        fmd.write(f"![MFCC Similarity]({features[0]['similarity_plot']})\n\n")

        # Stereo Balance
        fmd.write("## Stereo Balance\n\n")
        for f in features:
            fmd.write(f"### {f['label']}\n\n")
            fmd.write(f"![STFT Spectrogram (Left)]({f['spectrogram_L']})\n\n")
            fmd.write(f"![STFT Spectrogram (Right)]({f['spectrogram_R']})\n\n")
            fmd.write(f"![Mel Spectrogram (Left)]({f['melspec_L']})\n\n")
            fmd.write(f"![Mel Spectrogram (Right)]({f['melspec_R']})\n\n")
            fmd.write(f"![Stereo Balance Bars]({f['balance_plot']})\n\n")

        # Spectrograms (Mono)
        fmd.write("## Spectrograms (Mono)\n\n")
        for f in features:
            fmd.write(f"### {f['label']}\n\n")
            fmd.write(f"![STFT Spectrogram (Mono)]({f['spectrogram_Mono']})\n\n")
            fmd.write(f"![Mel Spectrogram (Mono)]({f['melspec_Mono']})\n\n")

        # Pitch & Speed Analysis
        pitch_csv = os.path.join(outdir, f"{song_label}-pitch_report.csv")
        pitch_summary = os.path.join(outdir, f"{song_label}-pitch_summary.txt")
        pitch_plot = os.path.join(outdir, f"{song_label}-pitch_offsets.png")
        if os.path.exists(pitch_csv):
            fmd.write("## Pitch & Speed Analysis (cents)\n\n")
            ref_used = None
            if ref_label:
                ref_used = ref_label[0] if isinstance(ref_label, list) else ref_label
            else:
                ref_used = features[0]["label"]
            fmd.write(f"Reference version: **{ref_used}**\n\n")

            df_pitch = pd.read_csv(pitch_csv)
            fmd.write(df_pitch.to_markdown(index=False))
            fmd.write("\n\n")

            if os.path.exists(pitch_plot):
                fmd.write(f"![Pitch Offsets]({features[0]['pitch_plot']})\n\n")

            if os.path.exists(pitch_summary):
                with open(pitch_summary, "r") as ps:
                    fmd.write("```\n")
                    fmd.write(ps.read())
                    fmd.write("```\n\n")
                    # Links to CSV and TXT
                    # fmd.write(f"- [Pitch report CSV]({os.path.basename(pitch_csv)})\n")
                    # fmd.write(
                    # f"- [Pitch summary TXT]({os.path.basename(pitch_summary)})\n\n"
                    # )

    return md_path


# --------------------------------------------
# Smoke check  (UPDATED for new outputs)
# --------------------------------------------
def _smoke_check_outputs(outdir, features):
    song_label = os.path.basename(outdir)

    expected = [
        os.path.join(outdir, f"{song_label}-waveforms_Mono.png"),
        os.path.join(outdir, f"{song_label}-waveforms_L.png"),
        os.path.join(outdir, f"{song_label}-waveforms_R.png"),
        os.path.join(outdir, f"{song_label}-radar_plot.png"),
        os.path.join(outdir, f"{song_label}-similarity_matrix.png"),
        os.path.join(outdir, f"{song_label}-features.csv"),
        os.path.join(outdir, f"{song_label}-features_normalised.csv"),
    ]

    for f in features:
        expected += [
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_Mono.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_Mono.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_L.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_L.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_spectrogram_R.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_melspec_R.png"),
            os.path.join(outdir, f"{song_label}-{f['label']}_balance.png"),
        ]

    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing outputs:")
        for m in missing:
            print(f"  - {m}")


# --------------------------------------------
# Song analysis
# --------------------------------------------
def analyse_song(
    song_label, song_title, files, outroot="results", dry_run=False, keep=False
):
    outdir = os.path.join(outroot, song_label)
    os.makedirs(outdir, exist_ok=True)

    features = []
    for item in files:
        md5 = compute_md5(item["file"])
        if md5 in EXISTING_MD5S:
            print(f"[SKIP] {item['file']} already analysed (md5={md5})")
            continue

        if dry_run:
            print(
                f"[DRY-RUN] Would analyse: {item['file']} (label={item['label']}, md5={md5})"
            )
            continue

        wav = convert_to_wav(item["file"])
        f = extract_features(
            wav, item["label"], item["file"], item["disc"], item["track"], md5
        )
        features.append(f)

        if not keep:
            os.remove(wav)

    if dry_run or not features:
        return

    # --- CSV (UPDATED: include stereo metrics) ---
    df = pd.DataFrame(
        [
            {
                "label": f["label"],
                "orig_file": os.path.basename(f["orig_file"]),
                "md5": f["md5"],
                "disc": f["disc"],
                "track": f["track"],
                "duration_sec": f["duration_sec"],
                "duration_fmt": f["duration_fmt"],
                "loudness": f["loudness"],
                "loudness_left": f["loudness_left"],
                "loudness_right": f["loudness_right"],
                "loudness_balance": f["loudness_balance"],
                "rms": f["rms"],
                "rms_left": f["rms_left"],
                "rms_right": f["rms_right"],
                "rms_balance": f["rms_balance"],
                "lr_corr": f["lr_corr"],
                "spectral_centroid": f["spectral_centroid"],
            }
            for f in features
        ]
    )
    df.to_csv(os.path.join(outdir, f"{song_label}-features.csv"), index=False)

    # Normalised (unchanged columns + keep only the numeric ones)
    df_norm = df.copy()
    for col in [
        "duration_sec",
        "loudness",
        "loudness_left",
        "loudness_right",
        "rms",
        "rms_left",
        "rms_right",
        "spectral_centroid",
    ]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    df_norm.to_csv(
        os.path.join(outdir, f"{song_label}-features_normalised.csv"), index=False
    )

    # Plots (UPDATED)
    plot_waveforms(features, outdir, song_label)
    plot_spectrograms(features, outdir, song_title, song_label)
    plot_mel_spectrograms(features, outdir, song_title, song_label)
    plot_similarity_matrix(features, outdir, song_label)
    plot_radar_chart(features, outdir, song_label)
    plot_lr_balance_bars(features, outdir, song_label)

    pitch_csv, pitch_summary, pitch_plot = pitch_speed_analysis(
        features, outdir, song_label, ref_label=args.ref_label
    )

    # Free audio data to save memory
    for f in features:
        if "signal" in f:
            del f["signal"]

    # Determine which version provides the reference title
    ref_title_used = None
    resolved_song_title = song_title  # fallback: the title from the first CSV row

    if args.ref_title:
        # Build lookup: version_label -> song_title from CSV input
        version_to_title = {
            v["label"]: v["song_title"] for v in files if "song_title" in v
        }

        for candidate in args.ref_title:
            if candidate in version_to_title and version_to_title[candidate].strip():
                ref_title_used = candidate
                resolved_song_title = version_to_title[candidate]
                break

    # Markdown (UPDATED with ref_title fallback)

    generate_markdown(song_label, resolved_song_title, features, outdir, ref_title_used)

    # JSONL append (unchanged fields)
    with open("metadata.jsonl", "a") as fjson:
        for f in features:
            entry = {
                "song_label": song_label,
                "filename": os.path.basename(f["orig_file"]),
                "version_label": f["label"],
                "version_disc": f["disc"],
                "version_track": f["track"],
                "duration": f["duration_fmt"],
                "duration_sec": f["duration_sec"],
                "loudness": f"{f['loudness']:.2f} LUFS",
                "rms": f["rms"],
                "spectral_centroid": f"{f['spectral_centroid']:.2f} Hz",
                "md5": f["md5"],
                "spectrogram": f"{song_label}-{f['label']}_spectrogram_Mono.png",
                "melspec": f"{song_label}-{f['label']}_melspec_Mono.png",
            }
            fjson.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Smoke check updated
    _smoke_check_outputs(outdir, features)

    print(
        f"[OK] Analysis completed for '{song_label}' ({song_title}). Results in: {outdir}"
    )


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="List operations without executing"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep intermediate wav files"
    )
    parser.add_argument(
        "--song", type=str, help="Process only the given song_label", default=None
    )

    parser.add_argument(
        "--ref-label",
        nargs="+",
        default=None,
        help="One or more version labels to use as reference for pitch/speed analysis. If the first is not found, fall back to the next.",
    )

    parser.add_argument(
        "--ref-title",
        nargs="+",
        default=None,
        help="One or more version labels to use as reference for extracting the song title. Fallback order as listed.",
    )

    args = parser.parse_args()

    EXISTING_MD5S = load_existing_md5s()

    input_csv = "songs.csv"
    songs = {}

    with open(input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            version_label = derive_version_label(row["filepath"])
            songs.setdefault(
                row["song_label"], {"title": row["song_title"], "versions": []}
            )
            songs[row["song_label"]]["versions"].append(
                {
                    "file": os.path.expanduser(row["filepath"]),
                    "label": version_label,
                    "disc": row["version_disc"],
                    "track": row["version_track"],
                }
            )

    for song_label, data in songs.items():
        if args.song and song_label != args.song:
            continue
        analyse_song(
            song_label,
            data["title"],
            data["versions"],
            dry_run=args.dry_run,
            keep=args.keep,
        )
