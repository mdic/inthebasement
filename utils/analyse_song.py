import os
import csv
import subprocess
import argparse
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import hashlib

# --------------------------------------------
# Colour mapping for version labels
# --------------------------------------------
LABEL_COLORS = {
    "HHST": "skyblue",
    "TBT": "orange",
    "CBST": "green",
    "FTR": "purple",
}


# --------------------------------------------
# Utilities
# --------------------------------------------
def convert_to_wav(infile, outdir="converted", force=False, dry_run=False):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(infile))[0]
    outfile = os.path.join(outdir, base + ".wav")

    if force or not os.path.exists(outfile):
        if dry_run:
            print(f"[DRY-RUN] Would run ffmpeg → {outfile}")
        else:
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
    else:
        if dry_run:
            print(f"[DRY-RUN] WAV already exists, would reuse → {outfile}")

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


def load_existing_md5_set(metadata_path="metadata.jsonl"):
    """Return a set of md5 already present in metadata.jsonl (if it exists)."""
    md5s = set()
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "md5" in obj:
                        md5s.add(obj["md5"])
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue
    return md5s


# --------------------------------------------
# Feature extraction
# --------------------------------------------
def extract_features(wavfile, label, orig_file, version_disc, version_track):
    y, sr = librosa.load(wavfile, sr=44100, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return {
        "label": label,
        "file": os.path.basename(wavfile),
        "orig_file": orig_file,
        "disc": int(version_disc),
        "track": int(version_track),
        "duration_sec": duration,
        "duration_fmt": format_duration(duration),
        "loudness": loudness,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "mfcc": mfcc_mean,
        "signal": y,
        "sr": sr,
    }


# --------------------------------------------
# Plotting
# --------------------------------------------
def plot_waveforms(features, outdir, song_title, dry_run=False):
    if dry_run:
        print(f"[DRY-RUN] Would write {os.path.join(outdir, 'waveforms.png')}")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for f in features:
        color = LABEL_COLORS.get(f["label"], None)
        y = np.array(f["signal"])
        sr = f["sr"]
        step = max(1, len(y) // 10000)
        t = np.arange(0, len(y), step) / sr
        y_ds = y[::step]
        ax.plot(t, y_ds, label=f["label"], color=color, alpha=0.8, linewidth=1.0)

    plt.legend()
    plt.title(f"Comparative Waveforms - {song_title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "waveforms.png"), dpi=150)
    plt.close()


def plot_spectrograms(features, outdir, song_title, dry_run=False):
    if dry_run:
        for f in features:
            print(
                f"[DRY-RUN] Would write {os.path.join(outdir, f'{f["label"]}_spectrogram.png')}"
            )
        return

    for f in features:
        D = np.abs(librosa.stft(f["signal"]))
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            librosa.amplitude_to_db(D, ref=np.max),
            sr=f["sr"],
            x_axis="time",
            y_axis="log",
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"STFT Spectrogram - {f['label']} {song_title}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{f['label']}_spectrogram.png"), dpi=150)
        plt.close()


def plot_mel_spectrograms(features, outdir, song_title, dry_run=False):
    if dry_run:
        for f in features:
            print(
                f"[DRY-RUN] Would write {os.path.join(outdir, f'{f["label"]}_melspec.png')}"
            )
        return

    for f in features:
        S = librosa.feature.melspectrogram(y=f["signal"], sr=f["sr"], n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8, 5))
        librosa.display.specshow(
            S_dB, sr=f["sr"], x_axis="time", y_axis="mel", cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Mel Spectrogram - {f['label']} {song_title}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{f['label']}_melspec.png"), dpi=150)
        plt.close()


def plot_similarity_matrix(features, outdir, song_title, dry_run=False):
    if dry_run:
        print(
            f"[DRY-RUN] Would write {os.path.join(outdir, 'similarity_matrix.png')} and CSV"
        )
        return

    mfcc_matrix = np.array([f["mfcc"] for f in features])
    sim = cosine_similarity(mfcc_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(
        range(len(features)), [f["label"] for f in features], rotation=45, ha="right"
    )
    plt.yticks(range(len(features)), [f["label"] for f in features])
    plt.title(f"MFCC Similarity Matrix - {song_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "similarity_matrix.png"), dpi=150)
    plt.close()

    sim_df = pd.DataFrame(
        sim,
        index=[f["label"] for f in features],
        columns=[f["label"] for f in features],
    )
    sim_df.to_csv(os.path.join(outdir, "similarity_matrix.csv"))


def plot_radar_chart(features, outdir, song_title, dry_run=False):
    if dry_run:
        print(f"[DRY-RUN] Would write {os.path.join(outdir, 'radar_plot.png')}")
        return

    metrics = ["duration_sec", "loudness", "rms", "spectral_centroid"]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    eps = 1e-9
    mins = {m: min(float(f[m]) for f in features) for m in metrics}
    maxs = {m: max(float(f[m]) for f in features) for m in metrics}

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for f in features:
        vals = []
        for m in metrics:
            rng = maxs[m] - mins[m]
            v = (float(f[m]) - mins[m]) / (rng + eps)
            vals.append(v)
        vals = np.array(vals)
        vals = np.concatenate([vals, vals[:1]])

        color = LABEL_COLORS.get(f["label"], None)
        ax.plot(angles, vals, label=f["label"], color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title(f"Radar Plot of Main Features - {song_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "radar_plot.png"), dpi=150)
    plt.close()


# --------------------------------------------
# Markdown generation
# --------------------------------------------
def generate_markdown(song_label, song_title, outdir, include_versions_block=False):
    """
    Create/overwrite the song markdown page.

    - Inject Notes (if results/<song_label>/notes.md exists) under main heading.
    - Replace 'Versions' with 'Details' and 'Details (normalised)' sections
      (tables rendered from CSVs).
    """
    md_path = os.path.join(outdir, f"{song_label}.md")
    with open(md_path, "w", encoding="utf-8") as fmd:
        # Front matter
        fmd.write(f"---\n")
        fmd.write(f'title: "{song_title}"\n')
        fmd.write(f"song_label: {song_label}\n")
        fmd.write(f"---\n\n")

        # Heading
        fmd.write(f"# {song_title}\n\n")

        # Optional Notes
        notes_path = os.path.join(outdir, "notes.md")
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as fn:
                notes_content = fn.read().strip()
            if notes_content:
                fmd.write("## Notes\n\n")
                fmd.write(notes_content + "\n\n")

        # Tables from CSVs
        # Details
        csv_path = os.path.join(outdir, "features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            fmd.write("## Details\n")
            fmd.write(df.to_markdown(index=False) + "\n\n")
            fmd.write(f"- [features.csv](features.csv)\n\n")

        # Details (normalised)
        csvn_path = os.path.join(outdir, "features_normalised.csv")
        if os.path.exists(csvn_path):
            dfn = pd.read_csv(csvn_path)
            fmd.write("## Details (normalised)\n")
            fmd.write(dfn.to_markdown(index=False) + "\n\n")
            fmd.write(f"- [features_normalised.csv](features_normalised.csv)\n\n")
        # Plots
        fmd.write("## Plots\n")
        fmd.write("![Waveforms](waveforms.png)\n")
        fmd.write("![Radar Plot](radar_plot.png)\n")
        fmd.write("![MFCC Similarity](similarity_matrix.png)\n\n")

    return md_path


# --------------------------------------------
# Song analysis
# --------------------------------------------
def analyse_song(
    song_label,
    song_title,
    files,
    outroot="results",
    dry_run=False,
    metadata_path="metadata.jsonl",
):
    outdir = os.path.join(outroot, song_label)
    os.makedirs(outdir, exist_ok=True)

    # Load existing MD5 set once
    existing_md5 = load_existing_md5_set(metadata_path)

    # Prepare processing list (skip already-seen md5)
    to_process = []
    for item in files:
        orig = os.path.expanduser(item["file"])
        md5 = compute_md5(orig)
        if md5 in existing_md5:
            print(f"[SKIP] {orig} (md5={md5}) already in {metadata_path}")
            continue
        to_process.append({**item, "orig": orig, "md5": md5})

    if not to_process:
        print(
            f"[INFO] No new versions to analyse for '{song_label}'. Writing/refreshing markdown only."
        )
        # Still (re)generate the page to pick up possible notes changes
        generate_markdown(song_label, song_title, outdir)
        return

    if dry_run:
        print(f"[DRY-RUN] Would analyse song '{song_label}' — {song_title}")
        for it in to_process:
            print(
                f"  - Would convert/analyse: {it['orig']} (label={it['label']}, disc={it['disc']}, track={it['track']}, md5={it['md5']})"
            )
        print(
            f"[DRY-RUN] Would write plots (waveforms/spectrograms/melspec/similarity/radar)"
        )
        print(
            f"[DRY-RUN] Would write features.csv, features_normalised.csv, {metadata_path}, and {song_label}.md"
        )
        return

    # Convert → Extract features → Delete WAV
    features = []
    wav_paths = []
    try:
        for it in to_process:
            wav = convert_to_wav(it["orig"])
            wav_paths.append(wav)
            fdict = extract_features(
                wav, it["label"], it["orig"], it["disc"], it["track"]
            )
            fdict["md5"] = it["md5"]  # attach md5 for CSV and metadata
            features.append(fdict)
    finally:
        # Delete all intermediate WAVs
        for wav in wav_paths:
            try:
                if os.path.exists(wav):
                    os.remove(wav)
                    print(f"[CLEAN] Removed intermediate WAV: {wav}")
            except Exception as e:
                print(f"[WARN] Could not remove WAV {wav}: {e}")

    # Save CSVs (features with md5 as 3rd column)
    df = pd.DataFrame(
        [
            {
                "label": f["label"],
                "orig_file": f["orig_file"],
                "md5": f["md5"],  # third column
                "disc": f["disc"],
                "track": f["track"],
                "duration_sec": f["duration_sec"],
                "duration_fmt": f["duration_fmt"],
                "loudness": f["loudness"],
                "rms": f["rms"],
                "spectral_centroid": f["spectral_centroid"],
            }
            for f in features
        ]
    )
    # Ensure column order explicitly
    df = df[
        [
            "label",
            "orig_file",
            "md5",
            "disc",
            "track",
            "duration_sec",
            "duration_fmt",
            "loudness",
            "rms",
            "spectral_centroid",
        ]
    ]
    df.to_csv(os.path.join(outdir, "features.csv"), index=False)

    df_norm = df.copy()
    for col in ["duration_sec", "loudness", "rms", "spectral_centroid"]:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max > col_min:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    df_norm.to_csv(os.path.join(outdir, "features_normalised.csv"), index=False)

    # Plots
    plot_waveforms(features, outdir, song_title)
    plot_spectrograms(features, outdir, song_title)
    plot_mel_spectrograms(features, outdir, song_title)
    plot_similarity_matrix(features, outdir, song_title)
    plot_radar_chart(features, outdir, song_title)

    # Markdown
    generate_markdown(song_label, song_title, outdir)

    # JSONL append
    with open(metadata_path, "a", encoding="utf-8") as fjson:
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
                "spectrogram": f"{f['label']}_spectrogram.png",
                "melspec": f"{f['label']}_melspec.png",
            }
            fjson.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"[OK] Analysis completed for '{song_label}' ({song_title}). Results in: {outdir}"
    )


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse multiple versions of a song and generate comparative outputs."
    )
    parser.add_argument(
        "--input-csv",
        default="songs.csv",
        help="Input CSV with columns: song_label,filepath,song_title,version_label,version_disc,version_track",
    )
    parser.add_argument(
        "--outroot", default="results", help="Root folder for per-song outputs"
    )
    parser.add_argument(
        "--metadata",
        default="metadata.jsonl",
        help="Path to the cumulative metadata .jsonl file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List operations without executing them"
    )
    args = parser.parse_args()

    songs = {}
    with open(args.input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            songs.setdefault(
                row["song_label"], {"title": row["song_title"], "versions": []}
            )
            songs[row["song_label"]]["versions"].append(
                {
                    "file": os.path.expanduser(row["filepath"]),
                    "label": row["version_label"],
                    "disc": row["version_disc"],
                    "track": row["version_track"],
                }
            )

    for song_label, data in songs.items():
        analyse_song(
            song_label,
            data["title"],
            data["versions"],
            outroot=args.outroot,
            dry_run=args.dry_run,
            metadata_path=args.metadata,
        )
