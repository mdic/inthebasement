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
    "The Genuine Basement Tapes": "TGBT",
    "Home Tape": "HHST",
    "The Basement Tapes - Captain Acid 2020 restoration": "CAR",
    "Basement Tapes Anthology [FLAC]": "TCB",
    "Mixin Up The Medicine": "MUTM",
    "A Tree With Roots": "ATWR",
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


# --------------------------------------------
# Feature extraction
# --------------------------------------------
def extract_features(wavfile, label, orig_file, version_disc, version_track, md5):
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
        "md5": md5,
    }


# --------------------------------------------
# Plotting
# --------------------------------------------
def plot_waveforms(features, outdir):
    plt.figure(figsize=(10, 6))
    for f in features:
        color = LABEL_COLORS.get(f["label"], None)
        plt.plot(
            np.linspace(0, f["duration_sec"], num=len(f["signal"])),
            f["signal"],
            alpha=0.6,
            label=f["label"],
            color=color,
        )
    plt.legend()
    plt.title("Comparative Waveforms")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "waveforms.png"), dpi=150)
    plt.close()


def plot_spectrograms(features, outdir, song_title):
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
        fname = f"{f['label']}_spectrogram.png"
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()


def plot_mel_spectrograms(features, outdir, song_title):
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
        fname = f"{f['label']}_melspec.png"
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()


def plot_similarity_matrix(features, outdir):
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
    plt.savefig(os.path.join(outdir, "similarity_matrix.png"), dpi=150)
    plt.close()

    sim_df = pd.DataFrame(
        sim,
        index=[f["label"] for f in features],
        columns=[f["label"] for f in features],
    )
    sim_df.to_csv(os.path.join(outdir, "similarity_matrix.csv"))


def plot_radar_chart(features, outdir):
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
    plt.title("Radar Plot of Main Features")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "radar_plot.png"), dpi=150)
    plt.close()


# --------------------------------------------
# Markdown generation
# --------------------------------------------
def generate_markdown(song_label, song_title, features, outdir):
    md_path = os.path.join(outdir, f"{song_label}.md")
    with open(md_path, "w") as fmd:
        fmd.write("---\n")
        fmd.write(f'title: "{song_title}"\n')
        fmd.write(f"song_label: {song_label}\n")
        fmd.write("---\n\n")
        fmd.write(f"# {song_title}\n\n")

        notes_path = os.path.join(outdir, "notes.md")
        if os.path.exists(notes_path):
            with open(notes_path) as nf:
                fmd.write("## Notes\n\n")
                fmd.write(nf.read() + "\n\n")

        df = pd.read_csv(os.path.join(outdir, "features.csv"))
        df_norm = pd.read_csv(os.path.join(outdir, "features_normalised.csv"))

        fmd.write("## Details\n\n")
        fmd.write(df.to_markdown(index=False))
        fmd.write("\n\n")

        fmd.write("## Details (normalised)\n\n")
        fmd.write(df_norm.to_markdown(index=False))
        fmd.write("\n\n")

        fmd.write("## Plots\n")
        fmd.write("![Waveforms](waveforms.png)\n")
        fmd.write("![Radar Plot](radar_plot.png)\n")
        fmd.write("![MFCC Similarity](similarity_matrix.png)\n\n")

        fmd.write("## Spectrograms\n\n")
        for f in features:
            fmd.write(f"### {f['label']}\n\n")
            fmd.write(f"![STFT Spectrogram]({f['label']}_spectrogram.png)\n\n")
            fmd.write(f"![Mel Spectrogram]({f['label']}_melspec.png)\n\n")

    return md_path


# Smoke check
# --------------------------------------------
def _smoke_check_outputs(outdir, features):
    expected = [
        os.path.join(outdir, "waveforms.png"),
        os.path.join(outdir, "radar_plot.png"),
        os.path.join(outdir, "similarity_matrix.png"),
        os.path.join(outdir, "features.csv"),
        os.path.join(outdir, "features_normalised.csv"),
    ]
    for f in features:
        expected += [
            os.path.join(outdir, f"{f['label']}_spectrogram.png"),
            os.path.join(outdir, f"{f['label']}_melspec.png"),
        ]
    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing outputs:")
        for m in missing:
            print(f"  - {m}")


# --------------------------------------------
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
                "rms": f["rms"],
                "spectral_centroid": f["spectral_centroid"],
            }
            for f in features
        ]
    )
    df.to_csv(os.path.join(outdir, "features.csv"), index=False)

    df_norm = df.copy()
    for col in ["duration_sec", "loudness", "rms", "spectral_centroid"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    df_norm.to_csv(os.path.join(outdir, "features_normalised.csv"), index=False)

    plot_waveforms(features, outdir)
    plot_spectrograms(features, outdir, song_title)
    plot_mel_spectrograms(features, outdir, song_title)
    plot_similarity_matrix(features, outdir)
    plot_radar_chart(features, outdir)

    generate_markdown(song_label, song_title, features, outdir)

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
                "spectrogram": f"{f['label']}_spectrogram.png",
                "melspec": f"{f['label']}_melspec.png",
            }
            fjson.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
        analyse_song(
            song_label,
            data["title"],
            data["versions"],
            dry_run=args.dry_run,
            keep=args.keep,
        )
