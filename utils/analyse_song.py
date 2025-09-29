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
def plot_waveforms(features, outdir, song_title):
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
        plt.savefig(os.path.join(outdir, f"{f['label']}_spectrogram.png"), dpi=150)
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
        plt.savefig(os.path.join(outdir, f"{f['label']}_melspec.png"), dpi=150)
        plt.close()


def plot_similarity_matrix(features, outdir, song_title):
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


def plot_radar_chart(features, outdir, song_title):
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
def generate_markdown(song_label, song_title, features, outdir):
    md_path = os.path.join(outdir, f"{song_label}.md")
    with open(md_path, "w") as fmd:
        fmd.write(f"---\n")
        fmd.write(f'title: "{song_title}"\n')
        fmd.write(f"song_label: {song_label}\n")
        fmd.write(f"---\n\n")

        fmd.write(f"# {song_title}\n\n")
        fmd.write(f"**Song Label:** {song_label}\n\n")

        fmd.write("## Versions\n")
        for f in features:
            fmd.write(
                f"- **{f['label']}** — Disc {f['disc']:02d}, Track {f['track']:02d} "
                f"— Duration {f['duration_fmt']} — Loudness {f['loudness']:.2f} LUFS "
                f"— RMS {f['rms']:.6f} — Spectral Centroid {f['spectral_centroid']:.2f} Hz\n"
            )
        fmd.write("\n")

        fmd.write("## Plots\n")
        fmd.write("![Waveforms](waveforms.png)\n")
        fmd.write("![Radar Plot](radar_plot.png)\n")
        fmd.write("![MFCC Similarity](similarity_matrix.png)\n\n")

        # CSVs rendered as Markdown tables
        for csv_name in ["features.csv", "features_normalised.csv"]:
            csv_path = os.path.join(outdir, csv_name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                fmd.write(f"## {csv_name}\n")
                fmd.write(df.to_markdown(index=False) + "\n\n")
                fmd.write(f"- [{csv_name}]({csv_name})\n\n")

    return md_path


# --------------------------------------------
# Song analysis
# --------------------------------------------
def analyse_song(song_label, song_title, files, outroot="results"):
    outdir = os.path.join(outroot, song_label)
    os.makedirs(outdir, exist_ok=True)

    wavs = [convert_to_wav(item["file"]) for item in files]
    features = [
        extract_features(w, item["label"], item["file"], item["disc"], item["track"])
        for item, w in zip(files, wavs)
    ]

    df = pd.DataFrame(
        [
            {
                "label": f["label"],
                "orig_file": f["orig_file"],
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

    plot_waveforms(features, outdir, song_title)
    plot_spectrograms(features, outdir, song_title)
    plot_mel_spectrograms(features, outdir, song_title)
    plot_similarity_matrix(features, outdir, song_title)
    plot_radar_chart(features, outdir, song_title)

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
                "md5": compute_md5(f["orig_file"]),
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
    input_csv = "songs.csv"
    songs = {}

    with open(input_csv, newline="") as csvfile:
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
        analyse_song(song_label, data["title"], data["versions"])
