import os
import csv
import subprocess
import librosa
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import hashlib

# -----------------------------------------------------
# Fixed color mapping for version identifiers (label)
# Extend this dictionary as new labels appear
# -----------------------------------------------------
LABEL_COLORS = {
    "HHBST": "b",
    "TBT": "r",
    "CBST": "g",
    "ALT3": "m",
}


# -----------------------------------------------------
# Utility functions
# -----------------------------------------------------
def convert_to_wav(infile, outdir="converted", force=False):
    """Convert audio file to standard WAV (44100 Hz, 16 bit, stereo)."""
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


def extract_features(wavfile, label, orig_file):
    """Extract audio features from WAV file."""
    y, sr = librosa.load(wavfile, sr=44100, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    return {
        "label": label,
        "file": os.path.basename(wavfile),
        "orig_file": orig_file,
        "duration": duration,
        "loudness": loudness,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "mfcc": mfcc_mean,
        "signal": y,
        "sr": sr,
    }


def dataframe_to_markdown_table(df):
    """Convert a pandas DataFrame into a Markdown table string."""
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in df.values.tolist()
    ]
    return "\n".join([header, separator] + rows)


def format_duration(seconds: float) -> str:
    """Format duration in MM:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_metadata_jsonl(filepath="metadata.jsonl"):
    """Load existing metadata entries from JSONL file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_metadata_jsonl(entries, filepath="metadata.jsonl"):
    """Save metadata entries back to JSONL file."""
    with open(filepath, "w") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def update_metadata(entry, filepath="metadata.jsonl"):
    """Add or update a metadata entry, avoiding duplicates by MD5."""
    existing = load_metadata_jsonl(filepath)
    duplicate = next((e for e in existing if e["md5"] == entry["md5"]), None)

    if duplicate:
        ans = input(
            f"⚠️ File {entry['filename']} (MD5 {entry['md5']}) already in metadata.jsonl. Overwrite? [y/n] "
        )
        if ans.lower() != "y":
            print(f"Skipped {entry['filename']}")
            return
        # Remove old entry with same md5
        existing = [e for e in existing if e["md5"] != entry["md5"]]

    existing.append(entry)
    save_metadata_jsonl(existing, filepath)


# -----------------------------------------------------
# Plotting functions
# -----------------------------------------------------


def plot_waveforms(features, outdir, song_title):
    """
    Plot comparative waveforms using librosa.display.waveshow with per-label colors.
    Colors come from LABEL_COLORS if available, otherwise from a stable fallback cmap.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
    from matplotlib.cm import get_cmap

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = get_cmap("tab10")

    for idx, f in enumerate(features):
        lbl = str(f["label"]).strip()
        color = LABEL_COLORS.get(lbl, cmap(idx % cmap.N))

        librosa.display.waveshow(
            y=f["signal"],
            sr=f["sr"],
            label=lbl,
            color=color,  # force explicit color
            alpha=0.8,
            ax=ax,
        )

    ax.legend(title="Version", frameon=False)
    ax.set_title(f"Comparative Waveforms — {song_title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "waveforms.png"), dpi=150)
    plt.close(fig)


def plot_similarity_matrix(features, outdir, song_title):
    """Plot similarity matrix of MFCC means."""
    mfcc_matrix = np.array([f["mfcc"] for f in features])
    sim = cosine_similarity(mfcc_matrix)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(
        range(len(features)), [f["label"] for f in features], rotation=45, ha="right"
    )
    plt.yticks(range(len(features)), [f["label"] for f in features])
    plt.title(f"MFCC Similarity Matrix — {song_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "similarity_matrix.png"), dpi=150)
    plt.close()

    sim_df = pd.DataFrame(
        sim,
        index=[f["label"] for f in features],
        columns=[f["label"] for f in features],
    )
    sim_df.to_csv(os.path.join(outdir, "similarity_matrix.csv"))


# -----------------------------------------------------
# Song analysis
# -----------------------------------------------------
def analyse_song(song_label, files, song_title="Unknown", outroot="results"):
    outdir = os.path.join(outroot, song_label)
    os.makedirs(outdir, exist_ok=True)

    wavs = [convert_to_wav(item["file"]) for item in files]
    features = [
        extract_features(w, item["label"], item["file"]) for item, w in zip(files, wavs)
    ]

    # Save features CSV
    df = pd.DataFrame(
        [
            {
                "label": f["label"],
                "orig_file": f["orig_file"],
                "duration": f["duration"],
                "loudness": f["loudness"],
                "rms": f["rms"],
                "spectral_centroid": f["spectral_centroid"],
            }
            for f in features
        ]
    )
    df.to_csv(os.path.join(outdir, "features.csv"), index=False)

    # Normalised features
    df_norm = df.copy()
    for col in ["duration", "loudness", "rms", "spectral_centroid"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df_norm[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.0
    df_norm.to_csv(os.path.join(outdir, "features_normalised.csv"), index=False)

    # Markdown report with MD5 + duration
    md_path = os.path.join(outdir, f"{song_label}.md")
    with open(md_path, "w") as fmd:
        fmd.write(f"# {song_title}\n\n")
        fmd.write("## Versions\n")
        for f in features:
            md5 = compute_md5(f["orig_file"])
            fmd.write(
                f"- **{f['label']}** ({f['orig_file']}) — Duration {format_duration(f['duration'])}, MD5 `{md5}`\n"
            )
        fmd.write("\n## Features (raw)\n")
        fmd.write(df.to_markdown() + "\n\n")
        fmd.write("## Features (normalised)\n")
        fmd.write(df_norm.to_markdown() + "\n\n")

    # Plots
    plot_waveforms(features, outdir, song_title)
    plot_similarity_matrix(features, outdir, song_title)

    # Update metadata.jsonl
    for f in features:
        md5 = compute_md5(f["orig_file"])
        entry = {
            "song_label": song_label,
            "song_title": song_title,
            "filename": f["orig_file"],
            "version_label": f["label"],
            "duration": format_duration(f["duration"]),
            "duration_sec": round(f["duration"], 3),
            "loudness": round(f["loudness"], 2),
            "rms": round(float(f["rms"]), 6),
            "spectral_centroid": round(f["spectral_centroid"], 2),
            "md5": md5,
        }
        update_metadata(entry)

    print(
        f"[OK] Analysis completed for '{song_label}' ({song_title}). Results in: {outdir}"
    )


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    input_csv = "songs.csv"
    songs = {}
    with open(input_csv, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        for row in reader:
            songs.setdefault(
                row["song"], {"title": row.get("title", "Unknown"), "files": []}
            )
            songs[row["song"]]["files"].append(
                {"file": row["file"], "label": row["label"]}
            )

    for song_label, data in songs.items():
        analyse_song(song_label, data["files"], song_title=data["title"])
