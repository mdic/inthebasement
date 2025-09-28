import os
import librosa
import librosa.display
import numpy as np
import pyloudnorm as pyln
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

RESULTS_DIR = "cd_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# -------------------------------------------------
# Detect CD mount point from /proc/mounts
# -------------------------------------------------
def get_cd_mountpoint(device="/dev/sr0"):
    with open("/proc/mounts") as f:
        for line in f:
            parts = line.split()
            if parts[0] == device:
                return parts[1]
    raise RuntimeError(f"Device {device} is not mounted")


# -------------------------------------------------
# Extract audio features for one WAV track
# -------------------------------------------------
def extract_features(wavfile):
    y, sr = librosa.load(wavfile, sr=44100, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return {
        "file": os.path.basename(wavfile),
        "duration": duration,
        "loudness": loudness,
        "rms": rms,
        "spectral_centroid": spectral_centroid,
        "mfcc": mfcc_mean,
        "signal": y,
        "sr": sr,
    }


# -------------------------------------------------
# Analyse all WAV files on the mounted CD
# -------------------------------------------------
def analyse_cd(device="/dev/sr0"):
    mountpoint = get_cd_mountpoint(device)
    print(f">> CD detected at mount point: {mountpoint}")

    wavs = sorted(
        [
            os.path.join(mountpoint, f)
            for f in os.listdir(mountpoint)
            if f.lower().endswith(".wav")
        ]
    )
    if not wavs:
        print(f"[ERROR] No WAV files found in {mountpoint}")
        return

    features = [extract_features(w) for w in wavs]

    rows = []
    for i, f in enumerate(features):
        title = os.path.splitext(f["file"])[0]
        rows.append(
            {
                "track": i + 1,
                "title": title,
                "file": f["file"],
                "duration": f["duration"],
                "loudness": f["loudness"],
                "rms": f["rms"],
                "spectral_centroid": f["spectral_centroid"],
            }
        )

        # Save spectrogram plot
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
        plt.title(f"{title} (Track {i + 1})")
        plt.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(
            os.path.join(RESULTS_DIR, f"track{i + 1:02d}_{safe_title}_spectrogram.png"),
            dpi=150,
        )
        plt.close()

    # Save summary CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "cd_baseline.csv"), index=False)

    print(
        f"[OK] Analysis complete. {len(wavs)} tracks processed. Results saved in {RESULTS_DIR}/"
    )


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    analyse_cd("/dev/sr0")
