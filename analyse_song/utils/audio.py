from __future__ import annotations
import os
import subprocess
from typing import List, Tuple
import numpy as np
import librosa
import pyloudnorm as pyln

from .io_utils import format_duration


FFMPEG_ARGS = ["-ar", "44100", "-ac", "2", "-sample_fmt", "s16"]


def convert_to_wav(infile: str, outdir: str = "converted", force: bool = False) -> str:
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(infile))[0]
    outfile = os.path.join(outdir, base + ".wav")
    if force or not os.path.exists(outfile):
        subprocess.run(
            ["ffmpeg", "-y", "-i", infile, *FFMPEG_ARGS, outfile], check=True
        )
    return outfile


def _stereo_load(wavfile: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(wavfile, sr=target_sr, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    return y, sr


def extract_features(wavfile: str, version, md5: str) -> None:
    """Populate a SongVersion with stereo-aware features.

    Mutates *version* in place to keep memory churn lower.
    """
    y, sr = _stereo_load(wavfile)
    left, right = y[0], y[1]
    mono = (left + right) / 2

    version.wav_path = wavfile
    version.sr = sr
    version.signal_left = left
    version.signal_right = right
    version.signal_mono = mono

    duration = librosa.get_duration(y=mono, sr=sr)
    version.duration_sec = float(duration)
    version.duration_fmt = format_duration(duration)

    meter = pyln.Meter(sr)
    version.loudness_left = float(meter.integrated_loudness(left))
    version.loudness_right = float(meter.integrated_loudness(right))
    version.loudness = float(meter.integrated_loudness(mono))

    rms_left = librosa.feature.rms(y=left)
    rms_right = librosa.feature.rms(y=right)
    rms_mono = librosa.feature.rms(y=mono)
    version.rms_left = float(np.mean(rms_left))
    version.rms_right = float(np.mean(rms_right))
    version.rms = float(np.mean(rms_mono))

    version.rms_balance = version.rms_left - version.rms_right
    version.loudness_balance = version.loudness_left - version.loudness_right

    if len(left) == len(right) and len(left) > 1:
        version.lr_corr = float(np.corrcoef(left, right)[0, 1])

    version.spectral_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr))
    )

    mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
    version.mfcc_mean = np.mean(mfcc, axis=1)

    version.md5 = md5


def _compute_tuning_and_chroma(y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
    bins = librosa.estimate_tuning(y=y, sr=sr)
    tuning_cents = float(bins * 100.0)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    if np.sum(chroma_mean) > 0:
        chroma_mean = chroma_mean / max(np.linalg.norm(chroma_mean), 1e-9)
    return tuning_cents, chroma_mean


def attach_pitch_features(versions: List) -> None:
    for v in versions:
        if v.signal_mono is None or v.sr is None:
            continue
        tc, ch = _compute_tuning_and_chroma(v.signal_mono, v.sr)
        v.tuning_cents = tc
        v.chroma_mean = ch
