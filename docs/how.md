# Song Analysis Script — Explanation and Interpretation

This page explains what the `analyse_song.py` script does, and how to interpret the results it produces. The goal is to make the analysis accessible to anyone, whether or not you have prior experience with audio engineering.

---

## What the script does (technical overview)

The script automates the process of comparing different versions of the same song:

1. **Preparation**
    - Converts each input file (`.flac`, `.shn`, etc.) into a temporary `.wav` at 44.1 kHz stereo.
    - Assigns an [*album label*](albums/index.md) (e.g. `BS11`, `CAR`) based on the file path.
    - Generates an MD5 checksum to skip re-analysing files already processed.

2. **Feature extraction**
    - Loads the audio as stereo, keeping left and right channels separate.
    - Computes:
        - **Duration** (in seconds and formatted as `MM:SS:ms`).
        - **Integrated loudness (LUFS)**.
        - **RMS energy** (average amplitude).
        - **Stereo balance** (differences between left/right).
        - **Left–Right correlation** (measure of stereo similarity).
        - **Spectral centroid** (perceived brightness).
        - **MFCCs** ([Mel-frequency cepstral coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum): a fingerprint of timbre).

3. **Plotting**
    - **Waveforms**: mono, left, and right signals, plotted separately and comparatively.
    - **Spectrograms (STFT and Mel)**: visualisations of frequency content over time, for mono and stereo channels.
    - **Stereo balance bar charts**: RMS and loudness side-by-side for left and right channels.
    - **Similarity matrix**: cosine similarity of MFCCs between versions, shown as a heatmap.
    - **Radar plot**: compares features (duration, loudness, RMS, spectral centroid) on the same scale.
     To avoid extreme values dominating, the plot uses **5th–95th percentile trimming** instead of raw min/max scaling.

4. **Pitch and speed analysis**
    - Estimates global tuning offset (in cents) and average chroma profile.
    - Compares each version to a chosen reference version (`--ref-label`).
    - Detects whether a version is sharper or flatter (pitch shifted).
    - Estimates playback speed differences (if pitch and duration disagree).
    - Produces:
        - CSV (`<song_label>-pitch_report.csv`).
        - Text summary (`<song_label>-pitch_summary.txt`).
        - Bar plot of pitch offsets in cents (`<song_label>-pitch_offsets.png`).

5. **Outputs**
    - Per-song CSVs:
        - `<song_label>-features.csv`
        - `<song_label>-features_normalised.csv`
        - `<song_label>-similarity_matrix.csv`
    - Markdown report (`<song_label>.md`) with tables, notes, plots, and summaries.
    - PNG plots for each version and comparative plots across versions.
    - JSONL metadata log (`metadata.jsonl`) for cumulative storage.

---

## How to interpret the results

#### **Waveforms**
  Show the “shape” of the song’s amplitude. Sudden cut-offs indicate edits, while broader differences may show variations in mixing or mastering.

#### **Spectrograms**

  Display the distribution of energy across frequencies over time.

  - A brighter image = more high frequencies.
  - Differences in colour bands between versions can indicate EQ or recording quality changes.

#### **Stereo balance plots**
  RMS and LUFS are shown for left and right.

  - Balanced versions → bars of equal height.
  - Stronger left or right bars → imbalance in mixing or tape transfer.

#### **Similarity matrix**
  A square heatmap of MFCC similarity.

  - Dark squares = very similar versions.
  - Lighter areas = audible differences (different takes, mixes, or transfers).

#### **Radar plot**
  Each axis = one feature (duration, loudness, RMS, spectral centroid).

  - Each version is drawn as a polygon.
  - Overlapping shapes → similar versions.
  - Spread-out shapes → different versions.
  - Percentile scaling (5–95%) ensures that outliers don’t flatten the differences.

#### **Pitch and speed analysis**
##### **Pitch offsets (cents)**:
  - Positive values → version is sharper (higher pitch).
  - Negative values → version is flatter (lower pitch).

##### **Speed factor**:
  - 1.0 = same speed as reference.
  - \>1.0 = plays faster, <1.0 = slower.


##### **Duration ratio**:
  - Confirms whether differences in pitch correlate with tape speed or editing.

---

## In summary

The script produces a **technical fingerprint** for each version of a song. This can be used to answer questions like:


- Is one version longer/shorter?
- Is it louder or quieter?
- Does it sound brighter or duller?
- Is it pitched higher or lower (tape speed difference)?
- Is the stereo image balanced?
- How close are two recordings, really?
