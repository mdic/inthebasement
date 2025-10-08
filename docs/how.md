
# Song Analysis Script — Explanation and Interpretation

All the details included in this website are produced using a custom-made tool called `analyse_song`, available under open-source GPL license here.
The analyse_song tool is a complete audio analysis and visualisation pipeline designed to compare different recorded versions of the same song.
It extracts detailed acoustic features from each version, produces multiple visual summaries, and compiles all results into a structured report.

---

## How `analyse_song` works

When you run

`uv run -m analyse_song --song [LABEL]`

the program follows a precise sequence of steps — converting, analysing, plotting, and reporting — to describe how each version of the song differs in timing, loudness, balance, and spectral profile.

All outputs are saved in a dedicated folder under `results/[song_label]/`.

### 1. Input

The process begins with a CSV file (usually songs.csv) that lists every available version of each song.
Each entry specifies:

- the song label (a short code such as `yagn_t2`),
- the file path to the audio,
- the title of the song, and
- the disc and track numbers identifying the version.

The script can process all songs or a specific one via the `--song` option.

### 2. Audio Preparation
Each file is:

- Converted to a standard .wav format (stereo, 44.1 kHz).
- Checked for duplicates using an MD5 fingerprint — if the same file has already been processed, it is skipped.
- Loaded into memory as left, right, and mono (combined) signals for further measurement.

### 3. Feature Extraction:
From each version, the program computes a set of audio features that quantify its sound properties:

- Duration (in seconds and in mm:ss format)
- Loudness of each channel (in LUFS)
- Root Mean Square (RMS) energy — an indicator of perceived intensity
- Left/right balance for both loudness and RMS
- Stereo correlation (how similar the two channels are)
- Spectral centroid (the "brightness" of the sound)
- MFCC coefficients for timbral similarity
- Tuning and chroma for pitch and harmonic content

These features are saved in two CSV files:

- `*-features.csv` (raw values)
- `*-features_normalised.csv` (values scaled between 0 and 1 for easy comparison)

### 4. Visualisation

The system then generates several plots to make differences visible at a glance:

- **Waveforms** (mono, left, right) – overall shape of the sound.
- **Spectrograms** – colour-coded frequency content over time.
- **Mel-spectrograms** – an alternative scale reflecting human hearing.
- **Stereo balance bars** – comparing left/right loudness and RMS.
- **Radar plot** – summarising duration, loudness, RMS, and brightness for all versions.
- **MFCC similarity matrix** – measuring how close each version sounds to the others.
- **Pitch offset chart** – showing how tuning differs between versions.

Each plot is stored as a `.png` file with a consistent naming pattern inside the song’s results folder.

### 5. Pitch and Speed Comparison
After extracting features, the system analyses relative pitch and tempo differences:

- Finds a reference version (specified via `--ref-label` or defaulting to the first).
- Compares each version’s chroma pattern to the reference.
- Detects semitone shifts, tuning offsets (in cents), and estimated playback speed.
- Writes a detailed summary to:
    - `*-pitch_report.csv` (structured data)
    - `*-pitch_summary.txt` (readable text)
    - `*-pitch_offsets.png` (visual plot of tuning deviations)

### 6. Report Generation (Webpages)
Finally, all results are compiled into a Markdown file named:

`results/[song_label]/[song_label].md`

This report includes:

- Song title and reference information
- A Markdown table of features
- All generated plots embedded inline
- Notes on overlapping radar plots (if any)
- Pitch and speed analysis summary
- References to original files and version labels

This Markdown report is then integrated into a `docs` folder, serving files for this MkDocs site.

### 7. Cumulative Metadata Log
Each time a version is analysed, a short summary entry is appended to the central [`metadata.jsonl`](https://github.com/mdic/inthebasement/blob/main/metadata.jsonl) file, a cumulative, machine-readable record of all analyses performed — useful for indexing, cross-referencing, or database import.

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
