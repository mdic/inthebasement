

# Glossary of technical terms

The analysis uses some technical audio measures. Here is what they mean in plain language:

## **LUFS (Loudness Units relative to Full Scale)**
  A standard measure of how loud a recording sounds to the human ear.

  - More negative values (e.g. –23 LUFS) = quieter.
  - Less negative values (e.g. –12 LUFS) = louder.

  LUFS is different from raw signal level because it models human perception, not just peak amplitude.

## **RMS (Root Mean Square energy)**
  A mathematical average of the audio signal’s energy.

  - Higher RMS = consistently louder audio.
  - Lower RMS = softer audio.

  Unlike LUFS, RMS is not perceptually weighted, but it still gives a good idea of “overall power”.

## **Spectral centroid**
  Indicates where the “centre of mass” of the spectrum lies.

  - Higher values = brighter sound (more treble).
  - Lower values = darker sound (more bass).

  Think of it as a rough proxy for how “bright” or “dull” a recording sounds.

## **MFCC (Mel-frequency cepstral coefficients)**
  A compact representation of the overall timbre of a recording, widely used in speech and music recognition.
  When compared across versions, MFCCs help quantify how similar two recordings “sound” beyond loudness or duration.

## **Stereo correlation (L/R correlation)**
  Measures how similar the left and right channels are.

  - 1.0 = channels are identical (mono).
  - 0.0 = completely different (maximal stereo).

  Negative values (rare in music) = channels are inverted relative to each other.

## **Pitch (measured in cents)**
  1 semitone = 100 cents.

  - +10 cents = slightly sharp (higher than reference).
  - –10 cents = slightly flat (lower than reference).

  Useful for detecting tape speed differences or detuned transfers.

## **Duration ratio**
  Compares the running time of a version to a reference.

  - 1.0 = same duration.
  - \>1.0 = version is longer.
  - <1.0 = version is shorter.

  Can reveal edits or speed differences.

---

## Why these measures matter

Together, these metrics give a multi-dimensional view of a recording:

- **LUFS and RMS** → how loud or powerful it sounds.
- **Spectral centroid** → how bright or dark it is.
- **MFCC similarity** → how close two versions “feel” in timbre.
- **Stereo metrics** → whether the mix is centred or imbalanced.
- **Pitch/Duration** → whether a transfer was sped up, slowed down, or detuned.
