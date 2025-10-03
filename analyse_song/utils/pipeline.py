from __future__ import annotations
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import load_runtime_config
from .models import SongVersion, SongAnalysis
from .io_utils import (
    append_jsonl,
    compute_md5,
    load_existing_md5s,
    read_input_csv,
)
from .audio import convert_to_wav, extract_features, attach_pitch_features
from .plotting import (
    plot_waveforms,
    plot_spectrograms,
    plot_mel_spectrograms,
    plot_similarity_matrix,
    plot_radar_chart,
    plot_lr_balance_bars,
)
from .reporting import generate_markdown, smoke_check_outputs


def pick_first_valid(candidates: Optional[Iterable[str]]) -> Optional[str]:
    """Return the first non-empty string in *candidates*, else None."""
    if not candidates:
        return None
    for c in candidates:
        if c and str(c).strip():
            return c
    return None


class SongAnalyser:
    """Encapsulates the end-to-end pipeline for one or more songs.

    This class makes the flow linear and testable. Methods are factored so
    that unit tests can target small pieces without touching audio.
    """

    def __init__(
        self,
        outroot: str = "results",
        dry_run: bool = False,
        keep: bool = False,
        ref_label: Optional[List[str]] = None,
        ref_title: Optional[List[str]] = None,
    ):
        self.outroot = outroot
        self.dry_run = dry_run
        self.keep = keep
        self.ref_label = ref_label
        self.ref_title = ref_title
        self.cfg = load_runtime_config()
        self.existing_md5s = load_existing_md5s()

    # --------------------- high-level API ---------------------

    def analyse_song(
        self, song_label: str, song_title: str, versions_rows: List[Dict]
    ) -> Optional[SongAnalysis]:
        outdir = os.path.join(self.outroot, song_label)
        os.makedirs(outdir, exist_ok=True)

        versions: List[SongVersion] = []
        for row in versions_rows:
            file_path = (
                os.path.expanduser(row["filepath"])
                if "filepath" in row
                else os.path.expanduser(row["file"])
            )
            label = row.get("label") or self._derive_version_label(file_path)
            disc = int(row.get("version_disc") or row.get("disc") or 0)
            track = int(row.get("version_track") or row.get("track") or 0)
            sv = SongVersion(
                label=label,
                file=file_path,
                disc=disc,
                track=track,
                song_title=row.get("song_title"),
            )

            md5 = compute_md5(file_path)
            if md5 in self.existing_md5s:
                logging.info(f"[SKIP] %s already analysed (md5=%s)", file_path, md5)
                continue

            if self.dry_run:
                logging.info(
                    "[DRY-RUN] Would analyse: %s (label=%s, md5=%s)",
                    file_path,
                    label,
                    md5,
                )
                continue

            sv.md5 = md5
            wav = convert_to_wav(file_path)
            sv.wav_path = wav
            extract_features(wav, sv, md5)
            versions.append(sv)

            if not self.keep and wav and os.path.exists(wav):
                try:
                    os.remove(wav)
                except OSError:
                    pass

        if self.dry_run or not versions:
            return None

        # CSV with stereo metrics
        df = pd.DataFrame(
            [
                {
                    "label": v.label,
                    "orig_file": os.path.basename(v.file),
                    "md5": v.md5,
                    "disc": v.disc,
                    "track": v.track,
                    "duration_sec": v.duration_sec,
                    "duration_fmt": v.duration_fmt,
                    "loudness": v.loudness,
                    "loudness_left": v.loudness_left,
                    "loudness_right": v.loudness_right,
                    "loudness_balance": v.loudness_balance,
                    "rms": v.rms,
                    "rms_left": v.rms_left,
                    "rms_right": v.rms_right,
                    "rms_balance": v.rms_balance,
                    "lr_corr": v.lr_corr,
                    "spectral_centroid": v.spectral_centroid,
                }
                for v in versions
            ]
        )
        df.to_csv(os.path.join(outdir, f"{song_label}-features.csv"), index=False)

        # Normalised subset
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
            cmin, cmax = df[col].min(), df[col].max()
            df_norm[col] = (df[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0
        df_norm.to_csv(
            os.path.join(outdir, f"{song_label}-features_normalised.csv"), index=False
        )

        # Attach pitch features here so we can liberate/free them after the plots
        attach_pitch_features(versions)

        # Plots
        plot_waveforms(versions, outdir, song_label)
        plot_spectrograms(versions, outdir, song_title, song_label)
        plot_mel_spectrograms(versions, outdir, song_title, song_label)
        plot_similarity_matrix(versions, outdir, song_label)
        plot_radar_chart(versions, outdir, song_label)
        plot_lr_balance_bars(versions, outdir, song_label)

        # Free heavy arrays before reporting
        for v in versions:
            v.signal_mono = None
            v.signal_left = None
            v.signal_right = None

        # Pitch/Speed
        pitch_csv, pitch_summary, pitch_plot = self._pitch_speed_analysis(
            versions, outdir, song_label
        )

        # Resolve title provider (ref_title fallback)
        ref_title_used, resolved_title = self._resolve_title(versions, song_title)

        analysis = SongAnalysis(
            song_label=song_label,
            song_title=resolved_title,
            versions=versions,
            outdir=outdir,
        )
        generate_markdown(
            analysis, ref_label=self._ref_label_used(versions), ref_title=ref_title_used
        )

        # JSONL cumulative log
        append_jsonl(
            "metadata.jsonl",
            [
                {
                    "song_label": song_label,
                    "filename": os.path.basename(v.file),
                    "version_label": v.label,
                    "version_disc": v.disc,
                    "version_track": v.track,
                    "duration": v.duration_fmt,
                    "duration_sec": v.duration_sec,
                    "loudness": f"{v.loudness:.2f} LUFS"
                    if v.loudness is not None
                    else None,
                    "rms": v.rms,
                    "spectral_centroid": f"{v.spectral_centroid:.2f} Hz"
                    if v.spectral_centroid is not None
                    else None,
                    "md5": v.md5,
                    "spectrogram": f"{song_label}-{v.label}_spectrogram_Mono.png",
                    "melspec": f"{song_label}-{v.label}_melspec_Mono.png",
                }
                for v in versions
            ],
        )

        smoke_check_outputs(analysis)
        logging.info(
            "[OK] Analysis completed for '%s' (%s). Results in: %s",
            song_label,
            resolved_title,
            outdir,
        )
        return analysis

    # --------------------- helpers ---------------------

    def _derive_version_label(self, filepath: str) -> str:
        m = self.cfg.get("VERSION_LABEL_MAP", {})
        for key, val in m.items():
            if key in filepath:
                return val
        return "UNKNOWN"

    def _ref_label_used(self, versions: List[SongVersion]) -> str:
        # If user provided a list, use first match; else default to first version
        if self.ref_label:
            for cand in self.ref_label:
                for v in versions:
                    if v.label == cand:
                        return cand
        return versions[0].label

    def _resolve_title(
        self, versions: List[SongVersion], default_title: str
    ) -> Tuple[Optional[str], str]:
        if not self.ref_title:
            return None, default_title
        # Build lookup from available rows (may carry song_title)
        version_to_title = {v.label: v.song_title for v in versions if v.song_title}
        for cand in self.ref_title:
            if cand in version_to_title and version_to_title[cand].strip():
                return cand, version_to_title[cand]
        return None, default_title

    def _best_semitone_shift(
        self, chroma_ref: np.ndarray, chroma_cmp: np.ndarray
    ) -> Tuple[int, float]:
        best_k, best_sim = 0, -1e9
        for k in range(12):
            sim = float(np.dot(chroma_ref, np.roll(chroma_cmp, k)))
            if sim > best_sim:
                best_sim, best_k = sim, k
        if best_k > 6:
            best_k -= 12
        return int(best_k), float(best_sim)

    def _pitch_speed_analysis(
        self, versions: List[SongVersion], outdir: str, song_label: str
    ):
        if not versions:
            return None, None, None
        # attach_pitch_features(versions)
        ref_label_used = self._ref_label_used(versions)
        ref = next((v for v in versions if v.label == ref_label_used), versions[0])

        rows = []
        for v in versions:
            shift, sim = self._best_semitone_shift(ref.chroma_mean, v.chroma_mean)
            speed_from_pitch = float(2.0 ** (shift / 12.0))
            dur_ratio = float(ref.duration_sec / max(v.duration_sec or 1e-9, 1e-9))
            rows.append(
                {
                    "song_label": song_label,
                    "ref_label": ref.label,
                    "cmp_label": v.label,
                    "cmp_file": os.path.basename(v.file),
                    "tuning_cents_cmp": float(v.tuning_cents),
                    "tuning_cents_ref": float(ref.tuning_cents),
                    "delta_tuning_cents": float(v.tuning_cents - ref.tuning_cents),
                    "semitone_shift_vs_ref": int(shift),
                    "chroma_similarity": float(sim),
                    "speed_factor_from_pitch": speed_from_pitch,
                    "duration_ratio_ref_over_cmp": dur_ratio,
                }
            )

        pitch_csv = os.path.join(outdir, f"{song_label}-pitch_report.csv")
        pd.DataFrame(rows).to_csv(pitch_csv, index=False)

        summary_path = os.path.join(outdir, f"{song_label}-pitch_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"Pitch/Speed analysis (reference = {ref.label})\n")
            fh.write("=" * 60 + "\n\n")
            for r in rows:
                fh.write(
                    f"{r['cmp_label']} - {r['cmp_file']}: shift={r['semitone_shift_vs_ref']} st ; "
                    f"Δtuning={r['delta_tuning_cents']:.1f} cents ; "
                    f"speed_from_pitch={r['speed_factor_from_pitch']:.4f} ; "
                    f"duration_ratio(ref/cmp)={r['duration_ratio_ref_over_cmp']:.4f}\n"
                )

        # Plot offsets
        seen = {}
        labels = []
        for r in rows:
            base = r["cmp_label"]
            count = seen.get(base, 0) + 1
            seen[base] = count
            if count == 1:
                labels.append(base)
            else:
                labels.append(f"{base}#{count}")

        shifts_cents = [r["delta_tuning_cents"] for r in rows]

        plt.figure(figsize=(6, 4))
        plt.bar(labels, shifts_cents, color="orange")
        plt.axhline(0, color="k", linewidth=1)
        plt.ylabel("Δ Tuning (cents vs reference)")
        plt.title(f"Pitch offsets (cents) — {song_label} (ref={ref.label})")
        plt.tight_layout()
        from .io_utils import unique_path

        path, name = unique_path(
            os.path.join(outdir, f"{song_label}-pitch_offsets.png")
        )
        plt.savefig(path, dpi=150)
        versions[0].plots["pitch_plot"] = name
        plt.close()

        return pitch_csv, summary_path, path
