from __future__ import annotations
import os
import pandas as pd
from .models import SongAnalysis


def generate_markdown(
    analysis: SongAnalysis, ref_label: str | None = None, ref_title: str | None = None
) -> str:
    song_label = analysis.song_label
    song_title = analysis.song_title
    outdir = analysis.outdir
    md_path = os.path.join(outdir, f"{song_label}.md")

    with open(md_path, "w", encoding="utf-8") as fmd:
        fmd.write("---\n")
        fmd.write(f'title: "{song_title}"\n')
        fmd.write(f"song_label: {song_label}\n")
        if ref_title is not None:
            fmd.write(f"ref_title_version: {ref_title}\n")
        fmd.write("---\n\n")

        if ref_title:
            fmd.write(f"# {song_title}\n\n")
            fmd.write(f"**(title taken from {ref_title})**\n\n")
        else:
            fmd.write(f"# {song_title}\n\n")

        fmd.write(f"[](){{ #{song_label} }}\n\n")
        notes_path = os.path.join("notes/", f"{song_label}.md")
        if os.path.exists(notes_path):
            with open(notes_path, encoding="utf-8") as nf:
                fmd.write("## Notes\n\n")
                fmd.write(nf.read() + "\n\n")

        df = pd.read_csv(os.path.join(outdir, f"{song_label}-features.csv"))
        fmd.write("## Details\n\n")
        fmd.write(df.to_markdown(index=False))
        fmd.write("\n\n")

        v0 = analysis.versions[0]
        fmd.write("## Plots\n")
        fmd.write(f"![Waveforms (Mono)]({v0.plots['waveform_mono']})\n\n")
        fmd.write(f"![Waveforms (Left)]({v0.plots['waveform_L']})\n\n")
        fmd.write(f"![Waveforms (Right)]({v0.plots['waveform_R']})\n\n")
        fmd.write(f"![Radar Plot]({v0.plots['radar_plot']})\n\n")

        # Code below should notify when radar plot lines overlap
        # however overlapping strategy seems to be too sensitive, leading to false positive
        # dup_file = os.path.join(outdir, f"{song_label}-radar_duplicates.txt")
        # if os.path.exists(dup_file):
        #     fmd.write("\n**Note:** Some versions overlap in the radar plot:\n\n")
        #     with open(dup_file, encoding="utf-8") as dfh:
        #         for line in dfh:
        #             fmd.write(f"- {line.strip()}\n")
        #     fmd.write("\n")

        fmd.write(f"![MFCC Similarity]({v0.plots['similarity_plot']})\n\n")

        pitch_csv = os.path.join(outdir, f"{song_label}-pitch_report.csv")
        pitch_summary = os.path.join(outdir, f"{song_label}-pitch_summary.txt")
        pitch_plot = os.path.join(outdir, f"{song_label}-pitch_offsets.png")
        if os.path.exists(pitch_csv):
            fmd.write("## Pitch & Speed Analysis (cents)\n\n")
            ref_used = ref_label or analysis.versions[0].label
            fmd.write(f"Reference version: **{ref_used}**\n\n")
            dfp = pd.read_csv(pitch_csv)
            fmd.write(dfp.to_markdown(index=False))
            fmd.write("\n\n")
            if os.path.exists(pitch_plot):
                fmd.write(
                    f"![Pitch Offsets]({analysis.versions[0].plots['pitch_plot']})\n\n"
                )
            if os.path.exists(pitch_summary):
                with open(pitch_summary, "r", encoding="utf-8") as ps:
                    fmd.write("````text\n")
                    fmd.write(ps.read())
                    fmd.write("\n````\n\n")

        fmd.write("## Stereo Balance\n\n")
        for v in analysis.versions:
            fmd.write(f"### {v.label}\n\n")
            fmd.write(f"![STFT Spectrogram (Left)]({v.plots['spectrogram_L']})\n\n")
            fmd.write(f"![STFT Spectrogram (Right)]({v.plots['spectrogram_R']})\n\n")
            fmd.write(f"![Mel Spectrogram (Left)]({v.plots['melspec_L']})\n\n")
            fmd.write(f"![Mel Spectrogram (Right)]({v.plots['melspec_R']})\n\n")
            fmd.write(f"![Stereo Balance Bars]({v.plots['balance_plot']})\n\n")

        fmd.write("## Spectrograms (Mono)\n\n")
        for v in analysis.versions:
            fmd.write(f"### {v.label}\n\n")
            fmd.write(f"![STFT Spectrogram (Mono)]({v.plots['spectrogram_Mono']})\n\n")
            fmd.write(f"![Mel Spectrogram (Mono)]({v.plots['melspec_Mono']})\n\n")

    return md_path


def smoke_check_outputs(analysis: SongAnalysis) -> None:
    song_label = analysis.song_label
    outdir = analysis.outdir
    expected = [
        os.path.join(outdir, f"{song_label}.md"),
        os.path.join(outdir, f"{song_label}-waveforms_Mono.png"),
        os.path.join(outdir, f"{song_label}-waveforms_L.png"),
        os.path.join(outdir, f"{song_label}-waveforms_R.png"),
        os.path.join(outdir, f"{song_label}-radar_plot.png"),
        os.path.join(outdir, f"{song_label}-similarity_matrix.png"),
        os.path.join(outdir, f"{song_label}-features.csv"),
        os.path.join(outdir, f"{song_label}-features_normalised.csv"),
    ]
    for v in analysis.versions:
        expected += [
            os.path.join(outdir, f"{song_label}.md"),
            os.path.join(outdir, f"{song_label}-{v.label}_spectrogram_Mono.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_Mono.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_spectrogram_L.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_L.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_spectrogram_R.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_melspec_R.png"),
            os.path.join(outdir, f"{song_label}-{v.label}_balance.png"),
        ]

    missing = [p for p in expected if not os.path.exists(p)]
    if missing:
        print("[WARN] Missing outputs:")
        for m in missing:
            print(f"  - {m}")
