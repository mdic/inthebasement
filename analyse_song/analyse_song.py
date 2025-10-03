from __future__ import annotations
import argparse
import logging
from analyse_song.utils import (
    setup_logging,
    read_input_csv,
    SongAnalyser,
    load_runtime_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="List operations without executing"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep intermediate wav files"
    )
    parser.add_argument(
        "--song", type=str, help="Process only the given song_label", default=None
    )
    parser.add_argument(
        "--ref-label",
        nargs="+",
        default=None,
        help="One or more version labels to use as reference for pitch/speed analysis. Fallback order as listed.",
    )
    parser.add_argument(
        "--ref-title",
        nargs="+",
        default=None,
        help="One or more version labels to use as reference for extracting the song title. Fallback order as listed.",
    )
    parser.add_argument(
        "--input-csv", type=str, default="songs.csv", help="Path to input CSV mapping"
    )
    parser.add_argument(
        "--outroot", type=str, default="results", help="Output root directory"
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    load_runtime_config()  # side-effect: allow JSON overrides

    songs = read_input_csv(args.input_csv)
    analyser = SongAnalyser(
        outroot=args.outroot,
        dry_run=args.dry_run,
        keep=args.keep,
        ref_label=args.ref_label,
        ref_title=args.ref_title,
    )

    for song_label, data in songs.items():
        if args.song and song_label != args.song:
            continue
        analyser.analyse_song(song_label, data["title"], data["versions"])


if __name__ == "__main__":
    main()
