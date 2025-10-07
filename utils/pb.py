import csv
import os
import subprocess
import sys

DEFAULT_CSV = os.path.expanduser("songs.csv")


def get_filepaths(label, csv_path):
    files = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["song_label"] == label:
                files.append(row["filepath"])
    return files


def is_vlc_running():
    result = subprocess.run(["pgrep", "-x", "vlc"], stdout=subprocess.DEVNULL)
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print("Usage: pb [song_label] [--keep] [--csv /path/to/songs.csv]")
        sys.exit(1)

    label = sys.argv[1]
    keep = "--keep" in sys.argv

    # trova indice e path CSV se specificato
    if "--csv" in sys.argv:
        idx = sys.argv.index("--csv")
        try:
            csv_path = sys.argv[idx + 1]
        except IndexError:
            print("Error: missing path after '--csv'")
            sys.exit(1)
    else:
        csv_path = DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    files = get_filepaths(label, csv_path)
    if not files:
        print(f"No files found for label '{label}' in {csv_path}")
        sys.exit(1)

    vlc_running = is_vlc_running()

    if not keep:
        # Replace playlist
        if vlc_running:
            subprocess.run(["pkill", "-x", "vlc"])
        subprocess.Popen(["vlc", "--one-instance", "--playlist-enqueue"] + files)
    else:
        # Keep existing playlist
        if not vlc_running:
            subprocess.Popen(["vlc", "--one-instance", "--playlist-enqueue"] + files)
        else:
            subprocess.run(["vlc", "--one-instance", "--playlist-enqueue"] + files)


if __name__ == "__main__":
    main()
