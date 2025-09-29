import os
import shutil
import yaml

# ----------------------------
# Configuration
# ----------------------------
RESULTS_DIR = "results"
DOCS_DIR = "docs"
SONGS_DIR = os.path.join(DOCS_DIR, "songs")
ASSETS_DIR = os.path.join(DOCS_DIR, "assets", "songs")
SONGS_NAV = os.path.join(SONGS_DIR, "_songs_nav.yml")
INDEX_MD = os.path.join(SONGS_DIR, "index.md")

# order by "label" or "title"
ORDER_BY = "title"

# optional intro text for index.md
INDEX_INTRO = """# Songs Index

This page lists all analysed songs with their unique IDs and titles.
Click on any entry to view detailed comparisons and plots.
"""


# ----------------------------
# Helpers
# ----------------------------
def get_song_title(md_file):
    """Extract song title from the YAML front matter in the .md file"""
    title = None
    with open(md_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
    return title or "Untitled"


def rewrite_links(md_content, song_label, song_path):
    """Replace relative links to local files with paths into assets/"""
    for fname in os.listdir(song_path):
        if fname.endswith(".md"):
            continue
        md_content = md_content.replace(
            f"({fname})", f"(../assets/songs/{song_label}/{fname})"
        )
        md_content = md_content.replace(
            f"![{fname}]",
            f"![{fname}](../assets/songs/{song_label}/{fname})",
        )
    return md_content


# ----------------------------
# Main sync
# ----------------------------
def sync_docs():
    os.makedirs(SONGS_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    entries = []

    for song_label in os.listdir(RESULTS_DIR):
        song_path = os.path.join(RESULTS_DIR, song_label)
        if not os.path.isdir(song_path):
            continue

        md_file = os.path.join(song_path, f"{song_label}.md")
        if not os.path.exists(md_file):
            print(f"[WARN] No .md file found for {song_label}, skipping")
            continue

        # get title
        song_title = get_song_title(md_file)

        # copy assets
        dest_assets_dir = os.path.join(ASSETS_DIR, song_label)
        os.makedirs(dest_assets_dir, exist_ok=True)
        for fname in os.listdir(song_path):
            src = os.path.join(song_path, fname)
            if os.path.isfile(src) and not fname.endswith(".md"):
                shutil.copy2(src, os.path.join(dest_assets_dir, fname))

        # rewrite markdown
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_content = rewrite_links(md_content, song_label, song_path)

        dest_md = os.path.join(SONGS_DIR, f"{song_label}.md")
        with open(dest_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        entries.append({
            # "label": song_label,
            "title": song_title,
            "md_file": f"{song_label}.md",
        })

    # sort entries
    if ORDER_BY == "title":
        entries.sort(key=lambda e: e["title"].lower())
    else:
        entries.sort(key=lambda e: e["label"].lower())

    # write index.md
    index_lines = [INDEX_INTRO.strip(), ""]
    for e in entries:
        # index_lines.append(f"- [{e['label']} — {e['title']}](./{e['md_file']})")
        index_lines.append(f"- {e['title']}](./{e['md_file']})")
    with open(INDEX_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    # write nav
    nav_entries = [{"All Songs": "index.md"}]
    for e in entries:
        # nav_entries.append({f"{e['label']} — {e['title']}": e["md_file"]})
        nav_entries.append({f"{e['title']}": e["md_file"]})

    with open(SONGS_NAV, "w", encoding="utf-8") as f:
        yaml.dump(nav_entries, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] Synced {len(entries)} songs into docs with index + nav")


if __name__ == "__main__":
    sync_docs()
