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
NAV_YML = os.path.join(SONGS_DIR, ".nav.yml")  # <- plugin awesome-nav legge questo
INDEX_MD = os.path.join(SONGS_DIR, "index.md")

# order by "label" or "title"
ORDER_BY = "title"

INDEX_INTRO = """# Songs Index

This page lists all analysed songs with their unique IDs and titles.
Click on any entry to view detailed comparisons and plots.
"""


# ----------------------------
# Helpers
# ----------------------------
def get_song_title(md_file):
    """Extract song title from the YAML front matter in the .md file."""
    title = None
    with open(md_file, "r", encoding="utf-8") as f:
        in_front = False
        for line in f:
            if line.strip() == "---":
                in_front = not in_front
                continue
            if in_front and line.strip().startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
    return title or "Untitled"


def rewrite_links(md_content, song_label, song_path):
    """Replace local links with ../assets/songs/<label>/..."""
    for fname in os.listdir(song_path):
        if fname.endswith(".md"):
            continue
        md_content = md_content.replace(
            f"({fname})", f"(../assets/songs/{song_label}/{fname})"
        )
        md_content = md_content.replace(
            f"]({fname})", f"](../assets/songs/{song_label}/{fname})"
        )
    return md_content


def inject_notes(md_content, notes_file):
    """Insert contents of notes.md (if present) after the main title."""
    if not os.path.exists(notes_file):
        return md_content

    with open(notes_file, "r", encoding="utf-8") as f:
        notes_content = f.read().strip()
    if not notes_content:
        return md_content

    lines = md_content.splitlines()
    injected = []
    inserted = False
    for i, line in enumerate(lines):
        injected.append(line)
        # Dopo il primo heading di livello 1 (# titolo)
        # if not inserted and line.startswith("# "):
        if not inserted and line.startswith("[](){"):
            injected.append("")
            injected.append("## Notes")
            injected.append("")
            injected.append(notes_content)
            injected.append("")
            inserted = True
    return "\n".join(injected)


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

        # read title
        song_title = get_song_title(md_file)

        # copy assets (everything except .md)
        dest_assets_dir = os.path.join(ASSETS_DIR, song_label)
        os.makedirs(dest_assets_dir, exist_ok=True)
        for fname in os.listdir(song_path):
            src = os.path.join(song_path, fname)
            if os.path.isfile(src) and not fname.endswith(".md"):
                shutil.copy2(src, os.path.join(dest_assets_dir, fname))

        # rewrite markdown links and copy md
        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()
        md_content = rewrite_links(md_content, song_label, song_path)

        # inject notes.md if present
        notes_file = os.path.join("notes/", f"{song_label}.md")
        md_content = inject_notes(md_content, notes_file)

        dest_md = os.path.join(SONGS_DIR, f"{song_label}.md")
        with open(dest_md, "w", encoding="utf-8") as f:
            f.write(md_content)

        entries.append(
            {
                "label": song_label,
                "title": song_title,
                "md_file": f"{song_label}.md",
            }
        )

    # sort entries
    if ORDER_BY == "title":
        entries.sort(key=lambda e: e["title"].lower())
    else:
        entries.sort(key=lambda e: e["label"].lower())

    # write songs/index.md
    index_lines = [INDEX_INTRO.strip(), ""]
    for e in entries:
        index_lines.append(f"- [{e['title']}](./{e['md_file']})")
    with open(INDEX_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")

    # write songs/.nav.yml for awesome-nav
    nav_yaml = {
        "title": "Songs",
        "nav": [{"All Songs": "index.md"}],
        # Line below activates the list of all songs in the navbar
        # + [{e["title"]: e["md_file"]} for e in entries],
    }
    with open(NAV_YML, "w", encoding="utf-8") as f:
        yaml.dump(nav_yaml, f, sort_keys=False, allow_unicode=True)

    print(
        f"[OK] Synced {len(entries)} songs into docs with index + .nav.yml (titles only)"
    )


if __name__ == "__main__":
    sync_docs()
