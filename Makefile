# Makefile for audio analysis & site generation

# Variables
PYRUN = uv run
ANALYSE = $(PYRUN) utils/analyse_song.py
SYNC = $(PYRUN) utils/sync_docs.py
MKDOCS = $(PYRUN) mkdocs

# Main Commands
.PHONY: all an sync serve clean

# Run default analysis and sync results to mkdocs 'docs' folder
all: an sync

# Analyse songs (optional ARGS="--keep --song LABEL --ref-label ALBUM")
# e.g. make an ARGS="--keep --song yagn"
an:
	$(ANALYSE) $(ARGS)

# Sync results to mkdocs 'docs' folder
sync:
	$(SYNC)

# Serve with MkDocs
serve:
	$(MKDOCS) serve

# Sync and serve:
sas:
	$(SYNC)
	$(MKDOCS) serve

# Pulizia file temporanei/intermedi (se usi .wav intermedi)
clean:
	find results -name "*.wav" -delete
