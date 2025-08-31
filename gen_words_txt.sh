#!/usr/bin/env bash
set -euo pipefail

# 1. Find the first directory named "words"
WORDS_DIR=$(find . -type d -name words -print -quit)

if [[ -z "$WORDS_DIR" ]]; then
  echo "Error: no directory named 'words' found." >&2
  exit 1
fi

# 2. List .wav files in it, strip extension, and write to words.txt
find "$WORDS_DIR" -maxdepth 1 -type f -name '*.wav' \
  -exec basename {} .wav \; \
  > words.txt

echo "Wrote list of files to words.txt"
