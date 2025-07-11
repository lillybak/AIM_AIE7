#!/bin/env zsh

# for d in [0]*; do
#   if [ -f "$d/$1" ]; then
#     rm "$d/$1"
#     echo "🗑️ Removed from $d"
#   else
#     echo "⚠️ No file in $d"
#   fi
# done

#!/usr/bin/env bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
  echo "❗ Usage: $0 <parent_directory> <filename_to_remove>"
  exit 1
fi

PARENT_DIR="$1"
FILENAME="$2"

# Check if parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
  echo "❗ Error: Parent directory '$PARENT_DIR' does not exist."
  exit 1
fi

# Loop over subdirectories starting with 0
for d in "$PARENT_DIR"/0*/; do
  # Check if it is a directory
  if [ -d "$d" ]; then
    TARGET_FILE="$d/$FILENAME"
    if [ -f "$TARGET_FILE" ]; then
      rm "$TARGET_FILE"
      echo "🗑️ Removed $TARGET_FILE"
    else
      echo "⚠️ No file '$FILENAME' found in $d"
    fi
  fi
done

echo "✅ All done!"

