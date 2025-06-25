#!/bin/bash

# Migration script to update old checkpoints

OLD_CHECKPOINT_DIR="/path/to/old/checkpoints"
NEW_CHECKPOINT_DIR="/path/to/new/checkpoints"

mkdir -p "$NEW_CHECKPOINT_DIR"

for file in "$OLD_CHECKPOINT_DIR"/*.json; do
  filename=$(basename "$file")
  new_file="$NEW_CHECKPOINT_DIR/$filename"

  echo "Converting $file to $new_file"

  # Read old checkpoint
  old_data=$(cat "$file")

  # Assuming old_data has the structure we need to change
  # and that jq is used to modify JSON data

  # JSON transformation logic here (using jq, or manually)
  # For demonstration purposes, let's say we add a `version` field and convert paths
  # This is just an example and needs to be customized based on actual requirements

  updated_data=$(echo "$old_data" |
    jq '. + {version: "2.0"}' |
    jq 'if .memoryState then
      .memoryState.shortTerm = .memoryState.shortTerm | map(./1000) | map(floor) |
      .memoryState.longTerm = .memoryState.longTerm | map(./1000) | map(floor)
    else
      .
    end')

  echo "$updated_data" > "$new_file"

done

echo "Migration complete."

