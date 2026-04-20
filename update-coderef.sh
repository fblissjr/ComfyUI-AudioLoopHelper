#!/bin/bash
# Script to update all git repositories in the custom_nodes directory.
# By default, it only pulls updates.
# Use the '--install' flag to also install requirements (skipping torch).

# Exit immediately if a command fails
set -e

# --- Check for the --install parameter ---
INSTALL_REQS=false
if [[ "$1" == "--install" ]]; then
  INSTALL_REQS=true
  echo "✅ Requirement installation is ENABLED for this run."
else
  echo "ℹ️  Running in pull-only mode. Use './update_custom_nodes.sh --install' to also install requirements."
fi

# Directory containing your custom nodes
CUSTOM_NODES_DIR="coderef"

# Check if the custom_nodes directory exists
if [ ! -d "$CUSTOM_NODES_DIR" ]; then
  echo "❌ Error: Directory not found: '$CUSTOM_NODES_DIR'"
  echo "Please make sure you are in your main ComfyUI directory before running this script."
  exit 1
fi

# Loop through each item in the custom_nodes directory
for dir in "$CUSTOM_NODES_DIR"/*/; do
  # Check if it's a git repository
  if [ -d "${dir}.git" ]; then
    echo "---"
    echo "Updating $(basename "$dir")"

    # Use a subshell ( ... ) to change directory temporarily
    (
      cd "$dir"
      echo "  -> Pulling latest changes..."
      git pull

      # --- Conditional Requirement Installation ---
      # Only run this block if the --install flag was used
      if [ "$INSTALL_REQS" = true ]; then
        if [ -s "requirements.txt" ]; then
          echo "  -> Installing dependencies (skipping torch)..."
          # Use grep to filter out torch before piping to uv
          grep -v '^torch' requirements.txt | uv pip install -r -
        else
          echo "  -> No requirements.txt found or file is empty."
        fi
      fi
    )
  fi
done

echo "---"
echo "✅ All custom node repositories have been updated!"
