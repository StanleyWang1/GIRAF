#!/usr/bin/env bash
set -e

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

# Find all venv-* directories in project root
mapfile -t VENV_DIRS < <(find "$PROJECT_ROOT" -maxdepth 1 -type d -name 'venv-*' | sort)

if [ ${#VENV_DIRS[@]} -eq 0 ]; then
  echo "Error: No virtual environments (venv-*) found in $PROJECT_ROOT."
  exit 1
fi

# If multiple venvs, prompt choice with a clear question
if [ ${#VENV_DIRS[@]} -gt 1 ]; then
  echo "Detected multiple Python virtual environments:"
  for i in "${!VENV_DIRS[@]}"; do
    name=$(basename "${VENV_DIRS[i]}")
    printf "  [%d] %s\n" $((i+1)) "$name"
  done

  read -p "Enter the number [1-${#VENV_DIRS[@]}] of the venv to activate: " idx
  if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 1 || idx > ${#VENV_DIRS[@]} )); then
    echo "Error: Invalid choice. Please enter a number between 1 and ${#VENV_DIRS[@]}."
    exit 1
  fi

  VENV_DIR="${VENV_DIRS[idx-1]}"
else
  VENV_DIR="${VENV_DIRS[0]}"
fi

echo "→ Activating: $(basename "$VENV_DIR") ($VENV_DIR)"

# Activate the chosen venv
source "$VENV_DIR/bin/activate"
