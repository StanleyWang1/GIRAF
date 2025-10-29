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
  
  # Add an option to remove all environments
  printf "  [%d] Remove all virtual environments\n" $((${#VENV_DIRS[@]}+1))

  read -p "Enter the number [1-$((${#VENV_DIRS[@]}+1))] of the venv to remove: " idx
  if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 1 || idx > ${#VENV_DIRS[@]}+1 )); then
    echo "Error: Invalid choice. Please enter a number between 1 and $((${#VENV_DIRS[@]}+1))."
    exit 1
  fi

  # Check if user wants to remove all environments
  if (( idx == ${#VENV_DIRS[@]}+1 )); then
    echo "→ Removing all virtual environments..."
    for venv in "${VENV_DIRS[@]}"; do
      echo "  Removing $(basename "$venv")..."
      rm -rf "$venv"
    done
    echo "All virtual environments have been removed."
    exit 0
  else
    VENV_DIR="${VENV_DIRS[idx-1]}"
  fi
else
  VENV_DIR="${VENV_DIRS[0]}"
fi

echo "→ Removing: $(basename "$VENV_DIR") ($VENV_DIR)"

# Confirm deletion with Yes as default
read -p "Are you sure you want to remove this virtual environment? [Y/n]: " confirm
if [[ -z "$confirm" || "$confirm" =~ ^[Yy]$ ]]; then
  rm -rf "$VENV_DIR"
  echo "Virtual environment has been removed."
else
  echo "Operation cancelled."
fi

exit 0