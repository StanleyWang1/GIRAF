#!/bin/bash

# Exit on error
set -e

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

# Find all venv-* directories in project root
mapfile -t VENV_DIRS < <(find "$PROJECT_ROOT" -maxdepth 1 -type d -name 'venv-*' | sort)

if [ ${#VENV_DIRS[@]} -eq 0 ]; then
  echo "Error: No virtual environments (venv-*) found in $PROJECT_ROOT."
  echo "Please run 01_create_venv.sh first."
  exit 1
fi

# If multiple venvs, prompt choice with a clear question
if [ ${#VENV_DIRS[@]} -gt 1 ]; then
  echo "Detected multiple Python virtual environments:"
  for i in "${!VENV_DIRS[@]}"; do
    name=$(basename "${VENV_DIRS[i]}")
    printf "  [%d] %s\n" $((i+1)) "$name"
  done

  read -p "Enter the number [1-${#VENV_DIRS[@]}] of the venv to set capabilities for: " idx
  if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 1 || idx > ${#VENV_DIRS[@]} )); then
    echo "Error: Invalid choice. Please enter a number between 1 and ${#VENV_DIRS[@]}."
    exit 1
  fi

  VENV_DIR="${VENV_DIRS[idx-1]}"
else
  VENV_DIR="${VENV_DIRS[0]}"
fi

# Extract the Python version from the venv directory name
PY_VER=$(basename "$VENV_DIR" | sed 's/venv-//')
echo "→ Setting capabilities for: $(basename "$VENV_DIR") ($VENV_DIR)"

# Set paths
PYTHON_BIN="$VENV_DIR/bin/python${PY_VER}"
SYSTEM_PYTHON="/usr/bin/python${PY_VER}"

# Ensure the system Python version exists
if [ ! -f "$SYSTEM_PYTHON" ]; then
    echo "Error: System Python $PY_VER not found at $SYSTEM_PYTHON."
    exit 1
fi

# Replace the Python binary inside the virtual environment
echo "Replacing venv Python with system Python to apply capabilities..."
rm -f "$PYTHON_BIN"
cp "$SYSTEM_PYTHON" "$PYTHON_BIN"

# Assign network interface raw access capabilities
echo "Assigning network interface raw access capabilities..."
sudo setcap cap_net_raw+ep "$PYTHON_BIN"

echo "Network interface raw access capabilities have been assigned to $PYTHON_BIN"
exit 0