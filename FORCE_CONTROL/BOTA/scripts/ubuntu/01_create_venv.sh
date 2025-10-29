#!/usr/bin/env bash
set -e

# --- CONFIG ---
SUPPORTED_VERSIONS=("3.8" "3.9" "3.10" "3.11" "3.12" "3.13")
SUPPORTED_VERSIONS_ABI3=("3.12" "3.13")
# --------------

# Build a list of versions that really run
AVAILABLE_VERSIONS=()
for v in "${SUPPORTED_VERSIONS[@]}"; do
  if command -v "python${v}" &>/dev/null && python${v} --version &>/dev/null; then
    AVAILABLE_VERSIONS+=("$v")
  fi
done

if [ ${#AVAILABLE_VERSIONS[@]} -eq 0 ]; then
  echo "Error: No supported Python interpreters found on your PATH."
  exit 1
fi

echo "Available Python versions installed in your system:"
for v in "${AVAILABLE_VERSIONS[@]}"; do
  echo " - $v"
done

read -p "Enter one of the above Python versions to use: " PY_VER

# Make sure they picked an actually available one
if [[ ! " ${AVAILABLE_VERSIONS[*]} " =~ " ${PY_VER} " ]]; then
  echo "Error: Python ${PY_VER} is not available on this system."
  exit 1
fi

# Locate project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
VENV_DIR="$PROJECT_ROOT/venv-${PY_VER}"

# Use the chosen interpreter directly
PY_BIN="python${PY_VER}"

# Re-create the venv
rm -rf "$VENV_DIR"
"$PY_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# Install the package from PyPI
echo "Installing bota-driver from PyPI"
pip install bota-driver

echo "Installed successfully into $VENV_DIR"
exit 0
