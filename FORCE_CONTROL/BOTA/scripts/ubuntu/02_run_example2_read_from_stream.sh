#!/usr/bin/env bash
set -e

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

# Source the helper script to activate a virtual environment
source "$SCRIPT_DIR/source_venv_helper.sh"

# Find the python executable
mapfile -t PYS < <(ls "$VENV_DIR/bin/python"* 2>/dev/null)
if [ ${#PYS[@]} -eq 0 ]; then
  echo "Error: No Python executable found in $VENV_DIR/bin/"
  exit 1
fi
PYTHON_EXEC="${PYS[0]}"

# Run the example script
"$PYTHON_EXEC" "$PROJECT_ROOT/examples/example2_read_from_stream.py"