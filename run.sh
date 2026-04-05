#!/bin/bash
# Wrapper that activates the venv automatically before running main.py
# Usage: bash run.sh book.pdf [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# Start ollama if not already running
if ! pgrep -x ollama &>/dev/null; then
    echo "[run] Starting Ollama in background..."
    ollama serve &>/dev/null &
    sleep 2
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$SCRIPT_DIR/main.py" "$@"
