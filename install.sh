#!/bin/bash
set -e

echo "=== pdf_to_video installer ==="
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
OLLAMA_MODEL="llama3.2:3b"

# Pass --clean to wipe the venv and start fresh
if [ "${1}" = "--clean" ]; then
    echo "[clean] Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

# ── helpers ───────────────────────────────────────────────────────────────────

pkg_version() {
    python -c "import importlib.metadata; print(importlib.metadata.version('$1'))" 2>/dev/null || echo ""
}

ollama_model_exists() {
    ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

# ── 0. Virtual environment ────────────────────────────────────────────────────
echo "[0/4] Virtual environment"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv..."
    python3 -m venv "$VENV_DIR"
else
    echo "  Already exists — reusing."
fi

source "$VENV_DIR/bin/activate"
echo "  Active: $(which python)  ($(python --version))"

# ── 1. PyTorch + torchvision + xformers (all at once, same index) ─────────────
# Installing together lets pip resolve xformers compatibility automatically.
# cu124 is required for torch >= 2.6; your CUDA 13.0 driver supports it.
echo
echo "[1/4] PyTorch 2.6 + torchvision + xformers"

TORCH_OK=false
if [ "$(pkg_version torch)" = "2.6.0+cu124" ] && \
   [ "$(pkg_version torchvision)" = "0.21.0+cu124" ] && \
   [ -n "$(pkg_version xformers)" ]; then
    echo "  All already installed — skipping."
    TORCH_OK=true
fi

if [ "$TORCH_OK" = false ]; then
    echo "  Installing (this may take a few minutes)..."
    pip install --quiet \
        "torch==2.6.0+cu124" \
        "torchvision==0.21.0+cu124" \
        "xformers" \
        --index-url https://download.pytorch.org/whl/cu124
    echo "  Done."
fi

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo
echo "[2/4] Python dependencies"
# Filter out torch/torchvision/xformers — handled above with custom index
grep -vE '^\s*(torch|#|$)' "$SCRIPT_DIR/requirements.txt" > /tmp/req_filtered.txt
pip install --quiet -r /tmp/req_filtered.txt
echo "  Done."

# ── 3. Ollama ─────────────────────────────────────────────────────────────────
echo
echo "[3/4] Ollama"
if command -v ollama &>/dev/null; then
    echo "  Already installed: $(ollama --version)"
else
    echo "  Downloading and installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo
echo "  Model: $OLLAMA_MODEL"
OLLAMA_STARTED=false
if ! pgrep -x ollama &>/dev/null; then
    ollama serve &>/dev/null &
    OLLAMA_STARTED=true
    sleep 2
fi

if ollama_model_exists "$OLLAMA_MODEL"; then
    echo "  Already downloaded — skipping."
else
    echo "  Pulling $OLLAMA_MODEL (~2 GB)..."
    ollama pull "$OLLAMA_MODEL"
fi

if [ "$OLLAMA_STARTED" = true ]; then
    pkill -x ollama 2>/dev/null || true
fi

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo
echo "[4/4] Verifying installation"
python - <<'EOF'
import torch

# torch + CUDA
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"  torch {torch.__version__}  |  CUDA OK — {name} ({vram} GB VRAM)")
else:
    print(f"  torch {torch.__version__}  |  WARNING: CUDA not available")

# xformers (optional)
try:
    import xformers
    print(f"  xformers {xformers.__version__}  |  OK")
except ImportError:
    print("  xformers  |  not installed (optional, will still work)")

# core dependencies
for pkg in ["diffusers", "transformers", "pdfplumber", "kokoro", "moviepy", "ollama"]:
    try:
        import importlib.metadata
        v = importlib.metadata.version(pkg)
        print(f"  {pkg} {v}  |  OK")
    except Exception:
        print(f"  {pkg}  |  MISSING")
EOF

echo
echo "=== Installation complete ==="
echo
echo "The image model (Lykon/dreamshaper-8, ~2 GB) downloads on first run."
echo
echo "Usage:"
echo "  bash $SCRIPT_DIR/run.sh book.pdf"
echo "  bash $SCRIPT_DIR/run.sh book.pdf --pages 1-10"
