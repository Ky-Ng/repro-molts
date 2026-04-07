#!/usr/bin/env bash
set -euo pipefail

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv and install all dependencies (including dev)
uv sync --extra dev

# Check for HF_TOKEN (needed for Gemma-3-1B experiments)
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set. Gemma-3-1B experiments will fail."
    echo "  export HF_TOKEN=hf_..."
fi

# Check for WANDB_API_KEY (optional, for experiment tracking)
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo ""
    echo "NOTE: WANDB_API_KEY not set. Set it to enable wandb logging."
    echo "  export WANDB_API_KEY=..."
fi

echo ""
echo "Setup complete. Run: uv run pytest tests/"
