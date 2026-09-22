#!/bin/sh
# SmolLM-135M QLoRA finetune (Rust, release).
#
# What it does: the train_smollm example from qlora-model — loads the
# safetensors weights, mounts r=8 LoRA adapters on the Q/V projections of
# all 30 layers, trains only the adapters (frozen fp32 base; wiring up the
# real NF4 + WGPU path is documented under "Scope" in the README), and runs
# causal-LM cross-entropy with paged 8-bit Adam.
# 30 steps, loss 8.07 -> 0.58.
#
# Usage: ./finetune_smollm.sh [steps] [lr]    (defaults: 30 0.005)
#
# Weights: needs models/SmolLM-135M/{model.safetensors,tokenizer.json}.
# If missing and guix is available, downloads them via huggingface_hub
# (~270MB, once).
set -eu

STEPS=${1:-30}
LR=${2:-0.005}

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODEL_DIR="$ROOT/models/SmolLM-135M"

if [ ! -f "$MODEL_DIR/model.safetensors" ] || [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "models/SmolLM-135M weights missing, downloading..."
    if command -v guix >/dev/null 2>&1; then
        guix shell -f "$ROOT/guix.scm" -- python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM-135M', local_dir='$MODEL_DIR',
                  allow_patterns=['*.safetensors', '*.json'])"
    else
        echo "error: guix not found, cannot download automatically." >&2
        echo "please place model.safetensors + tokenizer.json in $MODEL_DIR by hand" >&2
        exit 1
    fi
fi

cd "$ROOT/qlora-model"
exec cargo run --release --offline --example train_smollm -- "$STEPS" "$LR"
