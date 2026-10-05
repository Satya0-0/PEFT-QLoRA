#!/usr/bin/env bash
# Install the pinned HF stack without replacing a CUDA-matched torch that
# already satisfies Unsloth (torch>=2.4.0,<2.13.0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install --upgrade-strategy only-if-needed -r "$ROOT/requirements.txt"

"$PY" - <<'PY'
import torch
import transformers
import trl
import bitsandbytes

cuda = torch.version.cuda
device = "cpu"
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    device = f"{torch.cuda.get_device_name(0)} cc {major}.{minor}"
print(f"torch {torch.__version__} cuda {cuda} device {device}")
print(f"transformers {transformers.__version__}")
print(f"trl {trl.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
PY

echo "Kaggle smoke: PYTHONPATH=src ${PY} -m qlora_sft.train --profile kaggle"
echo "RunPod full:  PYTHONPATH=src ${PY} -m qlora_sft.train --profile runpod"
