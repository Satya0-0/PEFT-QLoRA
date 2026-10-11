# Qwen2.5-Coder-3B QLoRA Fine-Tuning

Code and configuration for fine-tuning `Qwen/Qwen2.5-Coder-3B-Instruct` on the `m-a-p/CodeFeedback-Filtered-Instruction` dataset using 4-bit QLoRA with Unsloth and Hugging Face TRL.

The setup supports two execution modes:
1. **`kaggle_smoke`**: A quick run (10 steps) on Kaggle T4 GPUs to verify memory limits, dataset filtering, and loss masking before spending money on cloud GPUs.
2. **`runpod_train`**: The full 2-epoch training run over 20,000 samples on RunPod GPUs.

---

## Artifacts

* **Model Weights (Hugging Face Hub):** [qwen2.5-coder-3b-sft-codefeedback](https://huggingface.co/Satya0-0/qwen2.5-coder-3b-sft-codefeedback/tree/main)

---

## Technical Details

* **Base Model:** `Qwen/Qwen2.5-Coder-3B-Instruct`
* **Dataset:** `m-a-p/CodeFeedback-Filtered-Instruction`
* **Method:** 4-bit QLoRA (`bitsandbytes`) targeting all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Loss Masking:** Uses Unsloth's `train_on_responses_only` to compute cross-entropy loss exclusively on assistant code completions (ignoring prompt tokens).
* **Data Processing:** Drops empty entries, strips duplicate input queries, applies Qwen's chat template, and filters out sequence lengths greater than 2048 tokens.
* **Experiment Tracking:** Metrics automatically logged to Weights & Biases (W&B).

---

## Repository Structure

```text
.
├── config.py          # Central hyperparameter configuration profiles
├── train.py           # Dataset loading, formatting, loss masking, and training loop
├── upload_model.py    # Merges LoRA adapters into base weights and uploads to Hugging Face
└── requirements.txt   # Dependencies
```

---

## Configuration Summary

| Setting | `kaggle_smoke` (Test Run) | `runpod_train` (Full Run) |
|---|---|---|
| **Max Samples** | 256 | 20,000 |
| **Duration** | 10 steps | 2 epochs |
| **Batch Size** | 1 per device | 2 per device |
| **Grad Accumulation** | 1 | 8 (Effective batch size = 16) |
| **Learning Rate** | `2e-4` (Cosine schedule) | `2e-4` (Cosine schedule) |
| **Optimizer** | `paged_adamw_8bit` | `paged_adamw_8bit` |
| **LoRA Rank ($r$) / Alpha ($\alpha$)** | 16 / 16 | 16 / 16 |
| **Eval & Save Strategy** | Every 10 steps | Every epoch |

---

## How to Run

### 1. Setup

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install -r requirements.txt
```

Set up W&B tracking:
```bash
wandb login
```

### 2. Run Training

**Test run (Kaggle T4 or local GPU):**
```bash
python train.py kaggle_smoke
```

**Full run (RunPod):**
```bash
python train.py runpod_train
```

### 3. Merge & Export

To merge the LoRA adapter with the base model and upload to Hugging Face:

```bash
huggingface-cli login
python upload_model.py
```

---

## Development & Tooling

* Built with **Unsloth**, **Hugging Face TRL / Transformers**, and **PyTorch**.
* Developed using **Cursor IDE** for AI-assisted script generation and environment setup.