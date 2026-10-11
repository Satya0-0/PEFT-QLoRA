"""Two runs, one training script.

    python train.py kaggle_smoke
    python train.py runpod_train

kaggle_smoke is a short T4 pass. It only checks that data, the 4-bit model,
QLoRA, training, evaluation, and adapter saving all work. It is not the experiment.

runpod_train is the experiment: about 20k rows, 2 epochs, and a real eval each epoch.
Edit the dict you are running. Values that should stay identical are named once above the dicts.
"""

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_NAME = "m-a-p/CodeFeedback-Filtered-Instruction"
MAX_SEQ_LENGTH = 2048
SEED = 79
LEARNING_RATE = 2e-4  # stay between 1e-4 and 3e-4
WARMUP_RATIO = 0.03
VAL_FRACTION = 0.05

# LoRA touches every linear projection in each transformer block.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

KAGGLE_SMOKE = {
    "name": "kaggle_smoke",
    "model_name": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "max_seq_length": MAX_SEQ_LENGTH,
    "seed": SEED,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "val_fraction": VAL_FRACTION,
    "target_modules": TARGET_MODULES,
    "lora_r": 16,
    "lora_alpha": 16,
    # A few hundred rows, then stop after 10 optimizer steps.
    # max_steps overrides epochs. eval/save fire on that same step.
    "max_samples": 256,
    "epochs": 1,
    "max_steps": 10,
    "batch_size": 1,
    "grad_accum": 1,
    "eval_strategy": "steps",
    "save_strategy": "steps",
    "every_n_steps": 10,
    "logging_steps": 1,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "output_dir": "outputs/kaggle_smoke",
}

RUNPOD_TRAIN = {
    "name": "runpod_train",
    "model_name": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "max_seq_length": MAX_SEQ_LENGTH,
    "seed": SEED,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "val_fraction": VAL_FRACTION,
    "target_modules": TARGET_MODULES,
    "lora_r": 16,
    "lora_alpha": 16,
    # The real run. max_steps=-1 means "use epochs".
    "max_samples": 20_000,
    "epochs": 2,
    "max_steps": -1,
    "batch_size": 2,
    "grad_accum": 8,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "every_n_steps": None,
    "logging_steps": 10,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "output_dir": "outputs/runpod_train",
}

CONFIGS = {
    "kaggle_smoke": KAGGLE_SMOKE,
    "runpod_train": RUNPOD_TRAIN,
}
