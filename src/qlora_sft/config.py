"""Profiles and the runtime version guard for the QLoRA SFT run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from packaging.version import Version

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_NAME = "m-a-p/CodeFeedback-Filtered-Instruction"
SEED = 42
LR_MIN = 1e-4
LR_MAX = 3e-4

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Versions Unsloth 2026.9.6 rejects even when they fall inside the broad range.
EXCLUDED_TRANSFORMERS = frozenset(
    {
        "4.52.0",
        "4.52.1",
        "4.52.2",
        "4.52.3",
        "4.53.0",
        "4.54.0",
        "4.55.0",
        "4.55.1",
        "4.57.0",
        "4.57.4",
        "4.57.5",
        "5.0.0",
        "5.1.0",
    }
)
EXCLUDED_BITSANDBYTES = frozenset({"0.46.0", "0.48.0"})


@dataclass(frozen=True)
class TrainConfig:
    profile: str
    model_name: str = MODEL_NAME
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = TARGET_MODULES
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    seed: int = SEED
    dataset_name: str = DATASET_NAME
    stratify_column: str = "resource"
    val_ratio: float = 0.05
    max_samples: int = 512
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    save_total_limit: int = 1
    logging_steps: int = 10
    output_dir: str = "outputs/kaggle"
    wandb_project: str = "qwen-coder-qlora"
    wandb_mode: str = "online"
    merge_16bit: bool = False


def profile_config(name: str) -> TrainConfig:
    if name == "kaggle":
        return TrainConfig(profile="kaggle")
    if name == "runpod":
        return TrainConfig(
            profile="runpod",
            max_samples=20_000,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            save_total_limit=2,
            output_dir="outputs/runpod",
        )
    raise ValueError(f"Unknown profile {name!r}. Choose kaggle or runpod.")


def validate_learning_rate(learning_rate: float) -> None:
    if learning_rate < LR_MIN - 1e-12 or learning_rate > LR_MAX + 1e-12:
        raise ValueError(
            f"learning_rate must be between {LR_MIN:g} and {LR_MAX:g}, got {learning_rate:g}."
        )


def _version(value: str) -> Version:
    return Version(value.split("+")[0])


def check_versions(versions: dict[str, str]) -> list[str]:
    """Return human-readable errors for a stack outside the Unsloth 2026.9.6 window."""
    errors: list[str] = []

    def require(name: str) -> Version | None:
        raw = versions.get(name)
        if not raw:
            errors.append(f"{name} is not installed.")
            return None
        return _version(raw)

    torch_version = require("torch")
    if torch_version is not None and not (
        Version("2.4.0") <= torch_version < Version("2.13.0")
    ):
        errors.append(f"torch {versions['torch']} is outside >=2.4.0,<2.13.0.")

    transformers_version = require("transformers")
    if transformers_version is not None and (
        transformers_version.base_version in EXCLUDED_TRANSFORMERS
        or not (Version("4.51.3") <= transformers_version <= Version("5.5.0"))
    ):
        errors.append(
            f"transformers {versions['transformers']} is outside the Unsloth-compatible range."
        )

    trl_version = require("trl")
    if trl_version is not None and (
        trl_version.base_version == "0.19.0"
        or not (Version("0.18.2") <= trl_version <= Version("0.24.0"))
    ):
        errors.append(f"trl {versions['trl']} is outside >=0.18.2,<=0.24.0 (excluding 0.19.0).")

    bnb_version = require("bitsandbytes")
    if bnb_version is not None and (
        bnb_version < Version("0.45.5") or bnb_version.base_version in EXCLUDED_BITSANDBYTES
    ):
        errors.append(
            f"bitsandbytes {versions['bitsandbytes']} is outside >=0.45.5 "
            "(excluding 0.46.0 and 0.48.0)."
        )

    peft_version = require("peft")
    if peft_version is not None and (
        peft_version < Version("0.18.0") or peft_version.base_version == "0.11.0"
    ):
        errors.append(f"peft {versions['peft']} is outside >=0.18.0.")

    datasets_version = require("datasets")
    if datasets_version is not None and (
        not (Version("3.4.1") <= datasets_version < Version("4.4.0"))
        or datasets_version.base_version == "4.1.0"
        or Version("4.0.0") <= datasets_version < Version("4.1.0")
    ):
        errors.append(
            f"datasets {versions['datasets']} is outside >=3.4.1,<4.4.0 "
            "(excluding 4.0.* and 4.1.0)."
        )

    accelerate_version = require("accelerate")
    if accelerate_version is not None and accelerate_version < Version("0.34.1"):
        errors.append(f"accelerate {versions['accelerate']} is outside >=0.34.1.")

    unsloth_version = require("unsloth")
    if unsloth_version is not None and unsloth_version.base_version != "2026.9.6":
        errors.append(f"unsloth {versions['unsloth']} is not 2026.9.6.")

    return errors


def _module_version(module, dist_name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    raw = getattr(module, "__version__", None)
    if isinstance(raw, str) and raw:
        return raw
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return ""


def validate_environment() -> None:
    """Refuse to train when the import stack or GPU cannot run 4-bit QLoRA."""
    import bitsandbytes
    import datasets
    import torch
    import transformers
    import trl
    import unsloth
    import accelerate
    import peft

    installed = {
        "torch": _module_version(torch, "torch"),
        "transformers": _module_version(transformers, "transformers"),
        "trl": _module_version(trl, "trl"),
        "bitsandbytes": _module_version(bitsandbytes, "bitsandbytes"),
        "peft": _module_version(peft, "peft"),
        "datasets": _module_version(datasets, "datasets"),
        "accelerate": _module_version(accelerate, "accelerate"),
        "unsloth": _module_version(unsloth, "unsloth"),
    }
    errors = check_versions(installed)
    if not torch.cuda.is_available():
        errors.append("CUDA is not available. QLoRA needs a GPU.")
    else:
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (7, 0):
            errors.append(
                f"GPU compute capability {major}.{minor} is below 7.0. "
                "4-bit bitsandbytes needs a T4 or newer."
            )
    if errors:
        raise RuntimeError("Refusing to train:\n- " + "\n- ".join(errors))


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT for Qwen2.5-Coder-3B-Instruct (Kaggle smoke or RunPod full run)."
    )
    parser.add_argument("--profile", choices=("kaggle", "runpod"), required=True)
    parser.add_argument("--max-samples", type=int, default=None, help="0 keeps every row after dedup.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--merge-16bit", action="store_true")
    args = parser.parse_args(argv)

    cfg = profile_config(args.profile)
    updates: dict[str, object] = {}
    if args.max_samples is not None:
        updates["max_samples"] = args.max_samples
    if args.learning_rate is not None:
        updates["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        updates["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        updates["per_device_train_batch_size"] = args.per_device_train_batch_size
        updates["per_device_eval_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        updates["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.model_name is not None:
        updates["model_name"] = args.model_name
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.wandb_project is not None:
        updates["wandb_project"] = args.wandb_project
    if args.wandb_mode is not None:
        updates["wandb_mode"] = args.wandb_mode
    if args.merge_16bit:
        updates["merge_16bit"] = True

    cfg = replace(cfg, **updates)
    try:
        validate_learning_rate(cfg.learning_rate)
    except ValueError as exc:
        parser.error(str(exc))
    if cfg.max_samples < 0:
        parser.error("--max-samples must be >= 0 (0 keeps every row after dedup).")
    if cfg.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be positive.")
    if cfg.per_device_train_batch_size < 1 or cfg.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be >= 1.")
    return cfg
