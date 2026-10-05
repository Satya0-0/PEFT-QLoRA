"""QLoRA SFT entry point.

Kaggle smoke check:
    PYTHONPATH=src python -m qlora_sft.train --profile kaggle

RunPod full run:
    PYTHONPATH=src python -m qlora_sft.train --profile runpod

Import unsloth before trl, transformers, and peft. That happens inside main,
after argument parsing, so --help does not need a GPU stack.
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import asdict
from pathlib import Path

from qlora_sft.config import TrainConfig, parse_args, validate_environment
from qlora_sft.data import INSTRUCTION_PART, RESPONSE_PART, prepare_datasets

logger = logging.getLogger(__name__)


def load_model(cfg: TrainConfig, fast_language_model):
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    model = fast_language_model.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def _filter_kwargs(callable_, kwargs: dict) -> dict:
    parameters = inspect.signature(callable_).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def build_sft_config(cfg: TrainConfig, bf16: bool, sft_config_cls):
    kwargs = {
        "output_dir": str(Path(cfg.output_dir) / "checkpoints"),
        "num_train_epochs": cfg.num_train_epochs,
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "optim": cfg.optim,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": cfg.save_total_limit,
        "logging_steps": cfg.logging_steps,
        "bf16": bf16,
        "fp16": not bf16,
        "packing": False,
        "dataset_text_field": "text",
        "max_length": cfg.max_seq_length,
        "max_seq_length": cfg.max_seq_length,
        "report_to": "wandb",
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "remove_unused_columns": False,
    }
    parameters = inspect.signature(sft_config_cls.__init__).parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return sft_config_cls(**_filter_kwargs(sft_config_cls.__init__, kwargs))


def build_trainer(cfg, model, tokenizer, train_dataset, eval_dataset, bf16, sft_config_cls, sft_trainer_cls):
    trainer_kwargs = {
        "model": model,
        "args": build_sft_config(cfg, bf16, sft_config_cls),
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    parameters = inspect.signature(sft_trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        trainer_kwargs["processing_class"] = tokenizer
    if "tokenizer" in parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" not in trainer_kwargs and "tokenizer" not in trainer_kwargs:
        raise RuntimeError("SFTTrainer accepts neither processing_class nor tokenizer.")
    trainer = sft_trainer_cls(**trainer_kwargs)
    if not hasattr(trainer, "tokenizer"):
        trainer.tokenizer = tokenizer
    return trainer


def assert_response_mask(trainer, tokenizer) -> None:
    """Abort when the first train row has no assistant tokens left for the loss."""
    if len(trainer.train_dataset) == 0 or _dataset_length(trainer.eval_dataset) == 0:
        raise RuntimeError(
            "Response masking removed every train or eval row. "
            f"Expected markers {INSTRUCTION_PART!r} and {RESPONSE_PART!r}."
        )
    row = trainer.train_dataset[0]
    labels = _as_list(row["labels"])
    input_ids = _as_list(row["input_ids"])
    supervised = [token for token, label in zip(input_ids, labels) if label != -100]
    if not supervised:
        raise RuntimeError(
            "All labels on the first training row are -100. "
            f"The assistant marker {RESPONSE_PART!r} was not found in the chat template."
        )
    if all(label != -100 for label in labels):
        raise RuntimeError(
            "No prompt tokens were masked, so the loss would include the user turn."
        )
    preview = tokenizer.decode(supervised, skip_special_tokens=False)
    logger.info("Response-only labels are set. Preview: %s", preview[:400])


def _dataset_length(dataset) -> int:
    if isinstance(dataset, dict):
        return sum(len(split) for split in dataset.values())
    return len(dataset)


def _as_list(values):
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def require_wandb_credentials(cfg: TrainConfig) -> None:
    if cfg.wandb_mode != "offline" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is not set. Export it, or pass --wandb-mode offline."
        )


def init_wandb(cfg: TrainConfig, stats: dict, bf16: bool):
    import wandb

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    require_wandb_credentials(cfg)
    config = asdict(cfg)
    config["target_modules"] = list(cfg.target_modules)
    config.update(stats)
    config["bf16"] = bf16
    config["fp16"] = not bf16
    return wandb.init(
        project=cfg.wandb_project,
        name=f"{cfg.profile}-qwen2.5-coder-3b-sft",
        config=config,
        mode=cfg.wandb_mode,
    )


def save_adapter(cfg: TrainConfig, model, tokenizer) -> None:
    adapter_dir = Path(cfg.output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved LoRA adapter to %s", adapter_dir)
    if cfg.merge_16bit:
        merged_dir = Path(cfg.output_dir) / "merged_16bit"
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("Saved merged 16-bit weights to %s", merged_dir)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import unsloth  # noqa: F401  -- must load before trl, transformers, and peft
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from trl import SFTConfig, SFTTrainer

    validate_environment()
    require_wandb_credentials(cfg)
    major, minor = torch.cuda.get_device_capability()
    bf16 = bool(is_bfloat16_supported())
    logger.info(
        "Profile %s | GPU %s capability %s.%s | %s | lr %s | epochs %s | max_samples %s",
        cfg.profile,
        torch.cuda.get_device_name(0),
        major,
        minor,
        "bf16" if bf16 else "fp16",
        cfg.learning_rate,
        cfg.num_train_epochs,
        cfg.max_samples,
    )

    model, tokenizer = load_model(cfg, FastLanguageModel)
    train_dataset, eval_dataset, stats = prepare_datasets(cfg, tokenizer)
    run = None
    try:
        run = init_wandb(cfg, stats, bf16)
        trainer = build_trainer(
            cfg,
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            bf16,
            SFTConfig,
            SFTTrainer,
        )
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )
        assert_response_mask(trainer, tokenizer)
        trainer.train()
        trained = trainer.model
        if hasattr(trained, "module"):
            trained = trained.module
        save_adapter(cfg, trained, tokenizer)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
