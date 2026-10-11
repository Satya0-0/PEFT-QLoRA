"""Supervised fine-tune Qwen2.5-Coder-3B with QLoRA.

    pip install -r requirements.txt
    python train.py kaggle_smoke
    python train.py runpod_train

Unsloth is imported first. It patches the Hugging Face training stack, and that
patch is skipped if transformers or trl load before it.

QLoRA keeps the base model in 4-bit and trains a small LoRA adapter on top.
The loss is applied only to the assistant's reply, not to the user's prompt.
"""

import sys

import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

import torch
from datasets import ClassLabel, load_dataset
from trl import SFTConfig, SFTTrainer

from config import CONFIGS

# These strings are how the Qwen2.5 chat template marks a turn.
# train_on_responses_only searches for them and ignores everything else.
USER_TURN = "<|im_start|>user\n"
ASSISTANT_TURN = "<|im_start|>assistant\n"


def dedupe(dataset):
    """Drop blank rows and repeated queries. One query must not sit in both splits."""
    seen = set()
    keep = []
    for index, (query, answer) in enumerate(zip(dataset["query"], dataset["answer"])):
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        key = " ".join(query.split())
        if not key or not answer.strip() or key in seen:
            continue
        seen.add(key)
        keep.append(index)
    print(f"dedup {len(dataset)} -> {len(keep)}")
    return dataset.select(keep)


def format_for_qwen(dataset, tokenizer, max_seq_length):
    """Add a `text` column in Qwen chat format, and drop rows that do not fit.

    `resource` is kept so the train/eval split can still stratify on it.
    Rows longer than max_seq_length are dropped because truncation would cut off
    the assistant reply, and the loss mask would then match nothing.
    """

    def render(batch):
        texts = []
        for query, answer in zip(batch["query"], batch["answer"]):
            texts.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    dataset = dataset.map(render, batched=True)

    def fits(batch):
        token_rows = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        return [
            USER_TURN in text and ASSISTANT_TURN in text and len(ids) <= max_seq_length
            for text, ids in zip(batch["text"], token_rows)
        ]

    before = len(dataset)
    dataset = dataset.filter(fits, batched=True)
    print(f"chat template kept {len(dataset)} of {before} (the rest were longer than {max_seq_length})")
    return dataset


def train_eval_split(dataset, val_fraction, seed):
    """Hold out `val_fraction` of the rows, stratified by `resource` when possible."""
    names = sorted({str(value) for value in dataset["resource"]})
    labeled = dataset.cast_column("resource", ClassLabel(names=names))
    try:
        parts = labeled.train_test_split(
            test_size=val_fraction,
            seed=seed,
            stratify_by_column="resource",
        )
    except ValueError as exc:
        print(f"stratified split unavailable ({exc}); using a plain seeded split")
        parts = dataset.train_test_split(test_size=val_fraction, seed=seed)
    # The trainer only reads `text`.
    return (
        parts["train"].remove_columns([c for c in parts["train"].column_names if c != "text"]),
        parts["test"].remove_columns([c for c in parts["test"].column_names if c != "text"]),
    )


def prepare_data(cfg, tokenizer):
    dataset = load_dataset(cfg["dataset_name"], split="train")
    dataset = dedupe(dataset)
    # Seeded shuffle, then the first max_samples rows. Same seed, same sample.
    dataset = dataset.shuffle(seed=cfg["seed"])
    if len(dataset) > cfg["max_samples"]:
        dataset = dataset.select(range(cfg["max_samples"]))
    dataset = format_for_qwen(dataset, tokenizer, cfg["max_seq_length"])
    if len(dataset) < 4:
        raise RuntimeError(
            f"Only {len(dataset)} rows fit in {cfg['max_seq_length']} tokens. Raise max_samples."
        )
    train_data, eval_data = train_eval_split(dataset, cfg["val_fraction"], cfg["seed"])
    if len(train_data) == 0 or len(eval_data) == 0:
        raise RuntimeError("The train/eval split is empty. Raise max_samples.")
    print(f"train rows {len(train_data)}  eval rows {len(eval_data)}")
    return train_data, eval_data


def load_qlora(cfg):
    """4-bit base model plus a LoRA adapter. Only the adapter receives gradients."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0,
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.print_trainable_parameters()
    return model, tokenizer


def build_trainer(cfg, model, tokenizer, train_data, eval_data):
    # T4 has no bf16. Newer GPUs (A100, H100, and similar) do.
    bf16 = bool(is_bfloat16_supported())
    print("precision", "bf16" if bf16 else "fp16")
    args = dict(
        output_dir=cfg["output_dir"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        num_train_epochs=cfg["epochs"],
        max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps"],
        bf16=bf16,
        fp16=not bf16,
        packing=False,
        dataset_text_field="text",
        max_length=cfg["max_seq_length"],
        seed=cfg["seed"],
        report_to="wandb",
    )
    if cfg["eval_strategy"] == "steps":
        args["eval_steps"] = cfg["every_n_steps"]
        args["save_steps"] = cfg["every_n_steps"]
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(**args),
    )


def check_loss_mask(trainer, tokenizer):
    """Print the tokens that still have a loss. Prompt tokens should be gone."""
    row = trainer.train_dataset[0]
    labels = list(row["labels"])
    input_ids = list(row["input_ids"])
    kept = [token for token, label in zip(input_ids, labels) if label != -100]
    if not kept or all(label != -100 for label in labels):
        raise RuntimeError(
            "The loss mask is wrong. Expected assistant tokens to be trained "
            f"and prompt tokens to be ignored. Markers: {USER_TURN!r} {ASSISTANT_TURN!r}"
        )
    preview = tokenizer.decode(kept, skip_special_tokens=False)
    print("loss is computed only on:", preview[:400].replace("\n", " "))


def save_adapter(cfg, model, tokenizer):
    adapter_dir = f"{cfg['output_dir']}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("saved adapter to", adapter_dir)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        names = ", ".join(CONFIGS)
        raise SystemExit(f"Usage: python train.py <{names}>")
    cfg = CONFIGS[sys.argv[1]]
    print("run", cfg["name"])

    if not torch.cuda.is_available():
        raise SystemExit("This script needs a CUDA GPU.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 0):
        raise SystemExit(
            f"GPU compute capability is {major}.{minor}. "
            "4-bit QLoRA needs 7.0 or newer (a T4, not a P100)."
        )
    print("gpu", torch.cuda.get_device_name(0), f"cc {major}.{minor}")

    model, tokenizer = load_qlora(cfg)
    train_data, eval_data = prepare_data(cfg, tokenizer)
    trainer = build_trainer(cfg, model, tokenizer, train_data, eval_data)
    trainer = train_on_responses_only(
        trainer,
        instruction_part=USER_TURN,
        response_part=ASSISTANT_TURN,
    )
    check_loss_mask(trainer, tokenizer)
    trainer.train()
    save_adapter(cfg, trainer.model, tokenizer)


if __name__ == "__main__":
    main()
