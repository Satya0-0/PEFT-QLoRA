"""Load, dedupe, shuffle, and format CodeFeedback for Qwen2.5-Coder SFT."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def normalize_query(text: str) -> str:
    return " ".join(text.split())


def dedupe_indices(queries: list[Any], answers: list[Any]) -> list[int]:
    """Keep the first row for each whitespace-normalized query. Drop blanks."""
    seen: set[str] = set()
    kept: list[int] = []
    for index, (query, answer) in enumerate(zip(queries, answers)):
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        key = normalize_query(query)
        if not key or not answer.strip() or key in seen:
            continue
        seen.add(key)
        kept.append(index)
    return kept


def allocate_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    """Proportional quotas that sum to sample_size and never exceed class counts."""
    labels = sorted(label for label, count in counts.items() if count > 0)
    total = sum(counts[label] for label in labels)
    if sample_size >= total:
        return {label: counts[label] for label in labels}
    if sample_size <= 0:
        return {label: 0 for label in labels}

    if sample_size < len(labels):
        ordered = sorted(labels, key=lambda label: counts[label], reverse=True)
        quotas = {label: 0 for label in labels}
        for label in ordered[:sample_size]:
            quotas[label] = 1
        return quotas

    floor_each = 2 if sample_size >= 2 * len(labels) else 1
    quotas = {label: min(counts[label], floor_each) for label in labels}
    remaining = sample_size - sum(quotas.values())
    capacity = {label: counts[label] - quotas[label] for label in labels}
    capacity_total = sum(capacity.values())
    if remaining <= 0 or capacity_total == 0:
        return quotas

    raw = {label: remaining * capacity[label] / capacity_total for label in labels}
    for label in labels:
        quotas[label] += min(capacity[label], int(raw[label]))
    leftover = sample_size - sum(quotas.values())
    order = sorted(
        labels,
        key=lambda label: (raw[label] - int(raw[label]), counts[label]),
        reverse=True,
    )
    while leftover > 0:
        progressed = False
        for label in order:
            if quotas[label] < counts[label]:
                quotas[label] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break
    return quotas


def stratified_indices(labels: list[str], sample_size: int) -> list[int]:
    """Take a seed-stable stratified subset by walking labels in their current order."""
    if sample_size <= 0 or sample_size >= len(labels):
        return list(range(len(labels)))
    quotas = allocate_quotas(Counter(labels), sample_size)
    taken: Counter[str] = Counter()
    chosen: list[int] = []
    target = sum(quotas.values())
    for index, label in enumerate(labels):
        if taken[label] < quotas.get(label, 0):
            chosen.append(index)
            taken[label] += 1
            if len(chosen) == target:
                break
    if len(chosen) != target:
        raise RuntimeError(
            f"Stratified subsample filled {len(chosen)} of {target} rows. "
            f"Quotas were {dict(quotas)}."
        )
    return chosen


def split_train_eval(dataset, val_ratio: float, seed: int, column: str):
    counts = Counter(dataset[column])
    rare = sorted(label for label, count in counts.items() if count < 2)
    if rare:
        logger.warning(
            "Column %s has classes with fewer than 2 rows (%s). "
            "Using a seeded random split.",
            column,
            rare,
        )
        return dataset.train_test_split(test_size=val_ratio, seed=seed)

    from datasets import ClassLabel

    names = sorted({str(value) for value in dataset[column]})
    labeled = dataset.map(
        lambda batch: {column: [str(value) for value in batch[column]]},
        batched=True,
    )
    labeled = labeled.cast_column(column, ClassLabel(names=names))
    try:
        return labeled.train_test_split(
            test_size=val_ratio,
            seed=seed,
            stratify_by_column=column,
        )
    except ValueError as exc:
        logger.warning("Stratified split failed (%s). Using a seeded random split.", exc)
        return dataset.train_test_split(test_size=val_ratio, seed=seed)


def format_split(dataset, tokenizer, max_seq_length: int):
    """Render the Qwen chat template and drop rows that would truncate the assistant span."""

    def _render(batch):
        texts = []
        for query, answer in zip(batch["query"], batch["answer"]):
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            if isinstance(rendered, list):
                rendered = rendered[0]
            texts.append(rendered)
        return {"text": texts}

    rendered = dataset.map(_render, batched=True, desc="Apply Qwen chat template")
    before = len(rendered)

    def _fits(batch):
        encoded = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
        )
        return [
            INSTRUCTION_PART in text
            and RESPONSE_PART in text
            and len(token_ids) <= max_seq_length
            for text, token_ids in zip(batch["text"], encoded["input_ids"])
        ]

    filtered = rendered.filter(_fits, batched=True, desc="Drop rows longer than the context")
    extra = [name for name in filtered.column_names if name != "text"]
    if extra:
        filtered = filtered.remove_columns(extra)
    return filtered, before - len(filtered)


def prepare_datasets(cfg, tokenizer):
    from datasets import load_dataset

    raw = load_dataset(cfg.dataset_name, split="train")
    required = {"query", "answer", cfg.stratify_column}
    missing = required.difference(raw.column_names)
    if missing:
        raise RuntimeError(f"Dataset is missing columns: {sorted(missing)}")

    rows_raw = len(raw)
    kept = dedupe_indices(raw["query"], raw["answer"])
    dataset = raw.select(kept)
    if len(dataset) == 0:
        raise RuntimeError("Every row was removed during dedup.")
    logger.info("Deduped %s rows down to %s.", rows_raw, len(dataset))

    dataset = dataset.shuffle(seed=cfg.seed)
    if cfg.max_samples and cfg.max_samples < len(dataset):
        chosen = stratified_indices(list(dataset[cfg.stratify_column]), cfg.max_samples)
        dataset = dataset.select(chosen)
        logger.info("Stratified subsample kept %s rows.", len(dataset))
    rows_sampled = len(dataset)

    parts = split_train_eval(dataset, cfg.val_ratio, cfg.seed, cfg.stratify_column)
    train_dataset, train_dropped = format_split(parts["train"], tokenizer, cfg.max_seq_length)
    eval_dataset, eval_dropped = format_split(parts["test"], tokenizer, cfg.max_seq_length)
    if len(train_dataset) == 0 or len(eval_dataset) == 0:
        raise RuntimeError(
            "Chat-template formatting left an empty split "
            f"(train={len(train_dataset)}, eval={len(eval_dataset)}). "
            "Raise max_samples or max_seq_length."
        )
    logger.info(
        "Formatted train=%s (dropped %s long) eval=%s (dropped %s long).",
        len(train_dataset),
        train_dropped,
        len(eval_dataset),
        eval_dropped,
    )
    stats = {
        "rows_raw": rows_raw,
        "rows_after_dedup": len(kept),
        "rows_after_subsample": rows_sampled,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset),
        "dropped_long_train": train_dropped,
        "dropped_long_eval": eval_dropped,
    }
    return train_dataset, eval_dataset, stats
