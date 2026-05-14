"""Data loading and preprocessing for the judge-distillation experiment.

Loads the judge-distillation JSONL, splits into train/val/test, and tokenizes
(prompt, response) pairs into a simple USER/ASSISTANT format.

Two split modes are supported:

- **leave_dose_out** (default, audit scenario):
    test  = every record at dose == holdout_dose (400 rows)
    val   = stratified val_fraction of the remaining doses, balanced over (category, dose)
    train = the rest
  Tests generalization to a poisoning level the judge never saw at training time.
  Prompt IDs intentionally leak across splits (the holdout is the dose, not the prompt).

- **stratified_prompt** (prompt generalization):
    Pick test_fraction × prompt_ids per category for test (all 6 doses each).
    Pick another val_fraction × prompt_ids per category for val.
    Train = the remaining prompts × all 6 doses.
  Tests generalization to unseen prompts. All doses appear in all splits.

We use a manual format string rather than tokenizer.apply_chat_template because
the base `google/gemma-2-2b` tokenizer does not ship a chat template (only the
-it variant does). For a regression head on pooled embeddings the exact format
is unimportant as long as it is consistent.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


DEFAULT_HOLDOUT_DOSE = 25
ALL_DOSES = (0, 5, 10, 25, 50, 100)
DEFAULT_MAX_LENGTH = 512
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10  # only used in stratified_prompt mode

SPLIT_MODES = ("leave_dose_out", "stratified_prompt")

# Backwards-compat aliases (used by callers that imported these directly).
HOLDOUT_DOSE = DEFAULT_HOLDOUT_DOSE
TRAIN_DOSES = tuple(d for d in ALL_DOSES if d != DEFAULT_HOLDOUT_DOSE)


def load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_leave_dose_out(
    records: list[dict[str, Any]],
    holdout_dose: int,
    val_fraction: float,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    train_doses = tuple(d for d in ALL_DOSES if d != holdout_dose)
    test = [r for r in records if r["dose"] == holdout_dose]
    pool = [r for r in records if r["dose"] in train_doses]

    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in pool:
        by_cell[(r["category"], r["dose"])].append(r)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for cell_records in by_cell.values():
        rng.shuffle(cell_records)
        n_val = max(1, int(round(len(cell_records) * val_fraction)))
        val.extend(cell_records[:n_val])
        train.extend(cell_records[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val, "test": test}


def _split_stratified_prompt(
    records: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Hold out prompt IDs (stratified by category). All 6 doses of each held-out
    prompt go to the same split — prevents within-prompt leakage."""
    by_cat_prompts: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_cat_prompts[r["category"]].add(r["prompt_id"])

    test_ids: set[str] = set()
    val_ids: set[str] = set()
    for cat, pids in by_cat_prompts.items():
        pid_list = sorted(pids)
        rng.shuffle(pid_list)
        n_test = max(1, int(round(len(pid_list) * test_fraction)))
        n_val = max(1, int(round(len(pid_list) * val_fraction)))
        test_ids.update(pid_list[:n_test])
        val_ids.update(pid_list[n_test : n_test + n_val])

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for r in records:
        if r["prompt_id"] in test_ids:
            test.append(r)
        elif r["prompt_id"] in val_ids:
            val.append(r)
        else:
            train.append(r)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def split_records(
    records: list[dict[str, Any]],
    split_mode: str = "leave_dose_out",
    holdout_dose: int = DEFAULT_HOLDOUT_DOSE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split_mode {split_mode!r}; must be one of {SPLIT_MODES}")
    rng = random.Random(seed)
    if split_mode == "leave_dose_out":
        return _split_leave_dose_out(records, holdout_dose, val_fraction, rng)
    if split_mode == "stratified_prompt":
        return _split_stratified_prompt(records, val_fraction, test_fraction, rng)
    raise AssertionError("unreachable")


def format_prompt_response(prompt: str, response: str) -> str:
    """Simple consistent format. Gemma tokenizer prepends <bos> automatically."""
    return f"USER: {prompt}\n\nASSISTANT: {response}"


def build_hf_datasets(
    records_by_split: dict[str, list[dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
    label_field: str = "drift_pct",
) -> DatasetDict:
    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        texts = [
            format_prompt_response(p, r)
            for p, r in zip(batch["prompt"], batch["response"])
        ]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = [float(d) for d in batch["_label"]]
        return enc

    out: dict[str, Dataset] = {}
    for split, recs in records_by_split.items():
        # Project to only the fields needed for tokenization + labels. The full
        # records (including potentially-INVALID claude_score strings that break
        # PyArrow schema inference) are still available via the raw splits dict
        # for holdout-eval baselines.
        slim = [
            {"prompt": r["prompt"], "response": r["response"],
             "_label": float(r[label_field])}
            for r in recs
            if isinstance(r.get(label_field), (int, float))
        ]
        ds = Dataset.from_list(slim)
        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        out[split] = ds
    return DatasetDict(out)


def load_and_prepare(
    jsonl_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    split_mode: str = "leave_dose_out",
    holdout_dose: int = DEFAULT_HOLDOUT_DOSE,
    seed: int = 42,
    smoke: bool = False,
    label_field: str = "drift_pct",
) -> tuple[DatasetDict, dict[str, list[dict[str, Any]]]]:
    """Load JSONL, split, tokenize. Returns (HF DatasetDict, raw records by split)."""
    records = load_records(jsonl_path)

    if smoke:
        # Keep dose balance so the test split (in leave_dose_out mode) is non-empty.
        rng = random.Random(seed)
        by_dose: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_dose[r["dose"]].append(r)
        records = []
        for recs in by_dose.values():
            rng.shuffle(recs)
            records.extend(recs[:20])
        rng.shuffle(records)

    splits = split_records(
        records,
        split_mode=split_mode,
        holdout_dose=holdout_dose,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    hf = build_hf_datasets(splits, tokenizer, max_length=max_length, label_field=label_field)
    return hf, splits
