"""
Dataset loading and preprocessing for fake news classification.

Supported datasets:
  - LIAR  (https://www.cs.ucsb.edu/~william/papers/liar_dataset.pdf)
  - FakeNewsNet / PolitiFact / GossipCop

Output: HuggingFace DatasetDict with 'train', 'validation', 'test' splits,
        each containing:  {'text': str, 'label': int}
          0 = fake  (pants-fire, false, barely-true → mapped to fake)
          1 = real  (true, mostly-true → mapped to real)
          2 = uncertain (half-true → uncertain)
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast

from backend.core.logging import get_logger

logger = get_logger(__name__)

# ── Label mapping ─────────────────────────────────────────────────────────────

LIAR_LABEL_MAP: Dict[str, int] = {
    "pants-fire": 0,
    "false": 0,
    "barely-true": 0,
    "half-true": 2,
    "mostly-true": 1,
    "true": 1,
}

BINARY_LABEL_MAP: Dict[str, int] = {
    "fake": 0,
    "real": 1,
    "0": 0,
    "1": 1,
}

LABEL_NAMES = ["fake", "real", "uncertain"]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str, max_chars: int = 3000) -> str:
    """
    Normalise whitespace, strip HTML artefacts, truncate to max_chars.
    We truncate here (not in the tokenizer) to fail fast and keep RAM usage low.
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Truncate
    return text[:max_chars]


# ── LIAR dataset ──────────────────────────────────────────────────────────────

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true_count", "false_count",
    "half_true_count", "mostly_true_count", "pants_fire_count", "context",
]


def _load_liar_split(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=LIAR_COLUMNS)
    df = df[["label", "statement", "speaker", "subject", "context"]].copy()
    df = df[df["label"].isin(LIAR_LABEL_MAP)].copy()
    df["label_id"] = df["label"].map(LIAR_LABEL_MAP)

    # Enrich text: combine statement + speaker context for better signal
    df["text"] = (
        df["statement"].fillna("")
        + " [SPEAKER] "
        + df["speaker"].fillna("unknown")
        + " [SUBJECT] "
        + df["subject"].fillna("")
        + " [CONTEXT] "
        + df["context"].fillna("")
    ).apply(clean_text)

    return df[["text", "label_id"]].rename(columns={"label_id": "label"})


def load_liar_dataset(data_dir: str = "data/raw/liar") -> DatasetDict:
    """Load LIAR dataset from the standard TSV files."""
    base = Path(data_dir)
    splits = {}
    for split, fname in [("train", "train.tsv"), ("validation", "valid.tsv"), ("test", "test.tsv")]:
        path = base / fname
        if not path.exists():
            raise FileNotFoundError(
                f"LIAR {split} file not found at {path}. "
                f"Run: python data/scripts/download_dataset.py"
            )
        df = _load_liar_split(path)
        splits[split] = Dataset.from_pandas(df, preserve_index=False)
        logger.info("Loaded LIAR split", split=split, rows=len(df))

    return DatasetDict(splits)


# ── FakeNewsNet (binary) ──────────────────────────────────────────────────────

def load_fakenewsnet_dataset(csv_path: str = "data/raw/fakenewsnet.csv") -> DatasetDict:
    """
    Loads a combined FakeNewsNet CSV.
    Expected columns: 'text', 'label' (fake|real or 0|1)
    """
    df = pd.read_csv(csv_path)
    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df["label"] = df["label"].astype(str).str.lower().map(BINARY_LABEL_MAP)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    df = df[["text", "label"]]

    # 80/10/10 split
    train_df = df.sample(frac=0.8, random_state=42)
    remaining = df.drop(train_df.index)
    val_df = remaining.sample(frac=0.5, random_state=42)
    test_df = remaining.drop(val_df.index)

    logger.info(
        "Loaded FakeNewsNet",
        train=len(train_df), val=len(val_df), test=len(test_df),
    )
    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    })


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
    text_column: str = "text",
    num_proc: int = 4,
) -> DatasetDict:
    """
    Tokenize all splits in-place.
    Returns a new DatasetDict with input_ids, attention_mask, labels columns.
    """

    def _tokenize_batch(batch: Dict[str, List]) -> Dict[str, List]:
        encodings = tokenizer(
            batch[text_column],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": batch["label"],
        }

    tokenized = dataset.map(
        _tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=[text_column, "label"],
        desc="Tokenising",
    )
    tokenized.set_format("torch")
    logger.info("Tokenisation complete", splits=list(tokenized.keys()))
    return tokenized


# ── Class weight helper ───────────────────────────────────────────────────────

def compute_class_weights(dataset: Dataset) -> List[float]:
    """
    Compute inverse-frequency class weights to handle imbalanced datasets.
    Returns a list of floats indexed by class id.
    """
    from collections import Counter
    import math

    counts = Counter(dataset["label"])
    n_total = sum(counts.values())
    n_classes = len(LABEL_NAMES)

    weights = []
    for cls_id in range(n_classes):
        count = counts.get(cls_id, 1)  # avoid division by zero
        weight = n_total / (n_classes * count)
        weights.append(weight)

    logger.info("Class weights computed", weights=weights, counts=dict(counts))
    return weights
