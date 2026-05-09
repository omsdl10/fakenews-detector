"""
Training pipeline for the FakeNewsClassifier.

Features:
  - AdamW optimiser with linear warmup + cosine decay
  - Mixed-precision (AMP) training when CUDA is available
  - Gradient clipping to prevent exploding gradients
  - Early stopping on validation F1
  - Checkpoint saving (best model only)
  - Comprehensive metric logging (loss, accuracy, F1, AUC)

Usage:
  python -m backend.ml.trainer \\
      --dataset liar \\
      --data_dir data/raw/liar \\
      --output_dir models/fakenews_roberta \\
      --epochs 5 \\
      --batch_size 16 \\
      --lr 2e-5
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from backend.core.config import get_settings
from backend.core.logging import get_logger
from backend.ml.dataset import (
    load_liar_dataset,
    load_fakenewsnet_dataset,
    tokenize_dataset,
    compute_class_weights,
    LABEL_NAMES,
)
from backend.ml.model import FakeNewsClassifier

logger = get_logger(__name__)
settings = get_settings()


class EarlyStopping:
    """Stop training if validation F1 does not improve for `patience` epochs."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: FakeNewsClassifier,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimiser.zero_grad()

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            logger.info(
                "Training step",
                step=step,
                loss=round(loss.item(), 4),
                lr=round(scheduler.get_last_lr()[0], 8),
            )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: FakeNewsClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate on val/test split. Returns dict with loss, accuracy, macro_f1, auc."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()

        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ROC-AUC (one-vs-rest, macro)
    try:
        auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = 0.0  # single-class batch edge case

    return {
        "loss": round(total_loss / len(loader), 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    }


def train(
    dataset_name: str = "liar",
    data_dir: str = "data/raw/liar",
    output_dir: str = "models/fakenews_roberta",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
    dropout: float = 0.1,
    label_smoothing: float = 0.1,
    max_seq_length: int = 512,
    patience: int = 3,
) -> None:
    """Full training run: load → tokenise → train → evaluate → save."""
    device = torch.device(settings.DEVICE)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training started",
        dataset=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=str(device),
    )

    # ── 1. Load dataset ────────────────────────────────────────────────────
    if dataset_name == "liar":
        raw_dataset = load_liar_dataset(data_dir)
    elif dataset_name == "fakenewsnet":
        raw_dataset = load_fakenewsnet_dataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ── 2. Tokenise ────────────────────────────────────────────────────────
    tokenizer = RobertaTokenizerFast.from_pretrained(settings.CLASSIFIER_MODEL_NAME)
    tokenized = tokenize_dataset(
        raw_dataset, tokenizer, max_length=max_seq_length
    )

    # ── 3. Build model with class weights ──────────────────────────────────
    class_weights = compute_class_weights(raw_dataset["train"])
    model = FakeNewsClassifier(
        model_name=settings.CLASSIFIER_MODEL_NAME,
        dropout=dropout,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
    ).to(device)

    # ── 4. DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
    )

    # ── 5. Optimiser + scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimiser,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    early_stopper = EarlyStopping(patience=patience)

    best_f1 = 0.0

    # ── 6. Training loop ────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimiser, scheduler, scaler, device
        )
        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            "Epoch complete",
            epoch=epoch,
            train_loss=round(train_loss, 4),
            **val_metrics,
        )

        # Save best checkpoint
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                model.state_dict(), output_path / "pytorch_model.bin"
            )
            tokenizer.save_pretrained(str(output_path))
            logger.info("Checkpoint saved", epoch=epoch, f1=best_f1)

        if early_stopper.step(val_metrics["macro_f1"]):
            logger.info("Early stopping triggered", epoch=epoch)
            break

    # ── 7. Final test evaluation ────────────────────────────────────────────
    test_loader = DataLoader(
        tokenized["test"], batch_size=batch_size * 2, shuffle=False, num_workers=2
    )
    # Reload best weights
    model.load_state_dict(
        torch.load(output_path / "pytorch_model.bin", weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test evaluation", **test_metrics)
    print(f"\n{'='*50}\nTest Results:\n{test_metrics}\n{'='*50}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news classifier")
    parser.add_argument("--dataset", default="liar", choices=["liar", "fakenewsnet"])
    parser.add_argument("--data_dir", default="data/raw/liar")
    parser.add_argument("--output_dir", default="models/fakenews_roberta")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
