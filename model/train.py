"""Train the SignLSTM on the processed dataset.

Loads data/processed/{X,y}.npy, does a stratified 70/15/15 split, trains with
Adam + CrossEntropyLoss, keeps the best model by validation accuracy, then
reports test accuracy and a confusion matrix (to spot commonly confused pairs).

Run from the project root:  python -m model.train
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import config
from model.model import SignLSTM, save_model


def _load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not config.X_PATH.exists():
        raise SystemExit(
            f"{config.X_PATH} not found. Run `python -m preprocessing.build_dataset` first."
        )
    X = np.load(config.X_PATH)
    y = np.load(config.Y_PATH)
    words = json.loads(config.PROCESSED_LABEL_MAP_PATH.read_text())["idx_to_word"]
    return X, y, words


def _stratify_or_none(y: np.ndarray):
    """Stratify only if every class has >=2 samples (train_test_split requirement)."""
    _, counts = np.unique(y, return_counts=True)
    return y if counts.min() >= 2 else None


def _split(X, y):
    test_frac = 1.0 - config.TRAIN_SPLIT - config.VAL_SPLIT
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=config.RANDOM_SEED,
        stratify=_stratify_or_none(y))
    val_frac = config.VAL_SPLIT / (config.TRAIN_SPLIT + config.VAL_SPLIT)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=config.RANDOM_SEED,
        stratify=_stratify_or_none(y_tmp))
    return X_tr, y_tr, X_val, y_val, X_test, y_test


def _accuracy(model, X, y, device) -> float:
    if len(X) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        pred = model(xb).argmax(1).cpu().numpy()
    return float((pred == y).mean())


def _print_confusion(model, X, y, words, device) -> list[list[int]]:
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    cm = confusion_matrix(y, pred, labels=list(range(len(words))))
    width = max(len(config.display_word(w)) for w in words)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(" " * (width + 2) + " ".join(f"{i:>4d}" for i in range(len(words))))
    for i, w in enumerate(words):
        row = " ".join(f"{v:>4d}" for v in cm[i])
        print(f"{config.display_word(w):>{width}}  {row}   [{i}]")
    return cm.tolist()


def train() -> None:
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, words = _load_dataset()
    print(f"dataset: X {X.shape}  y {y.shape}  classes {len(words)}  device {device}")
    X_tr, y_tr, X_val, y_val, X_test, y_test = _split(X, y)
    print(f"split: train {len(X_tr)}  val {len(X_val)}  test {len(X_test)}")

    model = SignLSTM(num_classes=len(words)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    n = len(X_tr_t)

    best_val, best_state = -1.0, None
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, config.BATCH_SIZE):
            bi = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(X_tr_t[bi]), y_tr_t[bi])
            loss.backward()
            opt.step()
            total += loss.item() * len(bi)
        val_acc = _accuracy(model, X_val, y_val, device)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {total / n:.4f}  val_acc {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc = _accuracy(model, X_tr, y_tr, device)
    test_acc = _accuracy(model, X_test, y_test, device)
    print(f"\nbest val_acc {best_val:.3f}  train_acc {train_acc:.3f}  test_acc {test_acc:.3f}")
    cm = _print_confusion(model, X_test, y_test, words, device) if len(X_test) else []

    save_model(model, words)
    config.METRICS_PATH.write_text(json.dumps({
        "classes": words,
        "train_acc": train_acc, "val_acc": best_val, "test_acc": test_acc,
        "confusion_matrix": cm,
        "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_test),
    }, indent=2))
    print(f"\nsaved model -> {config.MODEL_PATH}")
    print(f"saved labels -> {config.LABEL_MAP_PATH}")
    print(f"saved metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    train()
