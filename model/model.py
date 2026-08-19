"""PyTorch LSTM classifier — the spec's architecture, PyTorch equivalent.

Spec (Keras):
    LSTM(64, return_sequences=True) -> Dropout(0.3)
    -> LSTM(32) -> Dropout(0.3)
    -> Dense(32, relu) -> Dense(num_classes, softmax)

Here the final softmax is left out of the module because CrossEntropyLoss expects
raw logits; inference applies softmax explicitly (see load_model / predict).
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn

import config


class SignLSTM(nn.Module):
    def __init__(self, num_classes: int, num_features: int = config.FEATURES):
        super().__init__()
        self.lstm1 = nn.LSTM(num_features, config.LSTM1_UNITS, batch_first=True)
        self.drop1 = nn.Dropout(config.DROPOUT)
        self.lstm2 = nn.LSTM(config.LSTM1_UNITS, config.LSTM2_UNITS, batch_first=True)
        self.drop2 = nn.Dropout(config.DROPOUT)
        self.dense = nn.Linear(config.LSTM2_UNITS, config.DENSE_UNITS)
        self.relu = nn.ReLU()
        self.out = nn.Linear(config.DENSE_UNITS, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, SEQ_LEN, FEATURES)
        seq, _ = self.lstm1(x)          # return_sequences=True
        seq = self.drop1(seq)
        _, (h_n, _) = self.lstm2(seq)   # take last hidden state (return_sequences=False)
        h = self.drop2(h_n[-1])
        h = self.relu(self.dense(h))
        return self.out(h)              # logits


def save_model(model: SignLSTM, words: list[str]) -> None:
    torch.save(
        {"state_dict": model.state_dict(),
         "num_classes": len(words),
         "num_features": config.FEATURES,
         "seq_len": config.SEQ_LEN},
        config.MODEL_PATH,
    )
    config.LABEL_MAP_PATH.write_text(
        json.dumps({"idx_to_word": words,
                    "word_to_idx": {w: i for i, w in enumerate(words)}}, indent=2)
    )


def load_model(device: str = "cpu") -> tuple[SignLSTM, list[str]]:
    """Load the trained model and its label list, ready for inference (eval mode)."""
    ckpt = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    label_map = json.loads(config.LABEL_MAP_PATH.read_text())
    words = label_map["idx_to_word"]
    model = SignLSTM(ckpt["num_classes"], ckpt["num_features"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, words


@torch.no_grad()
def predict(model: SignLSTM, sequence, device: str = "cpu") -> tuple[int, float]:
    """Return (class_index, confidence) for one preprocessed (SEQ_LEN, FEATURES) seq."""
    import numpy as np
    x = torch.as_tensor(np.asarray(sequence)[None], dtype=torch.float32, device=device)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = torch.max(probs, dim=0)
    return int(idx.item()), float(conf.item())
