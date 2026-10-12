"""Training loop for the value head (Sec. III-D, Eq. 7): frozen ResNet-34 features + MSE on progress labels.

Because the encoder is frozen, features are extracted once and only the 2-layer MLP is optimized.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from vla_reasoner.value.model import ResNet34Value


def encode_samples(
    model: ResNet34Value, samples: Iterable[Tuple[np.ndarray, np.ndarray, int]], batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen encoder over (frames, labels, episode_id) samples -> (features, labels, episode ids)."""
    feats, labels, groups = [], [], []
    for frames, y, ep_id in samples:
        for i in range(0, len(frames), batch_size):
            feats.append(model.encode(frames[i:i + batch_size]).float().cpu().numpy())
        labels.append(np.asarray(y, dtype=np.float32))
        groups.append(np.full(len(y), ep_id, dtype=np.int64))
    if not labels:
        raise ValueError("No successful episodes found to train the value function.")
    return np.concatenate(feats), np.concatenate(labels), np.concatenate(groups)


def fit_value_head(
    model: ResNet34Value,
    feats: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Train `model.head` with MSE. Validation split is by episode. Returns per-epoch metrics."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.head.parameters()).device
    episodes = np.unique(groups)
    rng.shuffle(episodes)
    n_val = int(round(val_fraction * len(episodes))) if len(episodes) > 1 else 0
    val_mask = np.isin(groups, episodes[:n_val])
    x_tr, y_tr = torch.from_numpy(feats[~val_mask]).to(device), torch.from_numpy(labels[~val_mask]).to(device)
    x_va, y_va = torch.from_numpy(feats[val_mask]).to(device), torch.from_numpy(labels[val_mask]).to(device)

    opt = torch.optim.Adam(model.head.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(epochs):
        model.head.train()
        perm = torch.randperm(len(x_tr), device=device)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            loss = F.mse_loss(model.head(x_tr[b]).squeeze(-1), y_tr[b])  # Eq. 7
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        metrics = {"epoch": epoch, "train_mse": total / max(len(x_tr), 1)}
        if len(x_va):
            model.head.eval()
            with torch.no_grad():
                metrics["val_mse"] = F.mse_loss(model.head(x_va).squeeze(-1), y_va).item()
        history.append(metrics)
    model.eval()
    return history
