"""Cached-feature dataset for training the value head."""

from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import numpy as np

from vla_reasoner.data.episode import Episode
from vla_reasoner.value.labels import value_samples


def iter_value_samples(
    episodes: Iterable[Episode], mode: str = "visual", threshold: float = 0.01, stride: int = 5
) -> Iterator[Tuple[np.ndarray, np.ndarray, int]]:
    """Yield (kept frames (L, H, W, 3) uint8, labels (L,), episode id) for every successful episode."""
    for ep_id, ep in enumerate(episodes):
        if not ep.success:
            continue
        idx, labels = value_samples(ep.images, mode=mode, threshold=threshold, stride=stride)
        yield ep.images[idx], labels, ep_id
