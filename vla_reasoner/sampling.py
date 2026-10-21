"""Candidate samplers for the expansion step: KDE Top-k and the Gaussian-noise ablation."""

from __future__ import annotations

import math
from typing import Protocol, Tuple, Union

import numpy as np

from vla_reasoner.kde import ActionKDE


class CandidateSampler(Protocol):
    k: int

    def __call__(self, reference: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Return (candidates (k, d), log_densities (k,)) around the flattened reference action (d,)."""


class KDETopKSampler:
    def __init__(self, kde: ActionKDE, k: int, num_samples: int):
        if num_samples < k:
            raise ValueError(f"num_samples ({num_samples}) must be >= k ({k}).")
        self.kde, self.k, self.num_samples = kde, k, num_samples

    def __call__(self, reference, rng):
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        if reference.size != self.kde.dim:
            raise ValueError(
                f"Action of size {reference.size} does not match the KDE dimension {self.kde.dim}; was the KDE fit "
                f"with the same executed chunk size (KDE sample shape {self.kde.action_shape})?"
            )
        return self.kde.top_k(reference, self.k, self.num_samples, rng)


class GaussianNoiseSampler:
    """a_j = reference + eps_j,  eps_j ~ N(0, diag(std^2)),  j = 1..k."""

    def __init__(self, k: int, std: Union[float, np.ndarray]):
        self.k = k
        self.std = np.asarray(std, dtype=np.float64)
        if np.any(self.std <= 0):
            raise ValueError("Noise std must be positive.")

    def __call__(self, reference, rng):
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        std = np.broadcast_to(self.std, reference.shape)
        eps = rng.standard_normal((self.k, reference.size)) * std
        log_p = -0.5 * np.sum((eps / std) ** 2, axis=1) - np.sum(np.log(std)) - 0.5 * reference.size * math.log(2 * math.pi)
        return reference[None] + eps, log_p
