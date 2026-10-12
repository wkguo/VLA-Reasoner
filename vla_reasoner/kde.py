"""Gaussian kernel density estimate over (flattened) actions -- paper Sec. III-C, Eq. 6.

    pi_KDE(a) = 1/N * sum_i K_h(a - a_i),   K = Gaussian kernel, bandwidth h.

The KDE is fit on the actions of the same offline dataset that was used to fine-tune the VLA.
Action chunks are treated as single samples by flattening them (Sec. III-C, last paragraph).

Implementation notes (see README):
  * Optional per-dimension standardization z = (a - mean) / std before fitting. The kernel is
    isotropic in z-space, i.e. diagonal with per-dimension width h * std_j in action space.
    Densities returned by `log_density` are always in the *original* action space (the constant
    Jacobian -sum_j log std_j is included), so they are proper densities either way.
    Dimensions whose empirical std is below `min_std` (constant in the data) are clamped to
    std = min_std, so samples keep the data's constant value there, and they are left out of the
    Top-k distance, so a small deviation of the reference action in such a dimension cannot
    dominate the neighbour search.
  * Bandwidth h: Scott's rule n^(-1/(d+4)), Silverman's rule (n(d+2)/4)^(-1/(d+4)) (both in
    standardized units), or a manual float.
  * All densities are computed in log-space with a streaming logsumexp for numerical stability.

Only numpy is required.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

Bandwidth = Union[str, float]


def _as_2d(actions: np.ndarray) -> np.ndarray:
    """(n, d) or (n, H, A) -> (n, d) float64. A single action (d,) becomes (1, d)."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        return actions[None]
    return actions.reshape(actions.shape[0], -1)


def bandwidth_factor(rule: Bandwidth, n: int, d: int) -> float:
    """Bandwidth h for an isotropic Gaussian kernel (in standardized units for the rules)."""
    if isinstance(rule, str):
        if rule == "scott":
            return float(n ** (-1.0 / (d + 4)))
        if rule == "silverman":
            return float((n * (d + 2) / 4.0) ** (-1.0 / (d + 4)))
        try:
            rule = float(rule)
        except ValueError as e:
            raise ValueError(f"Unknown bandwidth rule {rule!r}; use 'scott', 'silverman' or a float.") from e
    h = float(rule)
    if not h > 0:
        raise ValueError(f"Bandwidth must be positive, got {h}.")
    return h


def nearest_k(candidates: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k candidates closest to `reference` in Euclidean distance (sorted by distance).

    This is the Top-k step of Sec. III-B(a), Eq. 2: A^Top-k = argmin_{A subset, |A|=k} sum_{a in A} ||a - a_i||_2.
    """
    candidates = _as_2d(candidates)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if not 1 <= k <= len(candidates):
        raise ValueError(f"k={k} must be in [1, {len(candidates)}].")
    dist = np.linalg.norm(candidates - reference[None], axis=1)
    idx = np.argpartition(dist, k - 1)[:k]
    return idx[np.argsort(dist[idx], kind="stable")]


class ActionKDE:
    """Gaussian KDE over flattened actions (Eq. 6)."""

    def __init__(self, bandwidth: Bandwidth = "scott", standardize: bool = True, min_std: float = 1e-6):
        self.bandwidth = bandwidth
        self.standardize = standardize
        self.min_std = min_std
        self._z: Optional[np.ndarray] = None  # fitted data in working (z) space, float32
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.h: Optional[float] = None
        self.action_shape: Optional[tuple] = None  # shape of one (unflattened) sample, e.g. (H, A)
        self.varying: Optional[np.ndarray] = None  # bool (d,): dimensions used by the Top-k distance

    # ------------------------------------------------------------------ fitting / io
    def fit(self, actions: np.ndarray) -> "ActionKDE":
        actions = np.asarray(actions)
        if actions.ndim < 2 or len(actions) < 2:
            raise ValueError("Need at least two samples of shape (n, d) or (n, H, A).")
        self.action_shape = tuple(actions.shape[1:])
        a = _as_2d(actions)
        n, d = a.shape
        if self.standardize:
            self.mean = a.mean(axis=0)
            std = a.std(axis=0)
            self.varying = std >= self.min_std
            self.std = np.maximum(std, self.min_std)  # (near-)constant dimensions stay (near-)constant
        else:
            self.mean = np.zeros(d)
            self.std = np.ones(d)
            self.varying = np.ones(d, dtype=bool)
        self._z = ((a - self.mean) / self.std).astype(np.float32)
        self.h = bandwidth_factor(self.bandwidth, n, d)
        return self

    @property
    def n(self) -> int:
        self._check()
        return self._z.shape[0]

    @property
    def dim(self) -> int:
        self._check()
        return self._z.shape[1]

    def _check(self):
        if self._z is None:
            raise RuntimeError("ActionKDE is not fitted.")

    def to_z(self, actions: np.ndarray) -> np.ndarray:
        self._check()
        return (_as_2d(actions) - self.mean) / self.std

    def from_z(self, z: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def save(self, path: str) -> None:
        self._check()
        np.savez_compressed(
            path,
            z=self._z,
            mean=self.mean,
            std=self.std,
            h=np.float64(self.h),
            standardize=np.bool_(self.standardize),
            bandwidth=np.array(str(self.bandwidth)),
            action_shape=np.array(self.action_shape, dtype=np.int64),
            varying=self.varying,
            min_std=np.float64(self.min_std),
        )

    @classmethod
    def load(cls, path: str) -> "ActionKDE":
        with np.load(path, allow_pickle=False) as f:
            min_std = float(f["min_std"]) if "min_std" in f.files else 1e-6
            kde = cls(bandwidth=str(f["bandwidth"]), standardize=bool(f["standardize"]), min_std=min_std)
            kde._z = f["z"].astype(np.float32)
            kde.mean, kde.std = f["mean"], f["std"]
            if "varying" in f.files:
                kde.varying = f["varying"].astype(bool)
            else:  # files written without the mask: recover the data's own spread from z and std
                kde.varying = kde._z.astype(np.float64).std(axis=0) * kde.std >= min_std
            kde.h = float(f["h"])
            kde.action_shape = tuple(int(x) for x in f["action_shape"])
        return kde

    # ------------------------------------------------------------------ density / sampling
    def log_density(self, actions: np.ndarray, block: int = 65536) -> np.ndarray:
        """log pi_KDE(a) for each row of `actions` (Eq. 6), in original action space."""
        self._check()
        z = self.to_z(actions).astype(np.float32)
        m, d = z.shape
        h2 = self.h ** 2
        zz = np.sum(z.astype(np.float64) ** 2, axis=1)  # (m,)
        running = np.full(m, -np.inf)
        for start in range(0, self.n, block):
            data = self._z[start:start + block]
            sq = zz[:, None] + np.sum(data.astype(np.float64) ** 2, axis=1)[None] - 2.0 * (z @ data.T)
            logk = -0.5 * np.maximum(sq, 0.0) / h2  # (m, b)
            running = np.logaddexp(running, _logsumexp_rows(logk))
        log_norm = -math.log(self.n) - 0.5 * d * math.log(2 * math.pi * h2) - float(np.sum(np.log(self.std)))
        return running + log_norm

    def density(self, actions: np.ndarray) -> np.ndarray:
        return np.exp(self.log_density(actions))

    def sample(self, num: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `num` samples a ~ pi_KDE (pick a data point, add kernel noise). Returns (num, d)."""
        self._check()
        idx = rng.integers(0, self.n, size=num)
        noise = self.h * rng.standard_normal((num, self.dim))
        if self.varying is not None:
            noise[:, ~self.varying] = 0.0  # dimensions that are constant in the data stay constant
        z = self._z[idx].astype(np.float64) + noise
        return self.from_z(z)

    def top_k(self, reference: np.ndarray, k: int, num_samples: int, rng: np.random.Generator):
        """Top-k expansion candidates (Sec. III-B(a), Eq. 2, Alg. 1 line 3).

        Sample N = `num_samples` actions from pi_KDE and keep the k closest to `reference`.
        Distances are measured in the KDE working space (standardized if `standardize=True`) over the
        dimensions that vary in the data. Returns (actions (k, d), log_densities (k,)).
        """
        samples = self.sample(num_samples, rng)
        mask = self.varying if self.varying is not None and self.varying.any() else slice(None)
        idx = nearest_k(self.to_z(samples)[:, mask], self.to_z(reference)[0][mask], k)
        chosen = samples[idx]
        return chosen, self.log_density(chosen)


def _logsumexp_rows(x: np.ndarray) -> np.ndarray:
    mx = np.max(x, axis=1)
    safe = np.where(np.isfinite(mx), mx, 0.0)
    return safe + np.log(np.sum(np.exp(x - safe[:, None]), axis=1))
