"""Value labels from successful demonstrations (paper Sec. III-D).

1. Down-sample each demo's image sequence so that consecutive kept frames differ enough (or, alternatively,
   with a fixed stride). The first and the last frame are always kept.
2. Assign labels by linear interpolation between 0 and 1 over the kept sequence: kept frame i of L
   (0-based) gets i / (L - 1); in a 10-frame sequence, frame 5 gets 5/9 (the example of Sec. III-D).

A frame is kept when its mean absolute grayscale difference (at low resolution) to the last kept frame is
at least tau, so that consecutive kept frames contain sufficient variation. Only numpy is required.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def progress_labels(length: int) -> np.ndarray:
    """Linear 0 -> 1 labels for `length` kept frames: label_i = i / (length - 1)."""
    if length < 1:
        raise ValueError("length must be >= 1.")
    if length == 1:
        return np.ones(1, dtype=np.float32)
    return (np.arange(length, dtype=np.float64) / (length - 1)).astype(np.float32)


def _small_gray(frame: np.ndarray, size: int) -> np.ndarray:
    """Grayscale in [0, 1], nearest-neighbour resized to (size, size)."""
    frame = np.asarray(frame)
    gray = frame.astype(np.float32).mean(axis=-1) / 255.0 if frame.ndim == 3 else frame.astype(np.float32) / 255.0
    rows = np.linspace(0, gray.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, gray.shape[1] - 1, size).round().astype(int)
    return gray[np.ix_(rows, cols)]


def select_keyframes(
    frames: np.ndarray,
    mode: str = "visual",
    threshold: float = 0.01,
    stride: int = 5,
    compare_size: int = 64,
) -> np.ndarray:
    """Indices of the kept frames of one demonstration (always includes the first and the last frame).

    Args:
        frames: (T, H, W, 3) uint8 image sequence.
        mode: "visual" keeps frame t when its mean absolute grayscale difference to the last kept frame is
            >= `threshold` (pixel values scaled to [0, 1], frames resized to compare_size x compare_size);
            "stride" keeps every `stride`-th frame; "all" keeps every frame.
    """
    num = len(frames)
    if num == 0:
        raise ValueError("Empty frame sequence.")
    if mode == "all":
        keep = list(range(num))
    elif mode == "stride":
        if stride < 1:
            raise ValueError("stride must be >= 1.")
        keep = list(range(0, num, stride))
    elif mode == "visual":
        keep = [0]
        last = _small_gray(frames[0], compare_size)
        for t in range(1, num):
            cur = _small_gray(frames[t], compare_size)
            if float(np.mean(np.abs(cur - last))) >= threshold:
                keep.append(t)
                last = cur
    else:
        raise ValueError(f"Unknown down-sampling mode {mode!r}.")
    if keep[-1] != num - 1:
        keep.append(num - 1)
    return np.asarray(keep, dtype=np.int64)


def value_samples(
    frames: np.ndarray, mode: str = "visual", threshold: float = 0.01, stride: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """(kept frame indices, labels) for one successful demonstration."""
    idx = select_keyframes(frames, mode=mode, threshold=threshold, stride=stride)
    return idx, progress_labels(len(idx))
