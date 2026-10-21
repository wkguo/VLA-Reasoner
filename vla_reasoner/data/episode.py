"""Episode container and npz serialization."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class Episode:
    images: np.ndarray
    actions: np.ndarray
    instruction: str = ""
    success: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.images) != len(self.actions):
            raise ValueError(f"images ({len(self.images)}) and actions ({len(self.actions)}) differ in length.")

    def __len__(self) -> int:
        return len(self.actions)


def save_episode_npz(path: str, ep: Episode) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        image=np.asarray(ep.images, dtype=np.uint8),
        action=np.asarray(ep.actions, dtype=np.float32),
        instruction=np.array(ep.instruction),
        success=np.bool_(ep.success),
    )


def load_episode_npz(path: str) -> Episode:
    with np.load(path, allow_pickle=False) as f:
        return Episode(
            images=f["image"],
            actions=f["action"].astype(np.float32),
            instruction=str(f["instruction"]) if "instruction" in f else "",
            success=bool(f["success"]) if "success" in f else True,
            meta={"path": path},
        )


def iter_npz_episodes(path: str, success: Optional[bool] = None, max_episodes: Optional[int] = None) -> Iterator[Episode]:
    """Episodes from one .npz file or a directory of them (sorted). `success` filters by the success flag."""
    files = [path] if path.endswith(".npz") else sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    count = 0
    for f in files:
        ep = load_episode_npz(f)
        if success is not None and ep.success != success:
            continue
        yield ep
        count += 1
        if max_episodes is not None and count >= max_episodes:
            return


def action_chunks(actions: np.ndarray, chunk: int) -> np.ndarray:
    """All length-`chunk` sliding windows (stride 1) of one episode's actions: (T - chunk + 1, chunk, A).

    Used to fit the KDE on action chunks treated as single samples (Sec. III-C). Episodes shorter than
    `chunk` contribute nothing.
    """
    actions = np.asarray(actions)
    num = len(actions) - chunk + 1
    if num <= 0:
        return np.zeros((0, chunk) + actions.shape[1:], dtype=actions.dtype)
    return np.stack([actions[i:i + chunk] for i in range(num)])


def pack_for_world_model(ep: Episode, chunk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Frames at stride `chunk` with the corresponding executed chunk packed into one action vector.

    frame_j = images[j * chunk],  action_j = concat(actions[j*chunk : (j+1)*chunk])  for j = 0..n-1, plus the
    frame reached after the last full chunk (frame_n) paired with a zero action, so that len(image) ==
    len(action) as iVideoGPT expects (its transformer never uses the action of the last frame).
    With chunk = 1 the episode is returned unchanged.
    """
    images, actions = np.asarray(ep.images), np.asarray(ep.actions, dtype=np.float32)
    if chunk == 1:
        return images, actions
    n = (len(actions) - 1) // chunk  # full chunks whose resulting frame is observed
    if n < 1:
        return images[:0], np.zeros((0, chunk * actions.shape[1]), dtype=np.float32)
    frames = images[0:n * chunk + 1:chunk]
    packed = actions[:n * chunk].reshape(n, chunk * actions.shape[1])
    packed = np.concatenate([packed, np.zeros((1, packed.shape[1]), dtype=np.float32)])
    return frames, packed
