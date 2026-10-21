"""BridgeData V2 episodes from the RLDS release, with the action transforms of Octo and OpenVLA."""
# Portions adapted from Octo (Copyright (c) 2023 Robotic AI & Learning Lab Berkeley, MIT); see THIRD_PARTY_NOTICES.md.

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from vla_reasoner.data.episode import Episode


def binarize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """numpy port of octo's `binarize_gripper_actions` (open > 0.95 -> 1, closed < 0.05 -> 0, in-between
    values take the value of the next decided step; a trailing in-between run keeps the last raw value)."""
    actions = np.asarray(actions, dtype=np.float32)
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    out = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(len(actions))):
        if open_mask[i] or closed_mask[i]:
            carry = np.float32(open_mask[i])
        out[i] = carry
    return out


def transform_bridge_episode(actions: np.ndarray, states: np.ndarray, convention: str = "octo"):
    """Returns (keep_index, actions) where keep_index selects the observation steps that remain."""
    actions = np.asarray(actions, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    index = np.arange(len(actions))
    if convention == "raw":
        return index, actions
    if convention == "bridge_orig":
        actions, states, index = actions[1:], states[1:], index[1:]
    elif convention != "octo":
        raise ValueError(f"Unknown Bridge action convention {convention!r}.")
    gripper = binarize_gripper_actions(actions[:, -1])
    movement = states[1:, :6] - states[:-1, :6]  # relabel with reached proprio, drop the last step
    new_actions = np.concatenate([movement, gripper[:-1, None]], axis=1).astype(np.float32)
    return index[:-1], new_actions


def iter_bridge_rlds(
    data_dir: str,
    dataset_name: str = "bridge_dataset",
    split: str = "train",
    image_key: str = "image_0",
    action_convention: str = "octo",
    max_episodes: Optional[int] = None,
) -> Iterator[Episode]:
    """Episodes from a TFDS builder directory, e.g. <data_dir>/bridge_dataset/1.0.0."""
    import os

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    builder_dir = data_dir
    if not os.path.exists(os.path.join(builder_dir, "dataset_info.json")):
        versions = sorted(os.listdir(os.path.join(data_dir, dataset_name)))
        builder_dir = os.path.join(data_dir, dataset_name, versions[-1])
    builder = tfds.builder_from_directory(builder_dir)
    ds = builder.as_dataset(split=split)
    count = 0
    for episode in tfds.as_numpy(ds):
        steps = list(episode["steps"])
        if len(steps) < 3:
            continue
        images = np.stack([s["observation"][image_key] for s in steps])
        states = np.stack([s["observation"]["state"] for s in steps])
        raw = np.stack([s["action"] for s in steps])
        instruction = steps[0].get("language_instruction", b"")
        instruction = instruction.decode() if isinstance(instruction, bytes) else str(instruction)
        keep, actions = transform_bridge_episode(raw, states, action_convention)
        if len(actions) == 0:
            continue
        yield Episode(images[keep], actions, instruction, True)
        count += 1
        if max_episodes is not None and count >= max_episodes:
            return
