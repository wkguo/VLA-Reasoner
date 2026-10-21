"""LIBERO hdf5 demonstrations as episodes in OpenVLA's action convention."""
# Portions adapted from OpenVLA (Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, MIT); see THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import glob
import json
import os
import re
from typing import Iterator, Optional

import numpy as np

from vla_reasoner.data.episode import Episode


def is_noop(action: np.ndarray, prev_action: Optional[np.ndarray], threshold: float = 1e-4) -> bool:
    """OpenVLA's no-op criterion: arm dims ~0 and (if there is a previous action) gripper unchanged."""
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def noop_keep_mask(actions: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(actions), dtype=bool)
    prev = None
    for t, a in enumerate(actions):
        if not is_noop(a, prev):
            keep[t] = True
            prev = a  # compared against the last *kept* action, as in OpenVLA's script
    return keep


def libero_to_openvla_actions(actions: np.ndarray) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    out[:, -1] = 1.0 - np.clip(out[:, -1], 0.0, 1.0)
    return out


def _instruction(h5, path: str) -> str:
    info = h5["data"].attrs.get("problem_info")
    if info is not None:
        try:
            return str(json.loads(info)["language_instruction"]).strip()
        except (ValueError, KeyError, TypeError):
            pass
    name = os.path.basename(path).replace("_demo.hdf5", "").replace(".hdf5", "")
    name = re.sub(r"^([A-Z0-9]+_)+", "", name)  # drop scene prefixes such as KITCHEN_SCENE3_
    return name.replace("_", " ")


def iter_libero_hdf5(
    path: str,
    image_key: str = "agentview_rgb",
    flip: bool = True,
    action_convention: str = "openvla",
    filter_noops: bool = True,
    max_episodes: Optional[int] = None,
) -> Iterator[Episode]:
    """Episodes from one LIBERO hdf5 file or a directory of them (e.g. a whole suite)."""
    import h5py

    files = [path] if path.endswith(".hdf5") else sorted(glob.glob(os.path.join(path, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found at {path}")
    count = 0
    for fpath in files:
        with h5py.File(fpath, "r") as h5:
            instruction = _instruction(h5, fpath)
            demos = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for demo in demos:
                actions = np.asarray(h5["data"][demo]["actions"][()], dtype=np.float32)
                images = np.asarray(h5["data"][demo]["obs"][image_key][()])
                if filter_noops:
                    keep = noop_keep_mask(actions)
                    actions, images = actions[keep], images[keep]
                if flip:
                    images = images[:, ::-1, ::-1]
                if action_convention == "openvla":
                    actions = libero_to_openvla_actions(actions)
                elif action_convention != "raw":
                    raise ValueError(f"Unknown LIBERO action convention {action_convention!r}.")
                yield Episode(np.ascontiguousarray(images), actions, instruction, True,
                              meta={"path": fpath, "demo": demo})
                count += 1
                if max_episodes is not None and count >= max_episodes:
                    return
