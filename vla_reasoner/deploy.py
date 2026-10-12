"""Robot-agnostic deployment loop for real-world runs (paper Sec. IV-B; no hardware drivers are included).

Implement `Robot` for your arm (the paper used a Galaxea-A1 with a side camera and a wrist camera). Per step
the robot returns a dict with
    "policy_input": the observation dict the VLA expects (e.g. openpi keys, or {"full_image": ...} for OpenVLA)
    "frame":        (H, W, 3) uint8 image of the camera the world model / value function were trained on.
`apply_action` receives one raw action of the VLA's action space and must perform any robot-specific
post-processing (unnormalization, gripper mapping, safety limits).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import numpy as np

from vla_reasoner.policies.base import VLAPolicy


class Robot(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Move to the start configuration."""

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Return {"policy_input": ..., "frame": ...}."""

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Execute one raw action."""

    def is_done(self) -> bool:
        """Optional early stop (e.g. operator key press)."""
        return False


def run_episode(
    robot: Robot,
    policy: VLAPolicy,
    instruction: str,
    max_steps: int,
    reasoner: Optional[Callable] = None,
    on_query: Optional[Callable[[int, Dict[str, Any], np.ndarray, np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Run one episode; `reasoner` is a `VLAReasoner` (or None for the vanilla VLA).

    Returns per-step frames (the observation before each executed action) and the executed raw actions, i.e.
    the npz episode format used for world-model fine-tuning (`vla_reasoner.data.episode`).
    """
    robot.reset()
    policy.reset(instruction)
    if reasoner is not None:
        reasoner.reset()
    frames, executed = [], []
    t = 0
    obs = robot.get_observation()
    while t < max_steps and not robot.is_done():
        a_vla = np.asarray(policy.predict(obs["policy_input"]), dtype=np.float64).reshape(policy.exec_chunk, -1)
        actions = reasoner(a_vla, obs["frame"]) if reasoner is not None else a_vla
        if on_query is not None:
            on_query(t, obs, a_vla, actions)
        for i, a in enumerate(actions):
            if i > 0:
                obs = robot.get_observation()
            frames.append(np.asarray(obs["frame"]))
            robot.apply_action(a)
            executed.append(a)
            t += 1
            if t >= max_steps or robot.is_done():
                break
        obs = robot.get_observation()
    return {"frames": frames, "actions": np.asarray(executed)}
