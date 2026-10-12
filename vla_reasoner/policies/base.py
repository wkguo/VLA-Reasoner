"""Frozen VLA interface: a_t^VLA = pi_theta(s_t, l) (paper Sec. III-A). The VLA's weights are never updated.

`predict` returns *raw* actions, i.e. the policy output before benchmark-specific post-processing (gripper
binarization/inversion, Euler -> axis-angle, ...). The reasoner works in this raw space, which is also the space
the KDE and the world model are fit in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class VLAPolicy(ABC):
    action_dim: int = 7
    exec_chunk: int = 1  # number of actions executed per query (OpenVLA/Octo 1, SpatialVLA 4, pi0-FAST 5)

    @abstractmethod
    def reset(self, instruction: str) -> None:
        """Start a new episode with language instruction l."""

    @abstractmethod
    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """Raw actions to execute for this query, shape (exec_chunk, action_dim)."""
