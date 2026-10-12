"""Value-function interface used by the tree search (Alg. 1 line 6: v_hat <- MLP(s_hat)).

A value function is any callable mapping a list of world-model states to a float array of values.
This module depends only on numpy; the paper's ResNet-34 + MLP model lives in `value/model.py`.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence

import numpy as np

ValueFunction = Callable[[List[Any]], np.ndarray]


def frame_of(state: Any) -> np.ndarray:
    """Image of a world-model state: `state.frame` if present, else the state itself."""
    frame = getattr(state, "frame", state)
    return np.asarray(frame)


class TokenRewardValue:
    """Ablation "token reward head" (Sec. IV-C, Fig. 6 right): use the reward predicted by the world model.

    Requires a world model whose states carry a scalar `reward` (the iVideoGPT adapter does so when it is
    built with `reward_head=True`). `--value=token_reward` expects a transformer checkpoint whose
    `reward_linear` head was trained on the Sec. III-D progress labels (see README, "Implementation details").
    """

    def __call__(self, states: Sequence[Any]) -> np.ndarray:
        rewards = [getattr(s, "reward", None) for s in states]
        if any(r is None for r in rewards):
            raise ValueError("TokenRewardValue needs states with a `reward` attribute (enable the WM reward head).")
        return np.asarray(rewards, dtype=np.float64)
