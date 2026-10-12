"""DEBUG ONLY -- NOT part of VLA-Reasoner.

A "perfect" world model that uses the simulator itself through state save/restore. It exists to debug the
search / value / injection pipeline independently of world-model quality (e.g. to check that a trained value
function ranks branches sensibly). Results obtained with it must not be reported as VLA-Reasoner results: the
paper's method uses a learned video world model (iVideoGPT) and never queries the simulator during search.

Caveats: restoring physics state does not restore every wrapper-side counter (time limits, success latches,
controller targets); the simulator is restored to the root snapshot after each search (`end_search`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np

from vla_reasoner.world_model.base import WorldModel


@dataclass(eq=False)
class OracleState:
    sim_state: Any
    frame: np.ndarray


class SimulatorOracleWorldModel(WorldModel):
    """DEBUG ONLY. W(s, a) = simulator step from a restored state."""

    context_length = 1

    def __init__(
        self,
        get_state: Callable[[], Any],
        set_state: Callable[[Any], Any],
        step: Callable[[np.ndarray], np.ndarray],
        action_dim: int,
    ):
        """
        Args:
            get_state: returns a snapshot of the simulator.
            set_state: restores a snapshot.
            step: executes ONE raw (pre-post-processing) action of size `action_dim` in the simulator and
                returns the resulting frame (H, W, 3) uint8; it must apply the benchmark's post-processing.
            action_dim: size of one action; flattened chunks are executed action by action.
        """
        self._get_state, self._set_state, self._step = get_state, set_state, step
        self.action_dim = action_dim
        self._root = None

    def init_state(self, frames: Sequence[np.ndarray]) -> OracleState:
        self._root = self._get_state()
        return OracleState(self._root, np.asarray(frames[-1]))

    def predict(self, state: OracleState, actions: np.ndarray) -> List[OracleState]:
        out = []
        for a in np.asarray(actions):
            self._set_state(state.sim_state)
            frame = None
            for sub in a.reshape(-1, self.action_dim):
                frame = self._step(sub)
            out.append(OracleState(self._get_state(), np.asarray(frame)))
        return out

    def end_search(self) -> None:
        if self._root is not None:
            self._set_state(self._root)
            self._root = None


def libero_oracle(env, action_to_env: Callable, get_frame: Callable, action_dim: int = 7):
    """DEBUG ONLY. LIBERO `OffScreenRenderEnv` via `get_sim_state()` / `set_init_state()`."""

    def step(a):
        obs, _, _, _ = env.step(action_to_env(a))
        return get_frame(obs)

    return SimulatorOracleWorldModel(env.get_sim_state, env.set_init_state, step, action_dim)


def maniskill2_oracle(env, action_to_env: Callable, get_frame: Callable, action_dim: int = 7):
    """DEBUG ONLY. SimplerEnv / ManiSkill2 via `get_state()` / `set_state()` on the unwrapped env.

    The unwrapped env is stepped so that the TimeLimit wrapper's step counter is not advanced by the search.
    """
    base = env.unwrapped

    def step(a):
        obs, *_ = base.step(action_to_env(a))
        return get_frame(base, obs)

    return SimulatorOracleWorldModel(base.get_state, base.set_state, step, action_dim)
