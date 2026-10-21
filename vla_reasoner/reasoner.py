"""VLA-Reasoner wrapper: propose with the frozen VLA, search, and inject the result (Eq. 1)."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from vla_reasoner.mcts import MCTS, SearchResult
from vla_reasoner.sampling import CandidateSampler
from vla_reasoner.value.base import ValueFunction
from vla_reasoner.world_model.base import WorldModel


@dataclass
class ReasonerConfig:
    alpha: float = 0.6  # injection strength (Eq. 1); paper default 0.6, alpha = 1.0 is the vanilla VLA
    max_depth: int = 2  # MaxDepth of Alg. 1 (loop d = 0..MaxDepth)
    top_k: int = 5  # k, candidates per expansion
    num_kde_samples: int = 1000  # N, KDE samples per expansion (N large, k small)
    c_ucb: float = 1.0 / math.sqrt(2.0)  # c in Eq. 5 (the paper's example value)
    search_interval: int = 1  # search every n-th VLA query (search is run sparsely, Sec. III-B)
    seed: int = 0

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if self.search_interval < 1:
            raise ValueError("search_interval must be >= 1.")


def inject(a_vla: np.ndarray, a_reasoner: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. 1: a_t = alpha * a^VLA + (1 - alpha) * a^Reasoner."""
    return alpha * np.asarray(a_vla, dtype=np.float64) + (1.0 - alpha) * np.asarray(a_reasoner, dtype=np.float64)


class VLAReasoner:
    def __init__(
        self,
        world_model: WorldModel,
        value_fn: ValueFunction,
        sampler: CandidateSampler,
        config: Optional[ReasonerConfig] = None,
    ):
        self.config = config or ReasonerConfig()
        self.world_model = world_model
        self.mcts = MCTS(world_model, value_fn, sampler, self.config.max_depth, self.config.c_ucb)
        self.rng = np.random.default_rng(self.config.seed)
        self._history: deque = deque(maxlen=max(1, int(getattr(world_model, "context_length", 1))))
        self._num_queries = 0
        self.last_result: Optional[SearchResult] = None
        self.stats = {"num_queries": 0, "num_searches": 0, "num_simulations": 0, "search_time_s": 0.0}

    def reset(self) -> None:
        """Call at the start of every episode."""
        self._history.clear()
        self._num_queries = 0
        self.last_result = None

    def _context_frames(self) -> list:
        frames = list(self._history)
        # At the beginning of an episode there are fewer real frames than the world model's context:
        # pad by repeating the oldest one.
        return [frames[0]] * (self._history.maxlen - len(frames)) + frames

    def __call__(self, vla_actions: np.ndarray, frame: Any) -> np.ndarray:
        """Return the action(s) to execute for one VLA query.

        Args:
            vla_actions: the frozen VLA's raw proposal for the executed part, (A,) or (exec_chunk, A).
            frame: the current real observation used as world-model context (e.g. an RGB image).
        Returns:
            array with the same shape as `vla_actions`.
        """
        a_vla = np.asarray(vla_actions, dtype=np.float64)
        self._history.append(frame)
        query = self._num_queries
        self._num_queries += 1
        self.stats["num_queries"] += 1
        self.last_result = None
        if self.config.alpha >= 1.0 or query % self.config.search_interval != 0:
            return a_vla.copy()

        t0 = time.perf_counter()
        root_state = self.world_model.init_state(self._context_frames())
        try:
            result = self.mcts.search(root_state, a_vla.reshape(-1), self.rng)
        finally:
            self.world_model.end_search()
        self.last_result = result
        self.stats["num_searches"] += 1
        self.stats["num_simulations"] += result.num_simulations
        self.stats["search_time_s"] += time.perf_counter() - t0
        return inject(a_vla.reshape(-1), result.action, self.config.alpha).reshape(a_vla.shape)
