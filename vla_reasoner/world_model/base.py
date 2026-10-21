"""World-model interface used by the tree search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence

import numpy as np


class WorldModel(ABC):
    #: number of most recent real observations needed to build the root state
    context_length: int = 1

    @abstractmethod
    def init_state(self, frames: Sequence[np.ndarray]) -> Any:
        """Build the root state s(0) = s_t from the last `context_length` real frames (oldest first)."""

    @abstractmethod
    def predict(self, state: Any, actions: np.ndarray) -> List[Any]:
        """Simulation step (Alg. 1 line 5) for every candidate action of one expansion.

        Args:
            state: the state s(d) of the node being expanded.
            actions: (k, d) flattened candidate actions A^Top-k.
        Returns:
            list of k next states [W(s(d), a_j)]_j.
        """

    def end_search(self) -> None:
        """Hook called once after every tree search (no-op by default)."""
