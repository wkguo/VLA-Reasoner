"""Value-function interface: states in, values out."""

from __future__ import annotations

from typing import Any, Callable, List

import numpy as np

ValueFunction = Callable[[List[Any]], np.ndarray]


def frame_of(state: Any) -> np.ndarray:
    """Image of a world-model state: `state.frame` if present, else the state itself."""
    frame = getattr(state, "frame", state)
    return np.asarray(frame)
