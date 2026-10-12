"""VLA-Reasoner: test-time online MCTS for frozen Vision-Language-Action models.

The core (KDE prior, MCTS, reasoner wrapper) depends only on numpy. torch, simulators and model code are
imported lazily inside the adapters under `vla_reasoner.value`, `vla_reasoner.world_model` and
`vla_reasoner.policies`.
"""

from vla_reasoner.kde import ActionKDE
from vla_reasoner.mcts import MCTS, Node, SearchResult
from vla_reasoner.reasoner import ReasonerConfig, VLAReasoner, inject
from vla_reasoner.sampling import GaussianNoiseSampler, KDETopKSampler

__version__ = "0.1.0"

__all__ = [
    "ActionKDE",
    "GaussianNoiseSampler",
    "KDETopKSampler",
    "MCTS",
    "Node",
    "ReasonerConfig",
    "SearchResult",
    "VLAReasoner",
    "inject",
]
