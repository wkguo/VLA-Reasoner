"""World models. Only the numpy-only base class is imported eagerly; adapters import torch lazily."""

from vla_reasoner.world_model.base import WorldModel

__all__ = ["WorldModel"]
