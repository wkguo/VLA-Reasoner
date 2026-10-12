"""Dataset loaders. Heavy readers (h5py, tensorflow_datasets) are imported inside the loader functions."""

from typing import Iterator

from vla_reasoner.data.episode import (
    Episode,
    action_chunks,
    iter_npz_episodes,
    load_episode_npz,
    pack_for_world_model,
    save_episode_npz,
)


def iter_episodes(data_format: str, path: str, **kwargs) -> Iterator[Episode]:
    """Dispatch on the dataset format: "npz", "libero_hdf5" or "bridge_rlds"."""
    if data_format == "npz":
        return iter_npz_episodes(path, **kwargs)
    if data_format == "libero_hdf5":
        from vla_reasoner.data.libero_hdf5 import iter_libero_hdf5

        return iter_libero_hdf5(path, **kwargs)
    if data_format == "bridge_rlds":
        from vla_reasoner.data.bridge_rlds import iter_bridge_rlds

        return iter_bridge_rlds(path, **kwargs)
    raise ValueError(f"Unknown data format {data_format!r}.")


__all__ = [
    "Episode",
    "action_chunks",
    "iter_episodes",
    "iter_npz_episodes",
    "load_episode_npz",
    "pack_for_world_model",
    "save_episode_npz",
]
