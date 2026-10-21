"""Tests for episode containers and dataset transforms."""

import numpy as np
import pytest

from vla_reasoner.data import Episode, action_chunks, iter_npz_episodes, pack_for_world_model, save_episode_npz
from vla_reasoner.data.bridge_rlds import binarize_gripper_actions, transform_bridge_episode
from vla_reasoner.data.libero_hdf5 import libero_to_openvla_actions, noop_keep_mask


def _episode(T=10, A=7, success=True):
    images = np.arange(T, dtype=np.uint8)[:, None, None, None] * np.ones((1, 4, 4, 3), np.uint8)
    actions = np.arange(T * A, dtype=np.float32).reshape(T, A)
    return Episode(images, actions, "pick up the cup", success)


def test_npz_roundtrip_and_success_filter(tmp_path):
    save_episode_npz(str(tmp_path / "a.npz"), _episode(success=True))
    save_episode_npz(str(tmp_path / "b.npz"), _episode(T=5, success=False))
    eps = list(iter_npz_episodes(str(tmp_path)))
    assert len(eps) == 2 and eps[0].instruction == "pick up the cup"
    failures = list(iter_npz_episodes(str(tmp_path), success=False))
    assert len(failures) == 1 and len(failures[0]) == 5


def test_action_chunks_sliding_windows():
    chunks = action_chunks(np.arange(12).reshape(6, 2), 4)
    assert chunks.shape == (3, 4, 2)
    np.testing.assert_array_equal(chunks[1], np.arange(2, 10).reshape(4, 2))
    assert action_chunks(np.zeros((3, 2)), 4).shape == (0, 4, 2)


def test_pack_for_world_model_chunked():
    ep = _episode(T=10, A=2)
    frames, packed = pack_for_world_model(ep, 4)
    # full chunks whose resulting frame is observed: (10 - 1) // 4 = 2 -> frames 0, 4, 8
    assert [int(f[0, 0, 0]) for f in frames] == [0, 4, 8]
    np.testing.assert_array_equal(packed[0], ep.actions[0:4].reshape(-1))
    np.testing.assert_array_equal(packed[1], ep.actions[4:8].reshape(-1))
    np.testing.assert_array_equal(packed[2], np.zeros(8))
    f1, a1 = pack_for_world_model(ep, 1)
    assert len(f1) == len(a1) == 10


def test_libero_noop_filter_and_gripper_convention():
    acts = np.array([
        [0, 0, 0, 0, 0, 0, -1],  # first action, arm ~0 -> no-op
        [0.1, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, -1],  # arm ~0, gripper unchanged -> no-op
        [0, 0, 0, 0, 0, 0, 1],  # gripper changes -> kept
    ], dtype=np.float32)
    np.testing.assert_array_equal(noop_keep_mask(acts), [False, True, False, True])
    out = libero_to_openvla_actions(acts)
    np.testing.assert_array_equal(out[:, -1], [1, 1, 1, 0])  # -1 (open) -> 1, +1 (close) -> 0


def test_bridge_binarize_matches_octo_semantics():
    g = np.array([1.0, 0.99, 0.5, 0.3, 0.01, 0.0, 0.6], dtype=np.float32)
    # in-between values take the next decided value; the trailing in-between value keeps its raw value
    np.testing.assert_allclose(binarize_gripper_actions(g), [1, 1, 0, 0, 0, 0, 0.6])


@pytest.mark.parametrize("convention,expected_len,first_index", [("octo", 4, 0), ("bridge_orig", 3, 1)])
def test_bridge_relabel(convention, expected_len, first_index):
    states = np.cumsum(np.ones((5, 7), np.float32), axis=0)
    actions = np.zeros((5, 7), np.float32)
    actions[:, -1] = 1.0
    keep, new = transform_bridge_episode(actions, states, convention)
    assert len(keep) == len(new) == expected_len and keep[0] == first_index
    np.testing.assert_allclose(new[:, :6], 1.0)  # reached-state differences
    np.testing.assert_allclose(new[:, 6], 1.0)
