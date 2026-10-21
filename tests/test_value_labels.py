"""Tests for progress labels and keyframe selection."""

import numpy as np
import pytest

from vla_reasoner.value.labels import progress_labels, select_keyframes, value_samples


def test_paper_example_5_of_9():
    labels = progress_labels(10)
    assert labels[5] == pytest.approx(5 / 9)  # "in a 10-frame sequence, the 5th frame is assigned 5/9"
    assert labels[0] == 0.0 and labels[-1] == 1.0
    assert np.all(np.diff(labels) > 0)


def test_single_frame_label():
    np.testing.assert_allclose(progress_labels(1), [1.0])


def _frames(levels):
    return np.stack([np.full((32, 32, 3), v, dtype=np.uint8) for v in levels])


def test_visual_downsampling_drops_static_frames():
    # 3 static frames, a change, 2 static, a change, then static until the end
    frames = _frames([0, 0, 0, 100, 100, 100, 200, 200, 200])
    idx = select_keyframes(frames, mode="visual", threshold=0.05)
    assert list(idx) == [0, 3, 6, 8]  # first, the two change points, and the last frame


def test_visual_threshold_is_relative_to_last_kept_frame():
    # slow drift: each step differs by ~2/255 but accumulates past the threshold
    frames = _frames(np.arange(0, 40, 2))
    idx = select_keyframes(frames, mode="visual", threshold=9 / 255)
    np.testing.assert_array_equal(idx, [0, 5, 10, 15, 19])


def test_stride_downsampling_keeps_last():
    idx = select_keyframes(_frames(range(10)), mode="stride", stride=4)
    assert list(idx) == [0, 4, 8, 9]


def test_value_samples_labels_follow_kept_frames():
    frames = _frames([0, 0, 0, 100, 100, 100, 200, 200, 200])
    idx, labels = value_samples(frames, mode="visual", threshold=0.05)
    np.testing.assert_allclose(labels, [0.0, 1 / 3, 2 / 3, 1.0])
    assert len(idx) == len(labels)
