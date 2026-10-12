import numpy as np
import pytest

from vla_reasoner.reasoner import ReasonerConfig, VLAReasoner, inject
from vla_reasoner.world_model.base import WorldModel


class ContextWM(WorldModel):
    def __init__(self, context_length=1):
        self.context_length = context_length
        self.init_calls, self.predict_dims, self.end_calls = [], [], 0

    def init_state(self, frames):
        self.init_calls.append(list(frames))
        return 0.0

    def predict(self, state, actions):
        self.predict_dims.append(actions.shape[1])
        return [state + float(a.sum()) for a in actions]

    def end_search(self):
        self.end_calls += 1


class OffsetSampler:
    """Deterministic candidates: reference + j * 0.1 (j = 1..k); equal densities."""

    k = 3

    def __call__(self, reference, rng):
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        return reference[None] + 0.1 * np.arange(1, 4)[:, None], np.zeros(3)


def value_fn(states):
    return np.asarray(states, dtype=np.float64)  # larger action sum = better


def test_inject_formula():
    a_vla, a_r = np.array([1.0, -2.0]), np.array([3.0, 4.0])
    np.testing.assert_allclose(inject(a_vla, a_r, 0.6), 0.6 * a_vla + 0.4 * a_r)
    np.testing.assert_allclose(inject(a_vla, a_r, 1.0), a_vla)
    np.testing.assert_allclose(inject(a_vla, a_r, 0.0), a_r)


def test_alpha_one_is_vanilla_vla_without_search():
    wm = ContextWM()
    r = VLAReasoner(wm, value_fn, OffsetSampler(), ReasonerConfig(alpha=1.0, max_depth=0))
    a = np.array([[0.5, 0.5]])
    np.testing.assert_allclose(r(a, "frame"), a)
    assert wm.init_calls == []


def test_reasoner_injects_search_result():
    wm = ContextWM()
    r = VLAReasoner(wm, value_fn, OffsetSampler(), ReasonerConfig(alpha=0.6, max_depth=0))
    a_vla = np.array([[0.0, 0.0]])
    out = r(a_vla, "frame")
    # max_depth=0: best root child = largest one-step value = reference + 0.3
    np.testing.assert_allclose(r.last_result.action, [0.3, 0.3])
    np.testing.assert_allclose(out, 0.6 * a_vla + 0.4 * np.array([[0.3, 0.3]]))
    assert wm.end_calls == 1


def test_search_interval():
    wm = ContextWM()
    r = VLAReasoner(wm, value_fn, OffsetSampler(), ReasonerConfig(search_interval=3, max_depth=0))
    a = np.zeros((1, 2))
    searched = []
    for _ in range(7):
        out = r(a, "f")
        searched.append(not np.allclose(out, a))
    assert searched == [True, False, False, True, False, False, True]
    assert r.stats["num_searches"] == 3
    r.reset()
    assert not np.allclose(r(a, "f"), a)  # the schedule restarts every episode


def test_chunk_is_flattened_and_restored():
    wm = ContextWM()
    r = VLAReasoner(wm, value_fn, OffsetSampler(), ReasonerConfig(max_depth=1))
    chunk = np.random.default_rng(0).normal(size=(4, 7))
    out = r(chunk, "f")
    assert out.shape == (4, 7)
    assert wm.predict_dims == [28, 28]  # MCTS sees one flattened 28-d action per candidate


def test_context_frames_are_padded_then_rolled():
    wm = ContextWM(context_length=2)
    r = VLAReasoner(wm, value_fn, OffsetSampler(), ReasonerConfig(max_depth=0))
    for f in ["f0", "f1", "f2"]:
        r(np.zeros((1, 2)), f)
    assert wm.init_calls == [["f0", "f0"], ["f0", "f1"], ["f1", "f2"]]


def test_invalid_config():
    with pytest.raises(ValueError):
        ReasonerConfig(alpha=1.5)
    with pytest.raises(ValueError):
        ReasonerConfig(search_interval=0)
