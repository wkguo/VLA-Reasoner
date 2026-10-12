import math

import numpy as np
import pytest

from vla_reasoner.mcts import MCTS, Node, backpropagate, eq4_q, pseudo_counts, select_ucb, ucb_scores
from vla_reasoner.world_model.base import WorldModel

C = 1 / math.sqrt(2)


def _child(parent, prior, value, action=0.0):
    n = Node(action=np.array([action]), state=None, value=value, prior_count=prior, parent=parent, q=value,
             depth=parent.depth + 1)
    parent.children.append(n)
    return n


# --------------------------------------------------------------- visit counts (Sec. III-C; N(o_i) of Eq. 4)
def test_pseudo_counts_proportional_to_density_and_sum_to_k():
    counts = pseudo_counts(np.log([1.0, 3.0]))
    np.testing.assert_allclose(counts, [0.5, 1.5])
    counts = pseudo_counts(np.log([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(counts, [0.75, 0.75, 1.5])


def test_pseudo_counts_numerically_stable():
    np.testing.assert_allclose(pseudo_counts(np.array([-1e4, -1e4 + math.log(3.0)])), [0.5, 1.5])
    np.testing.assert_allclose(pseudo_counts(np.array([-np.inf, 0.0])), [0.0, 2.0])
    np.testing.assert_allclose(pseudo_counts(np.array([-np.inf, -np.inf])), [1.0, 1.0])


def test_visit_count_eq4():
    root = Node(action=np.zeros(1), state=None)
    c1, c2 = _child(root, 1.0, 0.2), _child(root, 3.0, 0.6)
    assert root.visit_count == 4.0  # N(o_i) = sum over its Top-k actions of N(a)
    assert c1.visit_count == 1.0  # leaf: own pseudo-count
    _child(c2, 2.0, 1.0), _child(c2, 2.0, 0.0)
    assert c2.visit_count == 4.0


# ---------------------------------------------------------------------------------------------------- Eq. 4
def test_eq4_hand_computed_tree():
    root = Node(action=np.zeros(1), state=None)  # root value is never evaluated -> no self term
    c1, c2 = _child(root, 1.0, 0.2), _child(root, 3.0, 0.6)
    _child(c2, 2.0, 1.0), _child(c2, 2.0, 0.0)
    backpropagate(c2)
    # Q(c2) = (N(c2) v_c2 + 2*1.0 + 2*0.0) / (N(c2) + 2 + 2), N(c2) = 4
    assert c2.q == pytest.approx((4 * 0.6 + 2 * 1.0 + 2 * 0.0) / (4 + 4))  # 0.55
    # Q(root) = (N(c1) Q(c1) + N(c2) Q(c2)) / (N(c1) + N(c2))
    assert root.q == pytest.approx((1 * 0.2 + 4 * 0.55) / 5)  # 0.48
    assert c1.q == pytest.approx(0.2) and eq4_q(c1) == pytest.approx(0.2)  # leaf: Q = v


# ---------------------------------------------------------------------------------------------------- Eq. 5
def test_ucb_scores_hand_computed():
    root = Node(action=np.zeros(1), state=None)
    _child(root, 1.0, 0.5), _child(root, 3.0, 0.45)
    ln_n = math.log(4.0)
    expected = [0.5 + C * math.sqrt(ln_n / 2.0), 0.45 + C * math.sqrt(ln_n / 4.0)]
    np.testing.assert_allclose(ucb_scores(root, C), expected)
    assert select_ucb(root, C) is root.children[0]


def test_ucb_exploration_can_override_q():
    root = Node(action=np.zeros(1), state=None)
    a, b = _child(root, 3.0, 0.5), _child(root, 1.0, 0.45)  # b has lower Q but a smaller count
    assert select_ucb(root, C) is b
    assert select_ucb(root, 0.0) is a  # pure exploitation


# ---------------------------------------------------------------------------------------------------- Alg. 1
class RecordingWM(WorldModel):
    def __init__(self):
        self.calls = []

    def init_state(self, frames):
        return "s0"

    def predict(self, state, actions):
        self.calls.append((state, actions.copy()))
        return [f"{state}|{a[0]:g}" for a in actions]


class RecordingSampler:
    k = 3

    def __init__(self):
        self.refs = []

    def __call__(self, reference, rng):
        self.refs.append(np.asarray(reference).copy())
        return reference[None] + np.array([[1.0], [2.0], [3.0]]), np.log([1.0, 1.0, 2.0])  # counts .75 .75 1.5


VALUES = {
    "s0|1": 0.5, "s0|2": 0.4, "s0|3": 0.1,
    "s0|1|2": 0.0, "s0|1|3": 0.0, "s0|1|4": 0.0,
    "s0|1|2|3": 0.9, "s0|1|2|4": 0.9, "s0|1|2|5": 0.9,
}


class RecordingValue:
    def __init__(self):
        self.calls = []

    def __call__(self, states):
        self.calls.append(list(states))
        return np.array([VALUES[s] for s in states])


def test_algorithm1_control_flow():
    wm, sampler, value = RecordingWM(), RecordingSampler(), RecordingValue()
    result = MCTS(wm, value, sampler, max_depth=2, c_ucb=C).search("s0", np.array([0.0]), np.random.default_rng(0))

    # for d = 0..MaxDepth: one expansion per depth, k world-model calls each (batched)
    assert [s for s, _ in wm.calls] == ["s0", "s0|1", "s0|1|2"]
    # Expansion around a(d): a(0) = a_VLA, then the action of the UCB-selected child
    np.testing.assert_allclose(np.concatenate(sampler.refs), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(wm.calls[0][1].ravel(), [1, 2, 3])
    np.testing.assert_allclose(wm.calls[1][1].ravel(), [2, 3, 4])
    # the value function evaluates exactly the predicted states
    assert value.calls[0] == ["s0|1", "s0|2", "s0|3"]
    assert value.calls[2] == ["s0|1|2|3", "s0|1|2|4", "s0|1|2|5"]
    assert result.num_simulations == 9

    root = result.root
    c1, c2, c3 = root.children
    # depth-0 selection by Eq. 5 chose c1 (scores 1.060 / 0.960 / 0.569)
    assert [len(c.children) for c in root.children] == [3, 0, 0]
    g1 = c1.children[0]
    assert g1.q == pytest.approx((3 * 0.0 + 3 * 0.9) / 6)  # 0.45
    assert c1.q == pytest.approx((3 * 0.5 + 3 * 0.45 + 0.75 * 0.0 + 1.5 * 0.0) / (3 + 3 + 0.75 + 1.5))  # 0.3455
    assert c2.q == pytest.approx(0.4) and c3.q == pytest.approx(0.1)
    # a^Reasoner = argmax over the ROOT's children of Q -- here c2, not the child on the searched path
    assert result.best_child is c2
    np.testing.assert_allclose(result.action, [2.0])


def test_max_depth_zero_is_one_expansion():
    wm, sampler, value = RecordingWM(), RecordingSampler(), RecordingValue()
    result = MCTS(wm, value, sampler, max_depth=0).search("s0", np.array([0.0]), np.random.default_rng(0))
    assert len(wm.calls) == 1
    np.testing.assert_allclose(result.action, [1.0])  # highest one-step value


def test_wrong_number_of_outputs_raises():
    class BadValue:
        def __call__(self, states):
            return np.zeros(len(states) - 1)

    with pytest.raises(ValueError):
        MCTS(RecordingWM(), BadValue(), RecordingSampler()).search("s0", np.array([0.0]), np.random.default_rng(0))
