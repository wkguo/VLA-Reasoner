"""Online Monte Carlo Tree Search of VLA-Reasoner (paper Sec. III-B, Algorithm 1, Eqs. 2, 4, 5).

Algorithm 1:

    o(0): s(0) <- s_t, a(0) <- a_t^VLA
    for d = 0 .. MaxDepth:                          # inclusive: MaxDepth + 1 expansions
        A^Top-k <- TopK(pi_KDE, a(d))               # Expansion, Eq. 2
        for a in A^Top-k:
            s_hat <- W(s(d), a)                     # Simulation, Eq. 3 (batched over A^Top-k)
            v_hat <- MLP(s_hat)                     # Evaluation        (batched over A^Top-k)
            Update(o(d), v_hat)                     # Backpropagation, Eq. 4, from o(d) up to the root
        o(d) <- SelectUCB(o(d))                     # Selection, Eq. 5: the child expanded next
        a(d) <- a(o(d))
    a^Reasoner <- argmax_{o_i in children(root)} Q(o_i)

Visit counts (Eq. 4 and Sec. III-C) are not obtained by repeated simulation; they are pseudo-counts derived
from the KDE density, N(a) proportional to p_KDE(a). Per expansion we normalize the k sibling densities to
n_j = k * p_j / sum_l p_l, so an expanded node has N(o_i) = sum_{a in A^Top-k_i} N(a) = k. The Q backup of
Eq. 4 is invariant to the scale of the counts but Eq. 5 is not, hence this normalization. A leaf's count is
the pseudo-count of the action that created it, N(a_i).
The root's own value is never evaluated by Algorithm 1, so the root has no self term in Eq. 4.

Only numpy is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from vla_reasoner.sampling import CandidateSampler
from vla_reasoner.value.base import ValueFunction
from vla_reasoner.world_model.base import WorldModel

DEFAULT_C_UCB = 1.0 / math.sqrt(2.0)  # the example constant given after Eq. 5


@dataclass(eq=False)
class Node:
    """Tree node o_i with action a_i (leading into it), state s_i, value estimate v_i and Q(o_i)."""

    action: np.ndarray
    state: Any
    value: Optional[float] = None  # v_i; None for the root
    prior_count: float = 0.0  # N(a_i): density pseudo-count of the action leading into this node
    log_density: float = float("nan")
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    q: float = 0.0
    depth: int = 0

    @property
    def visit_count(self) -> float:
        """N(o_i): Eq. 4 (sum over the expanded Top-k actions) if expanded, else N(a_i)."""
        if self.children:
            return float(sum(c.prior_count for c in self.children))
        return float(self.prior_count)


def pseudo_counts(log_densities: np.ndarray, total: Optional[float] = None) -> np.ndarray:
    """N(a_j) proportional to p(a_j), normalized to sum to `total` (default: k = number of candidates).

    Computed from log-densities (softmax) so that tiny densities do not underflow.
    """
    log_p = np.asarray(log_densities, dtype=np.float64).reshape(-1)
    total = float(len(log_p)) if total is None else float(total)
    finite = np.isfinite(log_p)
    if not finite.any():
        return np.full(len(log_p), total / len(log_p))
    w = np.where(finite, np.exp(np.where(finite, log_p, 0.0) - log_p[finite].max()), 0.0)
    return total * w / w.sum()


def eq4_q(node: Node) -> float:
    """Eq. 4: Q(o_i) = (N(o_i) v_i + sum_j N(o_j) Q(o_j)) / (N(o_i) + sum_j N(o_j))."""
    num = den = 0.0
    if node.value is not None:
        n_i = node.visit_count
        num += n_i * node.value
        den += n_i
    for child in node.children:
        n_j = child.visit_count
        num += n_j * child.q
        den += n_j
    if den > 0:
        return num / den
    return float(node.value) if node.value is not None else 0.0


def backpropagate(node: Optional[Node]) -> None:
    """Recompute Q with Eq. 4 bottom-up, from `node` to the root."""
    while node is not None:
        node.q = eq4_q(node)
        node = node.parent


def ucb_scores(node: Node, c: float = DEFAULT_C_UCB) -> np.ndarray:
    """Eq. 5 for each child o_i of `node`: Q(o_i) + c * sqrt(ln N(parent) / (1 + N(o_i)))."""
    n_parent = node.visit_count
    log_n = math.log(n_parent) if n_parent > 1.0 else 0.0  # guard: ln N < 0 is undefined under sqrt
    return np.array([ch.q + c * math.sqrt(log_n / (1.0 + ch.visit_count)) for ch in node.children])


def select_ucb(node: Node, c: float = DEFAULT_C_UCB) -> Node:
    """SelectUCB (Alg. 1 line 9): the child with the highest Eq. 5 score (first one on ties)."""
    if not node.children:
        raise ValueError("Cannot select among the children of an unexpanded node.")
    return node.children[int(np.argmax(ucb_scores(node, c)))]


@dataclass
class SearchResult:
    action: np.ndarray  # a^Reasoner (flattened)
    root: Node
    best_child: Node
    num_simulations: int


class MCTS:
    def __init__(
        self,
        world_model: WorldModel,
        value_fn: ValueFunction,
        sampler: CandidateSampler,
        max_depth: int = 2,
        c_ucb: float = DEFAULT_C_UCB,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        self.world_model = world_model
        self.value_fn = value_fn
        self.sampler = sampler
        self.max_depth = max_depth
        self.c_ucb = c_ucb

    def search(self, root_state: Any, root_action: np.ndarray, rng: np.random.Generator) -> SearchResult:
        """Run Algorithm 1 from the real state s_t with the VLA proposal a_t^VLA (flattened)."""
        root = Node(action=np.asarray(root_action, dtype=np.float64).reshape(-1), state=root_state)
        node = root
        num_sim = 0
        for _ in range(self.max_depth + 1):  # line 2: for d = 0 to MaxDepth
            actions, log_p = self.sampler(node.action, rng)  # line 3: Expansion, Eq. 2 (Top-k around a(d))
            actions = np.asarray(actions, dtype=np.float64)
            counts = pseudo_counts(log_p)  # Sec. III-C: N(a) proportional to p_KDE(a)
            states = self.world_model.predict(node.state, actions)  # line 5: Simulation (Eq. 3)
            values = np.asarray(self.value_fn(states), dtype=np.float64).reshape(-1)  # line 6: Evaluation
            if not len(states) == len(values) == len(actions):
                raise ValueError("World model / value function returned the wrong number of outputs.")
            num_sim += len(actions)
            for a, s, v, n, lp in zip(actions, states, values, counts, np.asarray(log_p).reshape(-1)):
                child = Node(action=a, state=s, value=float(v), prior_count=float(n), log_density=float(lp),
                             parent=node, q=float(v), depth=node.depth + 1)
                node.children.append(child)
                backpropagate(node)  # line 7: Update(o(d), v_hat) by Eq. 4
            node = select_ucb(node, self.c_ucb)  # lines 9-10: Selection by Eq. 5
        best = root.children[int(np.argmax([ch.q for ch in root.children]))]  # line 12
        return SearchResult(action=best.action.copy(), root=root, best_child=best, num_simulations=num_sim)
