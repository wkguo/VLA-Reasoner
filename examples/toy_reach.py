"""Toy sanity check of the VLA-Reasoner machinery (KDE prior + Algorithm 1 MCTS + injection).

THIS IS NOT A PAPER RESULT. It only checks that the search/injection code does what it is supposed to do in a
setting where every component is known exactly:

  * Task: a 2-D point starts at x = 0 near the centre line of a corridor (|y| <= 0.15) and must reach x >= 1
    within 40 steps without leaving the corridor (leaving = failure).
  * Expert demonstrations (50, starting at |y| <= 0.14): a = (0.05, -0.3 y) + noise (move forward, steer back
    to the centre line).
  * Frozen "VLA": a = (0.05, 0.013 - 0.05 y) + noise -- a systematic lateral bias with too-weak feedback, so the
    lateral deviation accumulates step by step until the point leaves the corridor.
  * World model: the true dynamics s' = s + a (no learned model).
  * Value: task progress along the corridor minus the normalized lateral deviation, -1 outside the corridor.
  * KDE: fit on the expert actions only.

The constants were chosen by hand so that the VLA's bias is within reach of the Top-k neighbourhood, which
bounds the per-step correction in this toy (see README, "Toy example").

Run:  python examples/toy_reach.py [--episodes 100] [--alpha 0.6]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vla_reasoner import ActionKDE, KDETopKSampler, ReasonerConfig, VLAReasoner  # noqa: E402
from vla_reasoner.world_model.base import WorldModel  # noqa: E402

GOAL_X, HALF_WIDTH, HORIZON, SPEED = 1.0, 0.15, 40, 0.05


def expert_action(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([SPEED, -0.3 * s[1]]) + rng.normal(0.0, 0.004, size=2)


def vla_action(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([SPEED, 0.013 - 0.05 * s[1]]) + rng.normal(0.0, 0.002, size=2)


def failed(s: np.ndarray) -> bool:
    return abs(s[1]) > HALF_WIDTH


def progress_value(s: np.ndarray) -> float:
    if failed(s):
        return -1.0
    return float(np.clip(s[0] / GOAL_X, 0.0, 1.0) - abs(s[1]) / HALF_WIDTH)


class TrueDynamics(WorldModel):
    """The toy's exact dynamics used as the world model (state = position)."""

    context_length = 1

    def init_state(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        return np.asarray(frames[-1], dtype=np.float64)

    def predict(self, state, actions):
        return [state + a for a in np.asarray(actions)]


def value_fn(states) -> np.ndarray:
    return np.array([progress_value(s) for s in states])


def expert_dataset(num_demos: int = 50, seed: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    actions = []
    for _ in range(num_demos):
        s = np.array([0.0, rng.uniform(-0.14, 0.14)])
        while s[0] < GOAL_X:
            a = expert_action(s, rng)
            actions.append(a)
            s = s + a
    return np.asarray(actions)


def rollout(seed: int, reasoner=None) -> bool:
    rng = np.random.default_rng(seed)
    s = np.array([0.0, rng.uniform(-0.02, 0.02)])
    if reasoner is not None:
        reasoner.reset()
    for _ in range(HORIZON):
        a = vla_action(s, rng)
        if reasoner is not None:
            a = reasoner(a[None], s)[0]
        s = s + a
        if failed(s):
            return False
        if s[0] >= GOAL_X:
            return True
    return False


def run(episodes: int = 100, alpha: float = 0.6, top_k: int = 5, num_kde_samples: int = 50,
        max_depth: int = 2, seed: int = 0) -> dict:
    kde = ActionKDE(bandwidth="scott", standardize=True).fit(expert_dataset())
    config = ReasonerConfig(alpha=alpha, top_k=top_k, num_kde_samples=num_kde_samples, max_depth=max_depth, seed=seed)
    reasoner = VLAReasoner(TrueDynamics(), value_fn, KDETopKSampler(kde, top_k, num_kde_samples), config)
    seeds = range(seed, seed + episodes)
    base = [rollout(s) for s in seeds]
    reasoned = [rollout(s, reasoner) for s in seeds]
    return {"episodes": episodes, "base_success": float(np.mean(base)), "reasoner_success": float(np.mean(reasoned))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kde_samples", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = run(args.episodes, args.alpha, args.top_k, args.num_kde_samples, args.max_depth, args.seed)
    print(f"toy corridor, {out['episodes']} episodes (seeds {args.seed}..{args.seed + args.episodes - 1}), "
          f"alpha={args.alpha}: base VLA success {out['base_success']:.2f} | "
          f"+ reasoner success {out['reasoner_success']:.2f}")


if __name__ == "__main__":
    main()
