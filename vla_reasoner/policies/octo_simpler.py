"""Octo in SimplerEnv with VLA-Reasoner (paper Table I, "Octo-Small + VLA-Reasoner").

Built on `OctoInference` of the SimplerEnv fork used by V-GPS (github.com/nakamotoo/SimplerEnv@f343c82,
simpler_env/policies/octo/octo_model.py). In that file `step()` calls `get_action(image)`, which returns the
raw action dict (un-normalized, after Octo's temporal action ensembling), and then post-processes it
(Euler -> axis-angle, sticky gripper). We override only `get_action`: the reasoner edits the raw 7-d action
[world_vector(3), rotation_delta(3), open_gripper(1)] and the original post-processing in `step()` is reused.
Octo executes one action per step (Sec. IV-A).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def raw_action_to_vector(raw_action: dict) -> np.ndarray:
    return np.concatenate([raw_action["world_vector"], raw_action["rotation_delta"], raw_action["open_gripper"]])


def vector_to_raw_action(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    return {"world_vector": a[:3], "rotation_delta": a[3:6], "open_gripper": a[6:7]}


def make_octo_policy(
    model_type: str = "octo-small",
    policy_setup: str = "widowx_bridge",
    init_rng: int = 0,
    sticky_step: int = 3,
    reasoner: Optional[object] = None,
):
    """Returns a SimplerEnv model with the V-GPS interface: reset(instruction), step(image) -> (raw, action)."""
    from simpler_env.policies.octo.octo_model import OctoInference

    class ReasonedOctoInference(OctoInference):
        def __init__(self, reasoner=None, **kwargs):
            super().__init__(**kwargs)
            self.reasoner = reasoner

        def reset(self, task_description: str) -> None:
            super().reset(task_description)
            if self.reasoner is not None:
                self.reasoner.reset()

        def get_action(self, image: np.ndarray) -> dict:
            raw_action = super().get_action(image)  # a_t^VLA (raw, pre-post-processing)
            if self.reasoner is None:
                return raw_action
            a = self.reasoner(raw_action_to_vector(raw_action)[None], image)[0]  # search + inject (Eq. 1)
            return vector_to_raw_action(a)

    return ReasonedOctoInference(
        reasoner=reasoner, model_type=model_type, policy_setup=policy_setup, init_rng=init_rng, sticky_step=sticky_step
    )
