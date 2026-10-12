"""SpatialVLA in SimplerEnv with VLA-Reasoner (paper Table I, "SpatialVLA + VLA-Reasoner").

Built on `SpatialVLAInference` of github.com/DelinQu/SimplerEnv-OpenVLA@ccfe380
(simpler_env/policies/spatialvla/spatialvla_model.py), the evaluation code referenced by the SpatialVLA authors.
Its `step()` predicts an action chunk, temporally ensembles it and post-processes one action. The paper instead
executes 4-action chunks with SpatialVLA (Sec. IV-A), so this subclass splits `step()` into
  (1) `_predict_chunk`: raw chunk prediction (copied from upstream `step()`),
  (2) the reasoner on the flattened executed chunk,
  (3) `_postprocess`: upstream post-processing (copied from upstream `step()`), applied per executed action.
Temporal ensembling is disabled when exec_chunk > 1.

Portions adapted from SimplerEnv-OpenVLA, Copyright (c) 2024 simpler-env, MIT License (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


def make_spatialvla_policy(
    saved_model_path: str = "IPEC-COMMUNITY/spatialvla-4b-224-pt",
    policy_setup: str = "widowx_bridge",
    unnorm_key: Optional[str] = None,
    exec_chunk: int = 4,
    reasoner: Optional[object] = None,
):
    """Returns a SimplerEnv model: reset(instruction), step(image, instruction) -> (raw_action, action)."""
    import torch
    from simpler_env.policies.spatialvla.spatialvla_model import SpatialVLAInference
    from transforms3d.euler import euler2axangle

    class ReasonedSpatialVLAInference(SpatialVLAInference):
        def __init__(self, reasoner=None, exec_chunk=4, **kwargs):
            super().__init__(**kwargs)
            self.reasoner = reasoner
            self.exec_chunk = exec_chunk
            if exec_chunk > 1:
                self.action_ensemble = False
            self._queue: deque = deque()

        def reset(self, task_description: str) -> None:
            super().reset(task_description)
            self._queue.clear()
            if self.reasoner is not None:
                self.reasoner.reset()

        def _predict_chunk(self, image: np.ndarray) -> np.ndarray:
            # --- copied from upstream SpatialVLAInference.step (prediction part) ---
            image = self._resize_image(image)
            self._add_image_to_history(image)
            images = self._obtain_image_history()
            inputs = self.processor(images=images, text=self.task_description, unnorm_key=self.unnorm_key,
                                    return_tensors="pt", do_normalize=False)
            with torch.no_grad():
                if hasattr(self.processor, "action_tokenizer"):
                    generation_outputs = self.vla.predict_action(inputs)
                    raw_actions = self.processor.decode_actions(
                        generation_outputs=generation_outputs, unnorm_key=self.unnorm_key)["actions"]
                else:
                    raw_actions = self.vla.predict_action(**inputs)["actions"]
                    raw_actions = raw_actions.cpu().numpy()
            if self.action_ensemble:
                raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]
            # --- end of copied code ---
            return np.asarray(raw_actions, dtype=np.float64)[: self.exec_chunk]

        def _postprocess(self, a: np.ndarray):
            raw_action = {"world_vector": a[:3], "rotation_delta": a[3:6], "open_gripper": a[6:7]}
            # --- copied from upstream SpatialVLAInference.step (post-processing part) ---
            action = {}
            action["world_vector"] = raw_action["world_vector"] * self.action_scale
            roll, pitch, yaw = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action["rot_axangle"] = action_rotation_ax * action_rotation_angle * self.action_scale
            if self.policy_setup == "google_robot":
                current_gripper_action = raw_action["open_gripper"]
                if self.previous_gripper_action is None:
                    relative_gripper_action = np.array([0])
                    self.previous_gripper_action = current_gripper_action
                else:
                    relative_gripper_action = self.previous_gripper_action - current_gripper_action
                if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                    self.sticky_action_is_on = True
                    self.sticky_gripper_action = relative_gripper_action
                    self.previous_gripper_action = current_gripper_action
                if self.sticky_action_is_on:
                    self.gripper_action_repeat += 1
                    relative_gripper_action = self.sticky_gripper_action
                if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                    self.sticky_action_is_on = False
                    self.gripper_action_repeat = 0
                    self.sticky_gripper_action = 0.0
                action["gripper"] = relative_gripper_action
            elif self.policy_setup == "widowx_bridge":
                action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            action["terminate_episode"] = np.array([0.0])
            # --- end of copied code ---
            return raw_action, action

        def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs):
            if task_description is not None and task_description != self.task_description:
                self.reset(task_description)
            assert image.dtype == np.uint8
            if not self._queue:
                chunk = self._predict_chunk(image)  # a_t^VLA, (exec_chunk, 7)
                if self.reasoner is not None:
                    chunk = self.reasoner(chunk, image)  # search on the flattened chunk + inject (Eq. 1)
                self._queue.extend(chunk)
            return self._postprocess(self._queue.popleft())

    return ReasonedSpatialVLAInference(
        reasoner=reasoner, exec_chunk=exec_chunk, saved_model_path=saved_model_path, unnorm_key=unnorm_key,
        policy_setup=policy_setup,
    )
