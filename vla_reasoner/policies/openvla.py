"""OpenVLA as the frozen VLA."""
# Portions adapted from OpenVLA (Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, MIT); see THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

from vla_reasoner.policies.base import VLAPolicy


def center_crop_and_resize(image: np.ndarray, crop_scale: float = 0.9, size: int = 224) -> np.ndarray:
    """Copy of OpenVLA's `crop_and_resize` (TF ops, to match the training-time augmentation exactly)."""
    import tensorflow as tf

    img = tf.image.convert_image_dtype(tf.convert_to_tensor(image), tf.float32)[None]
    side = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(1,))
    offset = (1 - side) / 2
    boxes = tf.stack([offset, offset, offset + side, offset + side], axis=1)
    img = tf.image.crop_and_resize(img, boxes, tf.range(1), (size, size))[0]
    img = tf.image.convert_image_dtype(tf.clip_by_value(img, 0, 1), tf.uint8, saturate=True)
    return img.numpy()


class OpenVLAPolicy(VLAPolicy):
    action_dim = 7
    exec_chunk = 1

    def __init__(
        self,
        checkpoint: str,
        unnorm_key: str,
        center_crop: bool = True,
        device: str = "cuda:0",
        attn_implementation: Optional[str] = "flash_attention_2",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._torch = torch
        self.device = torch.device(device)
        self.center_crop = center_crop
        kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
                      load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.vla = AutoModelForVision2Seq.from_pretrained(checkpoint, **kwargs)
        if not (load_in_8bit or load_in_4bit):
            self.vla = self.vla.to(self.device)
        stats_path = os.path.join(checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats_path):  # local fine-tuned checkpoints keep their norm stats here
            with open(stats_path) as f:
                self.vla.norm_stats = json.load(f)
        # As in run_libero_eval.py: checkpoints trained on the no-op-free data use the "_no_noops" key.
        if unnorm_key not in self.vla.norm_stats and f"{unnorm_key}_no_noops" in self.vla.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        if unnorm_key not in self.vla.norm_stats:
            raise KeyError(f"Action un-norm key {unnorm_key} not found in the checkpoint's norm_stats.")
        self.unnorm_key = unnorm_key
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.instruction = ""

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def predict(self, observation: Dict[str, Any]) -> np.ndarray:
        """observation["full_image"]: (224, 224, 3) uint8, already preprocessed like at training time."""
        from PIL import Image

        image = np.asarray(observation["full_image"])
        if self.center_crop:
            image = center_crop_and_resize(image, 0.9, image.shape[0])
        prompt = f"In: What action should the robot take to {self.instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=self._torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=np.float64).reshape(1, self.action_dim)
