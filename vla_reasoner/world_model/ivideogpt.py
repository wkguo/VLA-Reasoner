"""Action-conditioned world model W on top of the official iVideoGPT code (Wu et al., NeurIPS 2024).

The paper follows the iVideoGPT architecture (~600M parameters; Sec. IV-A) and embeds robot actions into its
latent space (Sec. III-B(b)). iVideoGPT is used as an external dependency (github.com/thuml/iVideoGPT, pinned to
commit d601d5c in requirements/ivideogpt.txt); it is not vendored. Put the repository on PYTHONPATH or pass
`ivideogpt_root`.

One simulation step W(s, a) follows the per-step rollout of upstream `mbrl/video_predictor.py`
(`VideoPredictor.rollout`):
  1. the real context frames are tokenized by the compressive tokenizer (16x16 tokens + 1 separator each);
  2. the action is embedded by the transformer's `action_linear` and added to the embedding of the slot token
     that precedes the next frame's tokens (`HeadModelWithAction`);
  3. 16 dynamics tokens (+1 slot) are sampled (top-k sampling, as upstream);
  4. the new frame is decoded from the context tokens + the 16 new tokens.
Each tree node keeps its own embedding sequence, so all children of a node share the parent's prefix. With
`reward_head=True` the scalar of the transformer's `reward_linear` head is attached to each state (ablation
"token reward head"; see `vla_reasoner.value.base.TokenRewardValue`).

Portions adapted from iVideoGPT, Copyright (c) 2024 THUML @ Tsinghua University, MIT License
(see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from vla_reasoner.world_model.base import WorldModel

CONTEXT_TOKENS_PER_FRAME = 16 * 16 + 1  # upstream magic number: 256 tokens + 1 separator per context frame
DYNAMICS_TOKENS_PER_FRAME = 16  # upstream magic number: 4x4 tokens per predicted frame
PAD_TOKEN_ID = 50256  # meaningless pad id used upstream to silence a generate() warning


@dataclass(eq=False)
class IVideoGPTState:
    embeds: Any  # torch (1, L, E): input embeddings, ending with the slot where the next action is added
    context_tokens: Any  # torch (1, CONTEXT_TOKENS_PER_FRAME * context_length) tokens of the real context
    frame: np.ndarray  # (R, R, 3) uint8, last (real or predicted) frame
    reward: Optional[float] = None
    depth: int = 0


class IVideoGPTWorldModel(WorldModel):
    def __init__(
        self,
        tokenizer_path: str,
        transformer_path: str,
        transformer_config: str,
        action_dim: int,
        context_length: int = 2,
        segment_length: int = 12,
        resolution: int = 64,
        device: str = "cuda",
        temperature: float = 1.0,
        top_k: int = 100,
        reward_head: bool = False,
        ivideogpt_root: Optional[str] = None,
        max_batch: Optional[int] = None,
    ):
        """
        Args:
            tokenizer_path: directory loadable by `CompressiveVQModel.from_pretrained` (e.g. the
                `unwrapped_model` dir written by upstream train_tokenizer.py, or `<hf repo>/tokenizer`).
            transformer_path: `model.safetensors` of the action-conditioned `HeadModelWithAction`, or a directory
                containing it (an accelerate checkpoint dir of train_gpt.py, or `<hf repo>/transformer`).
            transformer_config: the llama config used for training (file or directory), e.g.
                `configs/llama/config_medium.json` of the iVideoGPT repo.
            action_dim: dimension of the (flattened, chunk-packed) action the model was fine-tuned with.
        """
        import torch

        if ivideogpt_root:
            sys.path.insert(0, os.path.abspath(ivideogpt_root))
        from ivideogpt.transformer import HeadModelWithAction
        from ivideogpt.vq_model import CompressiveVQModel
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM

        self._torch = torch
        self.context_length = context_length
        self.segment_length = segment_length
        self.resolution = resolution
        self.device = torch.device(device)
        self.temperature, self.top_k = temperature, top_k
        self.reward_head = reward_head
        self.max_batch = max_batch
        self._warned_depth = False

        tokenizer = CompressiveVQModel.from_pretrained(tokenizer_path, low_cpu_mem_usage=False)
        if tokenizer.context_length != context_length:
            tokenizer.set_context_length(context_length)
        config = AutoConfig.from_pretrained(transformer_config)
        config.vocab_size = tokenizer.num_vq_embeddings + tokenizer.num_dyn_embeddings + 2
        model = HeadModelWithAction(
            AutoModelForCausalLM.from_config(config),
            action_dim=action_dim,
            prelude_tokens_num=CONTEXT_TOKENS_PER_FRAME * context_length - 1,
            tokens_num_per_dyna=DYNAMICS_TOKENS_PER_FRAME,
            context=context_length,
            segment_length=segment_length,
            model_type=config.model_type,
            reward_prediction=reward_head,
        )
        weights = transformer_path
        if os.path.isdir(weights):
            weights = os.path.join(weights, "model.safetensors")
        model.load_state_dict(load_file(weights), strict=True)
        self.tokenizer = tokenizer.to(self.device).eval()
        self.model = model.to(self.device).eval()

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        torch = self._torch
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cpu", enabled=False)

    def _frames_to_tensor(self, frames: Sequence[np.ndarray]):
        torch = self._torch
        x = torch.as_tensor(np.stack([np.asarray(f) for f in frames]), device=self.device)
        x = x.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = torch.nn.functional.interpolate(
                x, size=(self.resolution, self.resolution), mode="bilinear", antialias=True, align_corners=False
            )
        return x.unsqueeze(0)  # (1, T, 3, R, R) in [0, 1]

    @staticmethod
    def _to_uint8(x) -> np.ndarray:
        return (x.float().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)

    # ------------------------------------------------------------------ WorldModel API
    def init_state(self, frames: Sequence[np.ndarray]) -> IVideoGPTState:
        torch = self._torch
        if len(frames) != self.context_length:
            raise ValueError(f"Expected {self.context_length} context frames, got {len(frames)}.")
        x = self._frames_to_tensor(frames)
        with torch.no_grad(), self._autocast():
            # As upstream: tokenize the context together with dummy future frames, keep the prelude + first slot.
            tokens, _ = self.tokenizer.tokenize(torch.cat([x, torch.zeros_like(x)], dim=1), self.context_length)
            context_tokens = tokens[:, : CONTEXT_TOKENS_PER_FRAME * self.context_length]
            embeds = self.model.get_input_embeddings(context_tokens)
        return IVideoGPTState(embeds=embeds, context_tokens=context_tokens, frame=self._to_uint8(x[0, -1:])[0])

    def predict(self, state: IVideoGPTState, actions: np.ndarray) -> List[IVideoGPTState]:
        if state.depth + 1 > self.segment_length - self.context_length and not self._warned_depth:
            warnings.warn("Tree depth exceeds the number of future frames the world model was trained on.")
            self._warned_depth = True
        actions = np.asarray(actions, dtype=np.float32)
        step = self.max_batch or len(actions)
        out: List[IVideoGPTState] = []
        for i in range(0, len(actions), step):
            out.extend(self._predict_batch(state, actions[i:i + step]))
        return out

    def _predict_batch(self, state: IVideoGPTState, actions: np.ndarray) -> List[IVideoGPTState]:
        torch = self._torch
        b = len(actions)
        a = torch.as_tensor(actions, device=self.device)
        with torch.no_grad(), self._autocast():
            embeds = state.embeds.expand(b, -1, -1).clone()
            embeds[:, -1] += self.model.action_linear(a).to(embeds.dtype)  # action embedded into the latent space
            result = self.model.llm.generate(
                inputs_embeds=embeds,
                do_sample=True,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=PAD_TOKEN_ID,
                use_cache=True,
                max_new_tokens=DYNAMICS_TOKENS_PER_FRAME + 1,
                return_dict_in_generate=True,
                output_hidden_states=self.reward_head,
            )
            new_tokens = result.sequences[:, :-1]  # the last sampled token is replaced by the slot token
            rewards = None
            if self.reward_head:  # as in upstream VideoPredictor.rollout
                rewards = self.model.reward_linear(result.hidden_states[-1][-1]).reshape(b).float().cpu().numpy()
            slot = torch.full((b, 1), self.model.token_for_sdf, dtype=new_tokens.dtype, device=self.device)
            next_embeds = torch.cat([embeds, self.model.get_input_embeddings(torch.cat([new_tokens, slot], 1))], 1)
            decoded = self.tokenizer.detokenize(
                torch.cat([state.context_tokens.expand(b, -1), new_tokens], dim=1), self.context_length
            )
            frames = self._to_uint8(decoded[:, -1])
        return [
            IVideoGPTState(
                embeds=next_embeds[j:j + 1],
                context_tokens=state.context_tokens,
                frame=frames[j],
                reward=None if rewards is None else float(rewards[j]),
                depth=state.depth + 1,
            )
            for j in range(b)
        ]
