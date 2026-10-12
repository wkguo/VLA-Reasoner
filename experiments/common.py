"""Shared absl flags and builders for the experiment scripts (data loading and VLA-Reasoner construction)."""

from __future__ import annotations

import math
import os
import sys
from typing import Callable, Optional

from absl import flags

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FLAGS = flags.FLAGS


# ------------------------------------------------------------------------------------------------ data
def define_data_flags() -> None:
    flags.DEFINE_enum("data_format", "libero_hdf5", ["libero_hdf5", "bridge_rlds", "npz"], "Dataset format.")
    flags.DEFINE_string("data_path", None, "LIBERO suite dir (hdf5), TFDS dir (Bridge) or npz dir.")
    flags.DEFINE_string("action_convention", "",
                        "LIBERO: openvla|raw (default openvla). Bridge: octo|bridge_orig|raw (default octo).")
    flags.DEFINE_string("image_key", "", "Camera key (default: agentview_rgb for LIBERO, image_0 for Bridge).")
    flags.DEFINE_integer("max_episodes", 0, "Use at most this many episodes (0 = all).")


def iter_dataset():
    from vla_reasoner.data import iter_episodes

    kwargs = {}
    if FLAGS.max_episodes:
        kwargs["max_episodes"] = FLAGS.max_episodes
    if FLAGS.data_format in ("libero_hdf5", "bridge_rlds"):
        if FLAGS.action_convention:
            kwargs["action_convention"] = FLAGS.action_convention
        if FLAGS.image_key:
            kwargs["image_key"] = FLAGS.image_key
    return iter_episodes(FLAGS.data_format, FLAGS.data_path, **kwargs)


# ------------------------------------------------------------------------------------------------ reasoner
def define_reasoner_flags() -> None:
    flags.DEFINE_boolean("use_reasoner", False, "Wrap the frozen VLA with VLA-Reasoner.")
    flags.DEFINE_float("alpha", 0.6, "Injection strength alpha (Eq. 1); 1.0 = vanilla VLA.")
    flags.DEFINE_integer("max_depth", 2, "MaxDepth of Algorithm 1 (the loop runs d = 0..MaxDepth).")
    flags.DEFINE_integer("top_k", 5, "k: candidates per expansion.")
    flags.DEFINE_integer("num_kde_samples", 1000, "N: KDE samples drawn per expansion before Top-k.")
    flags.DEFINE_float("c_ucb", 1.0 / math.sqrt(2.0), "c in the UCB rule (Eq. 5).")
    flags.DEFINE_integer("search_interval", 1, "Run the search every n-th VLA query.")
    flags.DEFINE_enum("sampler", "kde", ["kde", "gaussian"], "Expansion sampler; 'gaussian' is the ablation.")
    flags.DEFINE_float("noise_std", 0.1, "Gaussian ablation: noise std in units of the per-dimension dataset std "
                       "stored in the KDE file (absolute units if --kde_path is empty).")
    flags.DEFINE_string("kde_path", "", "KDE file from experiments/fit_kde.py.")
    flags.DEFINE_enum("value", "image", ["image", "token_reward"],
                      "Value estimate: ResNet-34 + MLP (paper) or the world model's reward head (ablation).")
    flags.DEFINE_string("value_path", "", "Value checkpoint from experiments/train_value.py.")
    flags.DEFINE_enum("world_model", "ivideogpt", ["ivideogpt", "oracle_debug"],
                      "'oracle_debug' uses simulator save/restore: DEBUG ONLY, not the paper's method.")
    flags.DEFINE_string("ivideogpt_root", "", "Path to a clone of thuml/iVideoGPT.")
    flags.DEFINE_string("wm_tokenizer", "", "iVideoGPT tokenizer dir (.../unwrapped_model).")
    flags.DEFINE_string("wm_transformer", "", "Action-conditioned transformer: model.safetensors or its directory.")
    flags.DEFINE_string("wm_config", "", "llama config used to train the transformer (file or dir).")
    flags.DEFINE_integer("wm_context_length", 2, "Context frames of the world model.")
    flags.DEFINE_integer("wm_segment_length", 12, "Segment length the world model was trained with.")
    flags.DEFINE_integer("wm_resolution", 64, "World-model frame resolution.")
    flags.DEFINE_integer("wm_max_batch", 0, "Max candidates per world-model generate call (0 = all k).")
    flags.DEFINE_string("reasoner_device", "cuda", "Device of the world model and the value network.")
    flags.DEFINE_integer("reasoner_seed", 0, "Seed of the reasoner's sampler.")


def reasoner_tag() -> str:
    """Name of the run setting, used in log / video paths; non-default settings are appended."""
    if not FLAGS.use_reasoner:
        return "vanilla"
    tag = f"reasoner_a{FLAGS.alpha:g}_d{FLAGS.max_depth}_k{FLAGS.top_k}_n{FLAGS.num_kde_samples}"
    if FLAGS.search_interval != 1:
        tag += f"_si{FLAGS.search_interval}"
    if not math.isclose(FLAGS.c_ucb, 1.0 / math.sqrt(2.0), rel_tol=1e-9):
        tag += f"_c{FLAGS.c_ucb:g}"
    if FLAGS.sampler != "kde":
        tag += f"_{FLAGS.sampler}{FLAGS.noise_std:g}"  # sampler and its noise std
    if FLAGS.value != "image":
        tag += f"_{FLAGS.value}"
    if FLAGS.world_model != "ivideogpt":
        tag += f"_{FLAGS.world_model}"
    if FLAGS.reasoner_seed != 0:
        tag += f"_rs{FLAGS.reasoner_seed}"
    return tag


def build_reasoner(flat_action_dim: int, oracle_factory: Optional[Callable] = None):
    """Build a VLAReasoner from the flags, or return None if --use_reasoner is off."""
    if not FLAGS.use_reasoner:
        return None
    from vla_reasoner import ActionKDE, GaussianNoiseSampler, KDETopKSampler, ReasonerConfig, VLAReasoner
    from vla_reasoner.value.base import TokenRewardValue

    kde = ActionKDE.load(FLAGS.kde_path) if FLAGS.kde_path else None
    if kde is not None and kde.dim != flat_action_dim:
        raise ValueError(f"KDE dimension {kde.dim} != executed action size {flat_action_dim} (check --chunk_size).")
    if FLAGS.sampler == "kde":
        if kde is None:
            raise ValueError("--sampler=kde needs --kde_path.")
        sampler = KDETopKSampler(kde, FLAGS.top_k, FLAGS.num_kde_samples)
    else:
        std = FLAGS.noise_std * (kde.std if kde is not None else 1.0)
        sampler = GaussianNoiseSampler(FLAGS.top_k, std)

    if FLAGS.world_model == "oracle_debug":
        if oracle_factory is None:
            raise ValueError("This script does not provide a debug oracle.")
        world_model = oracle_factory()
    else:
        from vla_reasoner.world_model.ivideogpt import IVideoGPTWorldModel

        world_model = IVideoGPTWorldModel(
            tokenizer_path=FLAGS.wm_tokenizer,
            transformer_path=FLAGS.wm_transformer,
            transformer_config=FLAGS.wm_config,
            action_dim=flat_action_dim,
            context_length=FLAGS.wm_context_length,
            segment_length=FLAGS.wm_segment_length,
            resolution=FLAGS.wm_resolution,
            device=FLAGS.reasoner_device,
            reward_head=FLAGS.value == "token_reward",
            ivideogpt_root=FLAGS.ivideogpt_root or None,
            max_batch=FLAGS.wm_max_batch or None,
        )

    if FLAGS.value == "image":
        from vla_reasoner.value.model import ImageValueFunction

        value_fn = ImageValueFunction.load(FLAGS.value_path, device=FLAGS.reasoner_device)
    else:
        value_fn = TokenRewardValue()

    config = ReasonerConfig(alpha=FLAGS.alpha, max_depth=FLAGS.max_depth, top_k=FLAGS.top_k,
                            num_kde_samples=FLAGS.num_kde_samples, c_ucb=FLAGS.c_ucb,
                            search_interval=FLAGS.search_interval, seed=FLAGS.reasoner_seed)
    return VLAReasoner(world_model, value_fn, sampler, config)
