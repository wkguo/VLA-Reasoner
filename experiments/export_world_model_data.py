"""Exports demonstrations and failure rollouts in the format iVideoGPT fine-tuning expects."""

import os
from collections import defaultdict

import numpy as np
from absl import app, flags

import common

FLAGS = flags.FLAGS
common.define_data_flags()
flags.DEFINE_string("failure_path", "", "Directory of npz rollouts of the frozen VLA (failures are used).")
flags.DEFINE_integer("max_failures_per_task", 10, "Failure rollouts per instruction (paper real world: 10).")
flags.DEFINE_integer("chunk_size", 1, "Executed actions per VLA query (packed into one world-model action).")
flags.DEFINE_integer("resolution", 64, "Store frames at this square resolution (0 = keep).")
flags.DEFINE_string("output_dir", None, "Root directory passed to iVideoGPT as --dataset_path.")
flags.DEFINE_string("dataset_name", "vla_reasoner", "Sub-directory / iVideoGPT mix name.")
flags.mark_flags_as_required(["data_path", "output_dir"])


def _resize(frames: np.ndarray, size: int) -> np.ndarray:
    if not size or frames.shape[1:3] == (size, size):
        return frames
    from PIL import Image

    return np.stack([np.asarray(Image.fromarray(f).resize((size, size), Image.BICUBIC)) for f in frames])


def _write(ep, prefix: str, index: int, out_dir: str) -> int:
    from vla_reasoner.data import pack_for_world_model

    frames, actions = pack_for_world_model(ep, FLAGS.chunk_size)
    if len(frames) < 2:
        return 0
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_{index:06d}.npz"),
                        image=_resize(np.asarray(frames, dtype=np.uint8), FLAGS.resolution),
                        action=actions.astype(np.float32))
    return actions.shape[1]


def main(_):
    from vla_reasoner.data import iter_npz_episodes

    out_dir = os.path.join(FLAGS.output_dir, FLAGS.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    action_dim, n_demo, n_fail = None, 0, 0
    for ep in common.iter_dataset():
        dim = _write(ep, "demo", n_demo, out_dir)
        if dim:
            action_dim, n_demo = dim, n_demo + 1
    if FLAGS.failure_path:
        per_task = defaultdict(int)
        for ep in iter_npz_episodes(FLAGS.failure_path, success=False):
            if per_task[ep.instruction] >= FLAGS.max_failures_per_task:
                continue
            if _write(ep, "fail", n_fail, out_dir):
                per_task[ep.instruction] += 1
                n_fail += 1
    print(f"wrote {n_demo} demonstrations + {n_fail} failure rollouts to {out_dir} (action_dim={action_dim})")
    print("Fine-tune iVideoGPT on it with scripts/finetune_world_model.sh, using:")
    print(f"  --dataset_path {os.path.abspath(FLAGS.output_dir)} --oxe_data_mixes_type {FLAGS.dataset_name} "
          f"--action_dim {action_dim}")


if __name__ == "__main__":
    app.run(main)
