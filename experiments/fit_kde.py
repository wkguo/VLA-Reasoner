"""Step 1: fit the KDE action prior pi_KDE (paper Sec. III-C, Eq. 6) on the VLA's fine-tuning dataset.

Actions are converted to the VLA's raw output space by the loader (--action_convention). For chunked policies,
every length-`chunk_size` window (stride 1) is flattened into one sample; `chunk_size` must equal the number of
actions the VLA executes per query (OpenVLA / Octo 1, SpatialVLA 4, pi0-FAST 5).

Example:
    python experiments/fit_kde.py --data_format=libero_hdf5 --data_path=$LIBERO/libero_spatial \
        --chunk_size=1 --output=checkpoints/libero_spatial/kde.npz
"""

import os

import numpy as np
from absl import app, flags

import common

FLAGS = flags.FLAGS
common.define_data_flags()
flags.DEFINE_integer("chunk_size", 1, "Executed actions per VLA query; each chunk is one KDE sample.")
flags.DEFINE_string("bandwidth", "scott", "'scott', 'silverman' or a float (standardized units if --standardize).")
flags.DEFINE_boolean("standardize", True, "Per-dimension standardization before the KDE.")
flags.DEFINE_integer("max_points", 0, "Randomly keep at most this many samples (0 = all).")
flags.DEFINE_integer("seed", 0, "Seed for --max_points subsampling.")
flags.DEFINE_string("output", None, "Output .npz path.")
flags.mark_flags_as_required(["data_path", "output"])


def main(_):
    from vla_reasoner import ActionKDE
    from vla_reasoner.data import action_chunks

    chunks, num_episodes = [], 0
    for ep in common.iter_dataset():
        if not ep.success:
            continue
        c = action_chunks(ep.actions, FLAGS.chunk_size)
        if len(c):
            chunks.append(c.astype(np.float32))
            num_episodes += 1
    data = np.concatenate(chunks)
    if FLAGS.max_points and len(data) > FLAGS.max_points:
        idx = np.random.default_rng(FLAGS.seed).choice(len(data), FLAGS.max_points, replace=False)
        data = data[idx]
    kde = ActionKDE(bandwidth=FLAGS.bandwidth, standardize=FLAGS.standardize).fit(data)
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output)), exist_ok=True)
    kde.save(FLAGS.output)
    print(f"KDE fit on {kde.n} samples of shape {kde.action_shape} from {num_episodes} episodes; "
          f"dim={kde.dim}, h={kde.h:.4f} ({FLAGS.bandwidth}), standardize={FLAGS.standardize} -> {FLAGS.output}")
    print("per-dim mean:", np.round(kde.mean, 4))
    print("per-dim std: ", np.round(kde.std, 4))


if __name__ == "__main__":
    app.run(main)
