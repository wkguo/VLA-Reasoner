"""Step 2: train the image value function (paper Sec. III-D, Eq. 7).

Successful demonstrations are down-sampled (visual-change criterion or fixed stride), kept frames get linear
progress labels i / (L - 1), and a 2-layer MLP on top of a frozen ImageNet ResNet-34 is fit with MSE. Frames
should come from the camera the world model predicts (e.g. LIBERO agentview, Bridge image_0).

Example:
    python experiments/train_value.py --data_format=libero_hdf5 --data_path=$LIBERO/libero_spatial \
        --output=checkpoints/libero_spatial/value.pt
"""

import json
import os
import time

import numpy as np
from absl import app, flags

import common

FLAGS = flags.FLAGS
common.define_data_flags()
flags.DEFINE_enum("downsample", "visual", ["visual", "stride", "all"], "Frame down-sampling rule.")
flags.DEFINE_float("visual_threshold", 0.01, "Min. mean abs. grayscale change (in [0, 1]) to keep a frame.")
flags.DEFINE_integer("stride", 5, "Stride for --downsample=stride.")
flags.DEFINE_integer("degrade_resolution", 0, "Resize frames to this resolution first (e.g. the world model's); 0 = off.")
flags.DEFINE_integer("hidden_dim", 256, "Hidden width of the 2-layer MLP head.")
flags.DEFINE_integer("epochs", 50, "Training epochs over the cached features.")
flags.DEFINE_float("lr", 1e-3, "Adam learning rate.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("val_fraction", 0.1, "Fraction of episodes held out for validation.")
flags.DEFINE_integer("encode_batch_size", 128, "Batch size of the frozen ResNet-34 forward passes.")
flags.DEFINE_string("device", "cuda", "Torch device.")
flags.DEFINE_boolean("pretrained", True, "ImageNet weights for ResNet-34 (the paper's setting; False only for smoke tests).")
flags.DEFINE_integer("seed", 0, "Seed.")
flags.DEFINE_string("output", None, "Output checkpoint (.pt).")
flags.mark_flags_as_required(["data_path", "output"])


def main(_):
    import torch

    from vla_reasoner.value.dataset import iter_value_samples
    from vla_reasoner.value.model import ResNet34Value
    from vla_reasoner.value.train import encode_samples, fit_value_head

    torch.manual_seed(FLAGS.seed)
    model = ResNet34Value(hidden_dim=FLAGS.hidden_dim, pretrained=FLAGS.pretrained,
                          degrade_resolution=FLAGS.degrade_resolution or None)
    model = model.to(FLAGS.device)
    stats = {"episodes": 0, "frames": 0, "kept": 0}

    def samples():
        for ep in common.iter_dataset():
            stats["episodes"] += 1
            stats["frames"] += len(ep)
            for frames, labels, ep_id in iter_value_samples([ep], FLAGS.downsample, FLAGS.visual_threshold,
                                                            FLAGS.stride):
                stats["kept"] += len(labels)
                yield frames, labels, stats["episodes"]

    t0 = time.time()
    feats, labels, groups = encode_samples(model, samples(), FLAGS.encode_batch_size)
    print(f"{stats['episodes']} episodes, kept {stats['kept']} / {stats['frames']} frames "
          f"({stats['kept'] / max(stats['frames'], 1):.1%}); feature extraction {time.time() - t0:.0f}s")
    history = fit_value_head(model, feats, labels, groups, epochs=FLAGS.epochs, lr=FLAGS.lr,
                             batch_size=FLAGS.batch_size, val_fraction=FLAGS.val_fraction, seed=FLAGS.seed)
    print(f"final: {history[-1]}  (total {time.time() - t0:.0f}s)")
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output)), exist_ok=True)
    model.save(FLAGS.output, extra={"flags": {k: getattr(FLAGS, k) for k in
                                              ["data_format", "data_path", "downsample", "visual_threshold",
                                               "stride", "epochs", "lr", "batch_size"]},
                                    "history": history, "data_stats": stats})
    with open(os.path.splitext(FLAGS.output)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print("saved", FLAGS.output, "| label mean", float(np.mean(labels)))


if __name__ == "__main__":
    app.run(main)
