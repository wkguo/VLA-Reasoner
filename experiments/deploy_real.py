"""Real-world deployment with a robot-agnostic loop (paper Sec. IV-B, Table II). No hardware drivers included.

Provide a class implementing `vla_reasoner.deploy.Robot` (see its docstring for the observation contract) and
point --robot at it as "package.module:ClassName". Policies:
  * pi0_fast: an openpi policy server (`uv run scripts/serve_policy.py ...` in the openpi repo); 5 actions of each
    chunk are executed (paper setting).
  * openvla:  OpenVLA loaded locally through HF transformers; 1 action per query.

Example:
    python experiments/deploy_real.py --robot=my_robot.galaxea:GalaxeaA1 --policy=pi0_fast --host=localhost \
        --port=8000 --instruction="stack the green cup on the purple cup" --flagfile=experiments/configs/real_pi0fast.flags
"""

import importlib
import json
import os
import time

import numpy as np
from absl import app, flags

import common

FLAGS = flags.FLAGS
flags.DEFINE_string("robot", None, "Robot class as 'package.module:ClassName' implementing vla_reasoner.deploy.Robot.")
flags.DEFINE_string("robot_kwargs", "{}", "JSON kwargs for the robot class.")
flags.DEFINE_enum("policy", "pi0_fast", ["pi0_fast", "openvla"], "Frozen VLA.")
flags.DEFINE_string("host", "localhost", "openpi policy server host.")
flags.DEFINE_integer("port", 8000, "openpi policy server port.")
flags.DEFINE_integer("exec_chunk", 5, "Actions executed per pi0-FAST query (paper: 5).")
flags.DEFINE_string("openvla_checkpoint", "", "OpenVLA checkpoint fine-tuned on the real-robot data.")
flags.DEFINE_string("unnorm_key", "", "OpenVLA action un-normalization key of that checkpoint.")
flags.DEFINE_integer("action_dim", 7, "Raw action dimension of the policy.")
flags.DEFINE_string("instruction", None, "Language instruction.")
flags.DEFINE_integer("num_episodes", 1, "Episodes to run (the robot's reset() is called before each).")
flags.DEFINE_integer("max_steps", 400, "Maximum executed actions per episode.")
flags.DEFINE_string("save_dir", "", "If set, save each episode as npz (per-step frames, executed raw actions, "
                    "success); failures of the vanilla VLA are used to fine-tune the world model.")
flags.mark_flags_as_required(["robot", "instruction"])
common.define_reasoner_flags()


def load_robot():
    module_name, class_name = FLAGS.robot.split(":")
    return getattr(importlib.import_module(module_name), class_name)(**json.loads(FLAGS.robot_kwargs))


def main(_):
    from vla_reasoner.data import Episode, save_episode_npz
    from vla_reasoner.deploy import run_episode

    if FLAGS.policy == "pi0_fast":
        from vla_reasoner.policies.openpi_client import OpenPIClientPolicy

        policy = OpenPIClientPolicy(FLAGS.host, FLAGS.port, exec_chunk=FLAGS.exec_chunk, action_dim=FLAGS.action_dim)
    else:
        from vla_reasoner.policies.openvla import OpenVLAPolicy

        policy = OpenVLAPolicy(FLAGS.openvla_checkpoint, FLAGS.unnorm_key)
    reasoner = common.build_reasoner(flat_action_dim=policy.exec_chunk * FLAGS.action_dim)
    robot = load_robot()

    for ep in range(FLAGS.num_episodes):
        input(f"[episode {ep}] reset the scene and press Enter to start...")
        t0 = time.time()
        out = run_episode(robot, policy, FLAGS.instruction, FLAGS.max_steps, reasoner)
        success = input("success? [y/N] ").strip().lower() == "y"
        print(f"episode {ep}: {len(out['actions'])} actions in {time.time() - t0:.1f}s, success={success}")
        if reasoner is not None:
            print("reasoner:", reasoner.stats)
        if FLAGS.save_dir and len(out["actions"]):
            path = os.path.join(FLAGS.save_dir, f"{common.reasoner_tag()}_{time.strftime('%Y%m%d-%H%M%S')}_{ep:03d}.npz")
            save_episode_npz(path, Episode(np.stack(out["frames"]), out["actions"], FLAGS.instruction, success))


if __name__ == "__main__":
    app.run(main)
