"""SimplerEnv (WidowX) evaluation of Octo / SpatialVLA with and without VLA-Reasoner (paper Table I, bottom).

The episode loop follows SimplerEnv's example (`simpler_env.make`, step until success or truncation, read
`episode_stats` from the last info); the script structure (absl flags, per-episode videos, per-task log files)
follows V-GPS's experiments/eval_vgps.py (nakamotoo/V-GPS@912a290). The reasoner edits the raw policy action
before SimplerEnv's post-processing.

Portions adapted from SimplerEnv, Copyright (c) 2024 simpler-env, MIT License (see THIRD_PARTY_NOTICES.md).

Environments: Octo needs nakamotoo/SimplerEnv@f343c82 + nakamotoo/octo@38b9b76 (as V-GPS); SpatialVLA needs
DelinQu/SimplerEnv-OpenVLA (transformers >= 4.47). See README.

Example:
    python experiments/eval_simpler.py --policy=octo --model_name=octo-small \
        --task_name=widowx_stack_cube --num_eval_episodes=100 --flagfile=experiments/configs/simpler_octo.flags
"""

import os

import imageio
import numpy as np
from absl import app, flags, logging

import common

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_enum("policy", "octo", ["octo", "spatialvla"], "Frozen VLA.")
flags.DEFINE_string("model_name", "octo-small", "Octo model type, or SpatialVLA checkpoint (HF id or path).")
flags.DEFINE_string("task_name", "widowx_put_eggplant_in_basket", "Task name.")
flags.DEFINE_integer("num_eval_episodes", 100, "Number of evaluation episodes.")
flags.DEFINE_integer("exec_chunk", 0, "Actions executed per VLA query (0 = paper setting: Octo 1, SpatialVLA 4).")
flags.DEFINE_string("logdir", "logs", "Where logs and videos are written.")
flags.DEFINE_boolean("save_videos", True, "Save one mp4 per episode.")
flags.DEFINE_string("save_rollouts_dir", "", "If set, save every episode as npz (frames, executed raw actions, "
                    "success); used to collect failure rollouts for world-model fine-tuning.")
common.define_reasoner_flags()


def widowx_raw_to_env(a: np.ndarray) -> np.ndarray:
    """Stateless raw -> env action conversion (no sticky gripper). Only used by the DEBUG oracle."""
    from transforms3d.euler import euler2axangle

    ax, angle = euler2axangle(*np.asarray(a[3:6], dtype=np.float64))
    return np.concatenate([a[:3], ax * angle, [2.0 * (a[6] > 0.5) - 1.0]])


def _tally(successes, grasps) -> str:
    """Running counts, e.g. 'success 37/100 (0.370), grasp 52/100' (grasp only on tasks that report it)."""
    n = len(successes)
    text = f"success {sum(successes)}/{n} ({sum(successes) / n:.3f})"
    return text + (f", grasp {sum(grasps)}/{n}" if grasps else "")


def main(_):
    logging.set_verbosity(logging.ERROR)
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    print(FLAGS.flag_values_dict())
    env = simpler_env.make(FLAGS.task_name)
    if "google" in FLAGS.task_name:
        policy_setup, sticky_steps = "google_robot", 15
    else:
        policy_setup, sticky_steps = "widowx_bridge", 3
    exec_chunk = FLAGS.exec_chunk or (1 if FLAGS.policy == "octo" else 4)

    def oracle_factory():  # DEBUG ONLY, not the paper's method
        from vla_reasoner.world_model.oracle import maniskill2_oracle

        return maniskill2_oracle(env, widowx_raw_to_env, get_image_from_maniskill2_obs_dict, action_dim=7)

    reasoner = common.build_reasoner(flat_action_dim=7 * exec_chunk, oracle_factory=oracle_factory)
    if FLAGS.policy == "octo":
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        if exec_chunk != 1:
            raise ValueError("The Octo adapter executes one (temporally ensembled) action per step.")
        from vla_reasoner.policies.octo_simpler import make_octo_policy

        model = make_octo_policy(FLAGS.model_name, policy_setup, FLAGS.seed, sticky_steps, reasoner)
    else:
        from vla_reasoner.policies.spatialvla_simpler import make_spatialvla_policy

        model = make_spatialvla_policy(FLAGS.model_name, policy_setup, exec_chunk=exec_chunk, reasoner=reasoner)

    tag = f"{FLAGS.policy}_{os.path.basename(FLAGS.model_name.rstrip('/'))}_{common.reasoner_tag()}"
    base_folder = os.path.join(FLAGS.logdir, tag)
    video_folder = os.path.join(base_folder, f"seed_{FLAGS.seed}", FLAGS.task_name)
    os.makedirs(video_folder, exist_ok=True)

    successes, grasps = [], []  # per episode; grasps only for tasks whose episode_stats has consecutive_grasp
    for i in range(FLAGS.num_eval_episodes):
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        model.reset(instruction)
        print(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)  # (H, W, 3) uint8
        images, raw_actions = [image], []
        success, truncated = False, False
        while not (success or truncated):
            if FLAGS.policy == "octo":
                raw_action, action = model.step(image)
            else:
                raw_action, action = model.step(image, instruction)
            raw_actions.append(np.concatenate([raw_action["world_vector"], raw_action["rotation_delta"],
                                               raw_action["open_gripper"]]))
            obs, reward, success, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            )
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)

        successes.append(bool(success))
        stats = info.get("episode_stats", {})
        if "consecutive_grasp" in stats:
            grasps.append(bool(stats["consecutive_grasp"]))
        print(f"episode {i}: success={bool(success)} | {_tally(successes, grasps)}")
        if reasoner is not None:
            print(f"reasoner: {reasoner.stats}")

        if FLAGS.save_videos:
            imageio.mimsave(os.path.join(video_folder, f"{i}_success{success}.mp4"), images, fps=10)
        if FLAGS.save_rollouts_dir:
            from vla_reasoner.data import Episode, save_episode_npz

            save_episode_npz(os.path.join(FLAGS.save_rollouts_dir, FLAGS.task_name, f"{i:04d}.npz"),
                             Episode(np.stack(images[:-1]), np.stack(raw_actions), instruction, bool(success)))

    summary = {
        "model": f"{FLAGS.policy}/{FLAGS.model_name}",
        "setting": common.reasoner_tag(),
        "task_name": FLAGS.task_name,
        "seed": FLAGS.seed,
        "episodes": len(successes),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes),
    }
    if grasps:
        summary["grasps"] = sum(grasps)
    report = "\n".join(f"{key}: {value}" for key, value in summary.items())
    print(report)
    with open(os.path.join(video_folder, "log.txt"), "w") as f:  # this run
        f.write(report + "\n")
    with open(os.path.join(base_folder, f"log_{FLAGS.task_name}.txt"), "a") as f:  # one line per seed
        f.write(f"seed {FLAGS.seed}: {_tally(successes, grasps)}\n")


if __name__ == "__main__":
    app.run(main)
