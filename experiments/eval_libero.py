"""LIBERO evaluation of OpenVLA-SFT with and without VLA-Reasoner (paper Table I top, Fig. 5, Fig. 6).

Follows openvla/openvla@c8f03f4 experiments/robot/libero/run_libero_eval.py: same task-suite loop over the
fixed initial states (50 per task -> 500 episodes per suite), 10 wait steps, per-suite step limits, image
preprocessing (180 deg rotation, JPEG round trip, lanczos resize to 224), gripper post-processing
(normalize to [-1, 1], binarize, invert) and logging / replay videos. The reasoner edits the raw OpenVLA action
before that post-processing; its world-model frame is the rotated full-resolution agentview image.

Examples:
    # OpenVLA-SFT + VLA-Reasoner
    python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
        --task_suite_name=libero_spatial --use_reasoner --alpha=0.6 \
        --kde_path=checkpoints/libero_spatial/kde.npz --value_path=checkpoints/libero_spatial/value.pt \
        --ivideogpt_root=$IVG --wm_tokenizer=... --wm_transformer=... --wm_config=$IVG/configs/llama/config_medium.json
    # failure rollouts of the vanilla VLA, written to rollouts/vanilla/libero_spatial and passed to
    # export_world_model_data.py as --failure_path=rollouts/vanilla/libero_spatial
    python experiments/eval_libero.py --flagfile=experiments/configs/libero_openvla.flags \
        --task_suite_name=libero_spatial --random_init_states --num_trials_per_task=20 \
        --save_videos=false --save_rollouts_dir=rollouts/vanilla

Portions adapted from OpenVLA, Copyright (c) 2024 Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, MIT License
(see THIRD_PARTY_NOTICES.md).
"""

import os
import random
import time

import numpy as np
from absl import app, flags

import common

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained_checkpoint", "", "OpenVLA checkpoint (default: openvla/openvla-7b-finetuned-<suite>).")
flags.DEFINE_enum("task_suite_name", "libero_spatial",
                  ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"], "Task suite.")
flags.DEFINE_boolean("center_crop", True, "Center crop (checkpoints fine-tuned with random-crop augmentation).")
flags.DEFINE_string("attn_implementation", "flash_attention_2", "HF attention implementation ('' = default).")
flags.DEFINE_integer("num_steps_wait", 10, "Steps to wait for objects to settle.")
flags.DEFINE_integer("num_trials_per_task", 50, "Rollouts per task.")
flags.DEFINE_string("local_log_dir", "./experiments/logs", "Local log directory.")
flags.DEFINE_boolean("save_videos", True, "Save a replay mp4 per episode.")
flags.DEFINE_string("save_rollouts_dir", "", "If set, save every episode as npz (frames, executed raw actions, "
                    "success); used to collect failure rollouts for world-model fine-tuning.")
flags.DEFINE_boolean("random_init_states", False, "Use env.reset() placements instead of the benchmark's fixed "
                     "initial states (for collecting world-model failure rollouts away from the test states).")
flags.DEFINE_integer("seed", 7, "Random seed.")
common.define_reasoner_flags()

MAX_STEPS = {  # from run_libero_eval.py (longest training demo + margin)
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400,
}
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def set_seed_everywhere(seed: int):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_libero_env(task, resolution=256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # as upstream: the seed affects object positions even with fixed initial states
    return env, task.language


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Copy of openvla's LIBERO `resize_image` (JPEG round trip + lanczos3, as in the RLDS pipeline)."""
    import tensorflow as tf

    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def rotated_agentview(obs) -> np.ndarray:
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # rotate 180 deg to match training data


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    action[..., -1] = 2 * action[..., -1] - 1  # [0, 1] -> [-1, 1]
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action[..., -1] = action[..., -1] * -1.0  # OpenVLA: 0 = close / 1 = open  ->  LIBERO: -1 = open / +1 = close
    return action


def to_env_action(raw: np.ndarray) -> list:
    return invert_gripper_action(normalize_gripper_action(np.array(raw, dtype=np.float64), binarize=True)).tolist()


def save_rollout_video(frames, path):
    import imageio

    writer = imageio.get_writer(path, fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()


def main(_):
    from libero.libero import benchmark

    from vla_reasoner.policies.openvla import OpenVLAPolicy

    set_seed_everywhere(FLAGS.seed)
    suite = FLAGS.task_suite_name
    checkpoint = FLAGS.pretrained_checkpoint or f"openvla/openvla-7b-finetuned-{suite.replace('_', '-')}"
    policy = OpenVLAPolicy(checkpoint, unnorm_key=suite, center_crop=FLAGS.center_crop,
                           attn_implementation=FLAGS.attn_implementation or None)

    run_id = f"EVAL-{suite}-openvla-{common.reasoner_tag()}-{time.strftime('%Y_%m_%d-%H_%M_%S')}"
    os.makedirs(FLAGS.local_log_dir, exist_ok=True)
    video_dir = os.path.join(FLAGS.local_log_dir, run_id + "_videos")
    log_file = open(os.path.join(FLAGS.local_log_dir, run_id + ".txt"), "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Task suite: {suite} | checkpoint: {checkpoint} | {common.reasoner_tag()}")
    task_suite = benchmark.get_benchmark_dict()[suite]()
    reasoner = None if FLAGS.world_model == "oracle_debug" else common.build_reasoner(flat_action_dim=7)

    total_episodes, total_successes = 0, 0
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        if FLAGS.world_model == "oracle_debug":  # DEBUG ONLY: re-created per task because it wraps this env
            from vla_reasoner.world_model.oracle import libero_oracle

            reasoner = common.build_reasoner(
                flat_action_dim=7, oracle_factory=lambda: libero_oracle(env, to_env_action, rotated_agentview))

        task_episodes, task_successes = 0, 0
        for episode_idx in range(FLAGS.num_trials_per_task):
            log(f"\nTask: {task_description}")
            obs = env.reset()
            if not FLAGS.random_init_states:
                obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])
            policy.reset(task_description)
            if reasoner is not None:
                reasoner.reset()
            t, done = 0, False
            replay, wm_frames, raw_actions = [], [], []
            max_steps = MAX_STEPS[suite]
            while t < max_steps + FLAGS.num_steps_wait:
                try:  # as upstream: a simulator or policy error is logged and ends the episode
                    if t < FLAGS.num_steps_wait:  # the simulator drops objects at the start
                        obs, reward, done, info = env.step(DUMMY_ACTION)
                        t += 1
                        continue
                    img = resize_image(rotated_agentview(obs), 224)
                    replay.append(img)
                    a_vla = policy.predict({"full_image": img})  # (1, 7) raw OpenVLA action
                except Exception as e:
                    log(f"Caught exception: {e}")
                    break
                frame = rotated_agentview(obs)
                # outside the try: a reasoner error propagates instead of being counted as a failed episode
                a = reasoner(a_vla, frame) if reasoner is not None else a_vla
                wm_frames.append(frame)
                raw_actions.append(a[0])
                try:
                    obs, reward, done, info = env.step(to_env_action(a[0]))
                except Exception as e:
                    log(f"Caught exception: {e}")
                    break
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
            task_episodes += 1
            total_episodes += 1

            if FLAGS.save_videos and replay:
                os.makedirs(video_dir, exist_ok=True)
                name = task_description.lower().replace(" ", "_")[:50]
                save_rollout_video(replay, os.path.join(video_dir, f"{total_episodes}--success={done}--{name}.mp4"))
            if FLAGS.save_rollouts_dir and raw_actions:
                from vla_reasoner.data import Episode, save_episode_npz

                save_episode_npz(os.path.join(FLAGS.save_rollouts_dir, suite, f"task{task_id:02d}_{episode_idx:03d}.npz"),
                                 Episode(np.stack(wm_frames), np.stack(raw_actions), task_description, bool(done)))
            log(f"Success: {done}")
            log(f"# episodes completed so far: {total_episodes}")
            log(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if reasoner is not None:
                log(f"reasoner: {reasoner.stats}")
        log(f"Current task success rate: {task_successes / task_episodes}")
        log(f"Current total success rate: {total_successes / total_episodes}")
    log_file.close()


if __name__ == "__main__":
    app.run(main)
