import numpy as np

from vla_reasoner.deploy import Robot, run_episode
from vla_reasoner.policies.base import VLAPolicy


class CounterRobot(Robot):
    def __init__(self):
        self.t, self.applied = 0, []

    def reset(self):
        self.t, self.applied = 0, []

    def get_observation(self):
        return {"policy_input": {"t": self.t}, "frame": np.full((2, 2, 3), self.t, np.uint8)}

    def apply_action(self, action):
        self.applied.append(np.asarray(action))
        self.t += 1


class ChunkPolicy(VLAPolicy):
    action_dim, exec_chunk = 2, 3

    def reset(self, instruction):
        self.instruction = instruction

    def predict(self, observation):
        t = observation["t"]
        return np.stack([np.full(2, t + i, dtype=float) for i in range(3)])


def test_run_episode_records_per_step_frames_and_actions():
    robot, policy = CounterRobot(), ChunkPolicy()
    out = run_episode(robot, policy, "do it", max_steps=7)
    assert len(out["frames"]) == len(out["actions"]) == 7  # stops mid-chunk at max_steps
    assert [int(f[0, 0, 0]) for f in out["frames"]] == list(range(7))  # frame before every executed action
    np.testing.assert_allclose(out["actions"][:, 0], [0, 1, 2, 3, 4, 5, 6])
    assert policy.instruction == "do it"


def test_run_episode_uses_reasoner_output():
    class Negate:
        def reset(self):
            pass

        def __call__(self, actions, frame):
            return -actions

    robot = CounterRobot()
    out = run_episode(robot, ChunkPolicy(), "x", max_steps=3, reasoner=Negate())
    np.testing.assert_allclose(out["actions"][:, 0], [0, -1, -2])
