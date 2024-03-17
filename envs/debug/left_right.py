import numpy as np

import gymnasium as gym
from gymnasium import spaces


class LeftRightDebugEnv(gym.Env):
    """
    A simple gym environment where the agent gets a reward for moving left or right.
    Moving right gives a +1 reward, and moving left gives a -1 reward.
    """

    def __init__(self, max_episode_steps=1):
        super().__init__()
        # Define action and observation space
        # The agent can move either left or right
        self.action_space = spaces.Discrete(2)  # 0: Left, 1: Right
        # No real observations as there is only one step
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

        self.position = 0
        self.cur_step = 0
        self.is_debug = True
        self.max_episode_steps = max_episode_steps

    def step(self, action):
        # Update position based on action
        self.cur_step += 1
        if action == 0:

            if self.position != -1:
                self.position -= 1
            reward = -1
        else:

            if self.position != 1:
                self.position += 1
            reward = 1

        done = False
        if self.cur_step == self.max_episode_steps:
            done = True

        obs = np.array([self.position]).astype(np.float32)
        return obs, reward, done, False, {}

    def reset(self, **kwargs):
        # Reset the environment state
        self.position = 0
        self.cur_step = 0
        return np.array([self.position]).astype(np.float32), {}

    def render(self, mode="human", close=False):
        # No rendering
        pass
