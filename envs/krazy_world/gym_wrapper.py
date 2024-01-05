import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.krazy_world.krazy_world import KrazyGridWorld


class KrazyWorld(gym.Env):
    """KrazyWorld Gymnasium wrapper"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed, task_seed=None, use_local_obs=True, max_episode_steps=100):

        # The settings except the seeds are taken from the E-MAML
        # implementation of krazyworld.
        self._env = KrazyGridWorld(
            screen_height=256,
            grid_squares_per_row=10,
            one_hot_obs=False,
            use_local_obs=use_local_obs,
            image_obs=False,
            seed=seed,
            task_seed=task_seed,
            num_goals=3,
            max_goal_distance=np.inf,
            min_goal_distance=2,
            death_square_percentage=0.08,
            num_steps_before_energy_needed=50,
            energy_sq_perc=0.05,
            energy_replenish=8,
            num_transporters=1,
            ice_sq_perc=0.05,
        )

        # Required environmental attributes
        self._observation_space = spaces.Box(0, 9, shape=(100,), dtype=np.int32)
        self._action_space = spaces.Discrete(4)
        self.reward_range = (0, 3)

        # Own implementation for the TimeLimit wrapper to pass truncation
        self.current_step = 0
        self.max_episode_steps = max_episode_steps

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def spec(self):
        return self._spec
    

    def reset(
        self,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)
        self.current_step = 0

        options = {
            "reset_board": False,
            "reset_colors": False,
            "reset_agent_start_pos": False,
            "reset_dynamics": False,
        }
        return self._env.reset(*options), {}

    def step(self, action):
        obs, rew, terminated, info = self._env.step(action, render=False)

        self.current_step += 1
        truncated = self.current_step == self.max_episode_steps

        return obs, rew, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
