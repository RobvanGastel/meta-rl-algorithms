import numpy as np
from gym import spaces

from envs.krazy_world.krazy_world import KrazyGridWorld

# TODO: Update the KrazyWorld wrapper for the newest version of Gymnasium


class KrazyWorld:
    """KrazyWorld Gym wrapper"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed, task_seed=None, max_ep_len=100, use_local_obs=True):

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

        self._observation_space = spaces.Box(0, 9, shape=(100,), dtype=np.int32)
        self._action_space = spaces.Discrete(4)

        self.curr_step = 0
        self.reward_range = (0, 3)
        self.max_ep_len = max_ep_len

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(
        self,
        reset_board=False,
        reset_colors=False,
        reset_agent_start_pos=False,
        reset_dynamics=False,
    ):
        self.curr_step = 0
        return self._env.reset(
            reset_board=reset_board,
            reset_colors=reset_colors,
            reset_agent_start_pos=reset_agent_start_pos,
            reset_dynamics=reset_dynamics,
        )

    def step(self, action, render=False):
        self.curr_step += 1
        # print("curr_step", self.curr_step)
        obs, rew, done, info = self._env.step(action, render=render)

        # print(done, self.curr_step == self.max_ep_len)

        # Add time-limit
        done = True if self.curr_step == self.max_ep_len else done
        return obs, rew, done, info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()
