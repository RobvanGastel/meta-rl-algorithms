import numpy as np

import gymnasium as gym
from gymnasium import spaces


class MDP(gym.Env):
    """MDP Gymnasium Environment"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, state_size, action_size, task_seed=42, max_episode_steps=10):
        self.state_size = state_size
        self.action_size = action_size
        self.max_episode_steps = max_episode_steps

        # Required environmental attributes
        self._observation_space = spaces.Box(
            0, action_size, shape=(state_size,), dtype=np.int32
        )
        self._action_space = spaces.Discrete(action_size)
        self.reward_range = (0, 10)

        # MDP variables
        self.current_step = 0
        self.state = 0

        # Sample the task dependent variables
        self.reward_means = None
        self.transition_probability_matrix = None
        self.sample_mdp_dynamics(task_seed=task_seed)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def sample_mdp_dynamics(self, task_seed):
        # Set seed for the current task
        np.random.seed(seed=task_seed)

        self.sample_reward_dynamics()
        self.sample_state_dynamics()
        self.state = 0

    def sample_reward_dynamics(self):
        self.reward_means = np.random.normal(
            loc=1.0, scale=1.0, size=(self.state_size, self.action_size)
        )

    def sample_state_dynamics(self):
        prob_aijs = []
        for _ in range(self.action_size):
            dirichlet_ij = np.random.dirichlet(
                alpha=np.ones(
                    shape=(self.state_size,),
                    dtype=np.float32,
                ),
                size=(self.state_size,),
            )
            prob_aijs.append(dirichlet_ij)
        self.transition_probability_matrix = np.stack(prob_aijs, axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.state = 0
        return self.state

    def step(self, action):
        self.current_step += 1

        # Indice next state
        next_state = np.random.choice(
            a=self.state_size,
            p=self.transition_probability_matrix[action, self.state],
        )
        self.state = next_state

        # Generate next reward and clip
        reward = np.random.normal(
            loc=self.reward_means[self.state, action],
            scale=1.0,
        )
        reward = np.clip(reward, self.reward_range[0], self.reward_range[1])

        truncated = self.current_step == self.max_episode_steps
        return next_state, reward, False, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
