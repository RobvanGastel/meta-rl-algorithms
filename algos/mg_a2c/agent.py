import torch
import torch.nn as nn

from algos.mg_a2c.core import ActorCritic
from algos.mg_a2c.buffer import RolloutBuffer


class A2C(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        gamma=0.99,
        device=None,
        seed=None,
        lr=3e-2,
        num_steps=200,
        max_grad_norm=0.5,
    ):
        super().__init__()

        self.gamma = gamma
        self.num_steps = num_steps

        self.buffer = RolloutBuffer(
            size=self.num_steps, obs_dim=obs_space.shape[0]
        )

        # TODO: Adjust to pass **ac_kwargs
        self.ac = ActorCritic(
            obs_space, action_space, hidden_size=128, device=None
        )

    def collect_rollouts(self, env, gamma):
        ep_rew = []
        termination, truncated = False, False
        self.buffer.reset()

        obs, _ = env.reset()
        for _ in range(self.num_steps):
            action, action_log_prob, value = self.ac.step(obs)
            next_obs, rew, termination, truncated, info = env.step(
                action.item()
            )

            # Update the observation
            obs = next_obs

            # Store
            self.buffer.store(obs, action_log_prob, rew, value)

            # if termination or truncated (time-limit)
            if termination or truncated:
                # TODO: Log every episode in tracking tool
                ep_rew.append(info["episode"]["r"])

                # Finish the episode
                last_value = None
                if truncated:
                    with torch.no_grad():
                        _, _, last_value = self.ac.step(obs)
                self.buffer.calculate_discounted_rewards(
                    last_value, gamma=gamma
                )

                obs, _ = env.reset()

        # Finish the episode
        last_value = None
        if truncated:
            with torch.no_grad():
                _, _, last_value = self.ac.step(obs)
        self.buffer.calculate_discounted_rewards(last_value, gamma=gamma)

        return self.buffer.get(), sum(ep_rew) / len(ep_rew)
