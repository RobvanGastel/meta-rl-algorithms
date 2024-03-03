import torch
import numpy as np
import torch.nn as nn

from algos.mg_a2c.core import ActorCritic
from algos.mg_a2c.buffer import RolloutBuffer


class A2C(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        ac_kwargs,
        writer,
        device,
        seed=42,
        value_coeff=0.5,
        max_episode_steps=200,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.writer = writer
        self.value_coeff = value_coeff
        self.max_episode_steps = max_episode_steps
        self.global_step = 0

        self.buffer = RolloutBuffer(
            size=self.max_episode_steps, obs_dim=obs_space.shape[0]
        )

        self.ac = ActorCritic(
            obs_space,
            action_space,
            **ac_kwargs,
        )

        self.value_loss = nn.MSELoss()

    def collect_rollouts(self, env, gamma):
        termination, truncated = False, False
        self.buffer.reset()

        obs, _ = env.reset()
        for _ in range(self.max_episode_steps):
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

                # Finish the episode
                last_value = None
                if truncated:
                    with torch.no_grad():
                        _, _, last_value = self.ac.step(obs)
                self.buffer.calculate_discounted_rewards(
                    last_value, gamma=gamma
                )

                # Log final episode statistics
                self.writer.add_scalar(
                    "env/ep_return", info["episode"]["r"], self.global_step
                )
                self.writer.add_scalar(
                    "env/ep_length", info["episode"]["l"], self.global_step
                )
                self.writer.add_scalar(
                    "A2C/gamma", gamma.cpu(), self.global_step
                )

                obs, _ = env.reset()

            self.global_step += 1

        # Finish the episode
        last_value = None
        if truncated:
            with torch.no_grad():
                _, _, last_value = self.ac.step(obs)

        self.buffer.calculate_discounted_rewards(last_value, gamma=gamma)
        return self.buffer.get()

    def optimize(self, batch):
        pi_loss = -(
            torch.sum(batch["action_log"] * batch["advantages"].detach())
        )
        v_loss = self.value_loss(batch["return"], batch["value"])
        loss = pi_loss + self.value_coeff * v_loss

        ret = batch["advantages"] + batch["value"]
        y_pred, y_true = (
            batch["value"].detach().reshape(-1).cpu().numpy(),
            ret.detach().reshape(-1).cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = 0 if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Value function distribution
        self.writer.add_histogram(
            "A2C/value_histogram", batch["value"], self.global_step
        )
        self.writer.add_scalar(
            "A2C/explained_variance", explained_var, self.global_step
        )
        self.writer.add_scalar(
            "A2C/value_loss", v_loss.item(), self.global_step
        )
        self.writer.add_scalar(
            "A2C/policy_loss", pi_loss.item(), self.global_step
        )
        return loss
