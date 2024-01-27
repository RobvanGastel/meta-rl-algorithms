import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from algos.mg_a2c.buffer import RolloutBuffer


class ActorCritic(nn.Module):
    def __init__(
        self, obs_space, action_space, hidden_size, device, **ac_kwargs
    ):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.n),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_space.shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def step(self, obs):
        obs = torch.tensor(obs)
        action_logits = self.actor(obs)
        pi = Categorical(logits=action_logits)

        # TODO: Move sampling and log_prob to
        # _distribution function for the actor interface.
        action = pi.sample()
        action_log_prob = pi.log_prob(action)
        value = self.critic(obs).view(-1).squeeze()
        return action, action_log_prob, value

    def act(self, obs):
        with torch.no_grad():
            action_logits = self.actor(obs)
            pi = Categorical(logits=action_logits)
            action = pi.sample()

            action_log_prob = pi.log_prob(action)

        return action, action_log_prob


class A2C(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        gamma=0.99,
        device=None,
        seed=None,
        lr=3e-2,
        max_grad_norm=0.5,
    ):
        super().__init__()

        self.num_steps = 200
        self.gamma = gamma

        self.buffer = RolloutBuffer(
            size=self.num_steps, obs_dim=obs_space.shape[0]
        )
        self.ac = ActorCritic(
            obs_space, action_space, hidden_size=128, device=None
        )

        self.actor_optim = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.ac.critic.parameters(), lr=lr)

    def collect_rollouts(self, env):
        ep_rew = []
        termination, truncated = False, False
        self.buffer.reset()

        obs, _ = env.reset()
        for _ in range(self.num_steps):
            obs = torch.tensor(obs)

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
                self.buffer.calculate_discounted_rewards(last_value)

                obs, _ = env.reset()

        # Finish the episode
        last_value = None
        if truncated:
            with torch.no_grad():
                _, _, last_value = self.ac.step(obs)
        self.buffer.calculate_discounted_rewards(last_value)

        return self.buffer.get(), sum(ep_rew) / len(ep_rew)

    def optimize(self, data):
        pass

        # Rough implementation of optimize
        # self.huber_loss = nn.SmoothL1Loss()

        # self.critic_optim.zero_grad()
        # self.actor_optim.zero_grad()

        # policy_loss = -(torch.sum(data["action_log"] * data["advantages"].detach()))
        # critic_loss = huber_loss(data["return"], data["value"])

        # policy_loss.backward()
        # critic_loss.backward()
        # critic_optim.step()
        # actor_optim.step()
