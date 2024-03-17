import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical

from utils.misc import mlp


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        actor_hidden_sizes,
        critic_hidden_sizes,
        **ac_kwargs
    ):
        super().__init__()

        # TODO: MLP function or not?
        self.actor = mlp(
            [obs_space.shape[0]] + list(actor_hidden_sizes) + [action_space.n],
            [np.sqrt(2)] * (len(actor_hidden_sizes)) + [0.01],
            nn.ReLU,
            nn.Identity,
        )

        self.critic = mlp(
            [obs_space.shape[0]] + list(critic_hidden_sizes) + [1],
            [np.sqrt(2)] * (len(critic_hidden_sizes)) + [1.00],
            nn.ReLU,
            nn.Identity,
        )

    def step(self, obs):
        obs = torch.tensor(obs)
        action_logits = self.actor(obs)
        pi = Categorical(logits=action_logits)

        action = pi.sample()
        action_log_prob = pi.log_prob(action)
        value = self.critic(obs).view(-1)
        return action, action_log_prob, value

    def act(self, obs):
        with torch.no_grad():
            action_logits = self.actor(obs)
            pi = Categorical(logits=action_logits)
            action = pi.sample()

            action_log_prob = pi.log_prob(action)

        return action, action_log_prob
