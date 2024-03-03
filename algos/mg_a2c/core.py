import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size, **ac_kwargs):
        super().__init__()

        # TODO: MLP function or not?
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
