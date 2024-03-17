import numpy as np
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from utils.misc import Actor, mlp, layer_init


class CategoricalActor(Actor):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_sizes,
        hidden_activation,
        output_activation,
    ):
        super().__init__()

        self.logits_network = mlp(
            [input_dim] + list(hidden_sizes) + [action_dim],
            [np.sqrt(2)] * (len(hidden_sizes)) + [0.01],
            hidden_activation,
            output_activation,
        )

    def _distribution(self, obs):
        logits = self.logits_network(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_sizes,
        hidden_activation,
        output_activation,
    ):
        super().__init__()

        self.mu_network = mlp(
            [input_dim] + list(hidden_sizes) + [action_dim],
            [np.sqrt(2)] * (len(hidden_sizes)) + [0.01],
            hidden_activation,
            output_activation,
        )
        self.log_std = nn.Parameter(-0.5 * torch.ones(1, action_dim))

    def _distribution(self, obs):
        mu = self.mu_network(obs)
        std = torch.exp(self.log_std)

        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class Critic(nn.Module):
    def __init__(
        self, input_dim, hidden_sizes, hidden_activation, output_activation
    ):
        super().__init__()
        self.value_network = mlp(
            [input_dim] + list(hidden_sizes) + [1],
            [np.sqrt(2)] * (len(hidden_sizes)) + [1.0],
            hidden_activation,
            output_activation,
        )

    def forward(self, obs):
        return self.value_network(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        rnn_type,
        rnn_size,
        obs_enc_dim,
        actor_hidden_sizes,
        critic_hidden_sizes,
        device,
        hidden_activation=nn.ReLU,
        output_activation=nn.ReLU,
    ):
        super().__init__()

        assert rnn_type in ["gru", "lstm"], "RNN type should be 'gru' or 'lstm'"
        assert type(action_space) in [
            Box,
            Discrete,
        ], "Action space should be 'Box' or 'Discrete'"

        self.device = device
        self.rnn_type = rnn_type
        self.rnn_size = rnn_size
        obs_dim = obs_space.shape[0]
        self.obs_enc_dim = obs_enc_dim

        if isinstance(action_space, Box):
            self.cont_action_space = True
            self.action_dim = action_space.shape[0]
            self.actor = GaussianActor(
                rnn_size,
                self.action_dim,
                actor_hidden_sizes,
                hidden_activation,
                output_activation,
            ).to(device)

        elif isinstance(action_space, Discrete):
            self.cont_action_space = False
            self.action_dim = action_space.n
            self.actor = CategoricalActor(
                rnn_size,
                self.action_dim,
                actor_hidden_sizes,
                hidden_activation,
                output_activation,
            ).to(device)

        self.critic = Critic(
            rnn_size, critic_hidden_sizes, hidden_activation, output_activation
        ).to(device)

        ### RNN embedding layer
        # Observation encoder input: (batch size, obs embedding)
        self.obs_enc = layer_init(nn.Linear(obs_dim, obs_enc_dim)).to(device)

        # RNN input: (batch size, obs embedding + one-hot actions + reward)
        rnn_input = obs_enc_dim + 1 + self.action_dim

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(rnn_input, rnn_size, batch_first=True).to(device)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(rnn_input, rnn_size, batch_first=True).to(device)

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        ###

    def _one_hot(self, act):
        if self.cont_action_space:
            if len(act.shape) == 2:
                act = act.unsqueeze(-1)

            return act
        return torch.eye(self.action_dim).to(self.device)[act.long(), :]

    def initialize_state(self, batch_size):
        if self.rnn_type == "lstm":
            rnn_state = (
                torch.zeros(1, batch_size, self.rnn_size).to(self.device),
                torch.zeros(1, batch_size, self.rnn_size).to(self.device),
            )
        elif self.rnn_type == "gru":
            rnn_state = torch.zeros(1, batch_size, self.rnn_size).to(
                self.device
            )

        return rnn_state

    def recurrent_state(
        self, obs, prev_action, prev_reward, rnn_state, training=False
    ):

        # previous action one-hot encoding: (batch_size, act_dim)
        prev_action = self._one_hot(prev_action)

        # observation encoding
        obs_enc = self.obs_enc(obs)

        # Concat the encodings with previous reward
        rnn_input = torch.cat(
            [obs_enc, prev_action, prev_reward], dim=-1
        ).float()

        if training:
            # Input rnn: (batch size, sequence length, features)
            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)
        else:
            # Input rnn: (1, 1, features)
            # Alternative to training bool: len(rnn_input) == 2 do the unsqueezing.

            rnn_input = rnn_input.unsqueeze(1)
            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)
            rnn_out = rnn_out.squeeze(1)
            # Output rnn: (1, features)

        return rnn_out, rnn_state_out

    def value(self, obs, prev_action, prev_reward, rnn_state, training=False):

        rnn_out, rnn_state = self.recurrent_state(
            obs, prev_action, prev_reward, rnn_state, training
        )

        return self.critic(rnn_out), rnn_state

    def pi(
        self,
        obs,
        prev_action,
        prev_reward,
        rnn_state,
        action=None,
        training=False,
    ):
        rnn_out, rnn_state = self.recurrent_state(
            obs, prev_action, prev_reward, rnn_state, training
        )

        pi, logp_a = self.actor(rnn_out, act=action)

        return pi, logp_a, rnn_state

    def step(self, obs, prev_action, prev_reward, rnn_state):
        with torch.no_grad():
            rnn_out, rnn_state = self.recurrent_state(
                obs, prev_action, prev_reward, rnn_state
            )

            pi = self.actor._distribution(rnn_out)
            action = pi.sample()

            # Log_prob of action a
            logp_a = self.actor._log_prob_from_distribution(pi, action)
            value = self.critic(rnn_out)

        return action, value, logp_a, rnn_state
