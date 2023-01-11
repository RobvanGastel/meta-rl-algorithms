import torch
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(dims, stds, hidden_activation=nn.Tanh, output_activation=nn.Identity):
    """
    Construct a multi-layer perceptron with given hidden and output activations.
    @param dims: list of dimensions, including input and output dims
    @param stds: list of standard deviations for each layer, shape = (len(dims)-1,)
    @param hidden_activation: activation function for all hidden layers
    @param output_activation: activation function for output layer
    """
    assert len(dims) == len(stds) + 1

    layers = []
    for i in range(len(dims) - 2):
        layers.extend(
            [
                _layer_init(nn.Linear(dims[i], dims[i + 1]), stds[i]),
                hidden_activation(),
            ]
        )
    layers.extend(
        [
            _layer_init(nn.Linear(dims[-2], dims[-1]), stds[-1]),
            output_activation(),
        ]
    )
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        rnn_type,
        obs_enc_dim,
        rnn_state_size,
        actor_hidden_sizes,
        critic_hidden_sizes,
        actor_stds=None,
        critic_stds=None,
        hidden_activation=nn.Tanh,
        output_activation=nn.ReLU,
    ):
        """
        Actor Critic network for the agent.
        @param obs_dim: dimension of the state space
        @param action_dim: dimension of the action space
        @param actor_hidden_sizes: list of hidden layer sizes for actor network
        @param critic_hidden_sizes: list of hidden layer sizes for critic network
        @param hidden_activation: activation function for all hidden layers
        @param output_activation: activation function for output layer
        """
        super().__init__()
        assert rnn_type == "gru" or rnn_type == "lstm", f"Passed unsuported rnn type"

        self.action_dim = action_dim
        self.device = device

        ### RNN layer
        self.rnn_type = rnn_type
        self.obs_enc_dim = obs_enc_dim
        self.rnn_state_size = rnn_state_size

        # Observation encoder MLP
        # Input: (batch size, obs embedding)
        self.obs_enc = _layer_init(nn.Linear(obs_dim, obs_enc_dim).to(device))

        # RNN input: (batch size, obs embedding + one-hot actions + reward)
        rnn_input = obs_enc_dim + 1 + action_dim

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(rnn_input, self.rnn_state_size, batch_first=True).to(
                device
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(rnn_input, self.rnn_state_size, batch_first=True).to(
                device
            )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        ###

        # The heads on the embeddings
        if not actor_stds:
            actor_stds = [np.sqrt(2)] * (len(actor_hidden_sizes)) + [0.01]
        self.actor_network = mlp(
            [self.rnn_state_size] + list(actor_hidden_sizes) + [action_dim],
            actor_stds,
            hidden_activation,
            output_activation,
        ).to(device)

        if not critic_stds:
            critic_stds = [np.sqrt(2)] * (len(critic_hidden_sizes)) + [1.0]
        self.critic_network = mlp(
            [self.rnn_state_size] + list(critic_hidden_sizes) + [1],
            critic_stds,
            hidden_activation,
            output_activation,
        ).to(device)

    def _one_hot(self, act):
        return torch.eye(self.action_dim).to(self.device)[act.long(), :]

    def initialize_state(self, batch_size):
        if self.rnn_type == "lstm":
            rnn_state = (
                torch.zeros(1, batch_size, self.rnn_state_size).to(self.device),
                torch.zeros(1, batch_size, self.rnn_state_size).to(self.device),
            )
        elif self.rnn_type == "gru":
            rnn_state = torch.zeros(1, batch_size, self.rnn_state_size).to(self.device)

        return rnn_state

    def recurrent_state(self, obs, prev_action, prev_reward, rnn_state, training=False):

        # previous action one-hot encoding: (batch_size, act_dim)
        prev_action = self._one_hot(prev_action)

        # observation encoding
        obs_enc = self.obs_enc(obs)

        # Concat the encodings with previous reward
        rnn_input = torch.cat([obs_enc, prev_action, prev_reward], dim=-1)

        if training:
            # Input rnn: (batch size, sequence length, features)
            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)
        else:
            # Input rnn: (1, 1, features)
            # TODO: len(rnn_input) == 2 do the unsqueezing.

            rnn_input = rnn_input.unsqueeze(1)
            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)
            rnn_out = rnn_out.squeeze(1)
            # Output rnn: (1, features)

        return rnn_out, rnn_state_out

    def pi(self, obs, prev_action, prev_reward, rnn_state, action=None, training=False):

        # Condition on the recurrent neural network before passing to policy network
        rnn_out, rnn_state_out = self.recurrent_state(
            obs, prev_action, prev_reward, rnn_state, training
        )

        # TODO: Continuous policy or discrete policy
        logits = self.actor_network(rnn_out)
        pi = Categorical(logits=logits)

        logp_a = None
        if action is not None:
            # TODO: Ugly fix to indice like this for discrete policy
            action = action[:, :, 0]
            logp_a = pi.log_prob(action)
        return pi, logp_a, rnn_state_out

    def v(self, obs, prev_action, prev_reward, rnn_state, training=False):

        # Condition on the recurrent neural network before passing to value network
        rnn_out, rnn_state_out = self.recurrent_state(
            obs, prev_action, prev_reward, rnn_state, training
        )

        value_logits = self.critic_network(rnn_out)
        value_logits = value_logits.squeeze(-1)
        return value_logits, rnn_state_out

    def step(self, obs, prev_action, prev_reward, rnn_state):
        with torch.no_grad():
            pi, _, rnn_state_out = self.pi(obs, prev_action, prev_reward, rnn_state)
            action = pi.sample()

            # Log_prob of action a
            logp_a = pi.log_prob(action)
            value, _ = self.v(obs, prev_action, prev_reward, rnn_state)

        return action, value, logp_a, rnn_state_out
