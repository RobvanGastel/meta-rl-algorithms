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

        # RNN input: (batch size, obs embedding + one-hot actions + reward)
        # LSTM layer + observation encoding with both orthogonal initialization
        # TODO:
        self.action_dim = action_dim
        self.device = device

        hidden_size = 512
        self.hidden_dim = 128
        # TODO: MLP
        self.obs_enc = _layer_init(nn.Linear(obs_dim, hidden_size).to(device))
        # Input: (batch size, observation embedding)
        # TODO: rnn output size
        # TODO: LSTM or GRU
        self.rnn = nn.LSTM(hidden_size + 1 + action_dim, 128, batch_first=True).to(
            device
        )

        # TODO: Sequence length

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # The heads on the embeddings
        if not actor_stds:
            actor_stds = [np.sqrt(2)] * (len(actor_hidden_sizes)) + [0.01]
        self.actor_network = mlp(
            [128] + list(actor_hidden_sizes) + [action_dim],
            actor_stds,
            hidden_activation,
            output_activation,
        ).to(device)

        if not critic_stds:
            critic_stds = [np.sqrt(2)] * (len(critic_hidden_sizes)) + [1.0]
        self.critic_network = mlp(
            [128] + list(critic_hidden_sizes) + [1],
            critic_stds,
            hidden_activation,
            output_activation,
        ).to(device)

    def _one_hot(self, act):
        # TODO: Needs to be adjusted for continuous state space
        # TODO: Check implementation
        return torch.eye(self.action_dim)[act.long(), :].to(self.device)

    def initialize_state(self, batch_size):
        return torch.zeros(2 * self.hidden_dim).unsqueeze(0).repeat(batch_size, 1)

    def embedding_state(self, obs, prev_act, prev_rew, rnn_state, training=False):

        # previous action one-hot encoding: (batch_size, act_dim)
        prev_act = self._one_hot(prev_act)
        obs_enc = self.obs_enc(obs)

        print(obs_enc.shape, prev_act.shape, prev_rew.shape)
        rnn_input = torch.cat([obs_enc, prev_act, prev_rew], dim=-1)
        print(rnn_input.shape)

        if training:
            # TODO: If we dont want to pad to longest sequence change this.
            # Input rnn: (batch size, sequence length, features)
            h = rnn_input.size()
            rnn_input = rnn_input.reshape(
                (h[0] // self.sequence_length), self.sequence_length, h[1]
            )

            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)

            h = rnn_out.size()
            rnn_out = rnn_out.reshape(h[0] * h[1], h[2])

        else:
            # Input rnn: (1, 1, features)
            rnn_input = rnn_input.unsqueeze(1)
            rnn_out, rnn_state_out = self.rnn(rnn_input, rnn_state)
            rnn_out = rnn_out.squeeze(1)
            # Output rnn: (1, features)

        return rnn_out, rnn_state_out

    def pi(self, obs, prev_act, prev_rew, rnn_state, action=None, training=False):

        rnn_out, rnn_state_out = self.embedding_state(
            obs, prev_act, prev_rew, rnn_state, training
        )

        # TODO: Continuous policy
        logits = self.actor_network(rnn_out)
        pi = Categorical(logits=logits)

        logp_a = None
        if action is not None:
            logp_a = pi.log_prob(action)
        return pi, logp_a, rnn_state_out

    def v(self, obs, prev_act, prev_rew, rnn_state, training=False):

        rnn_out, rnn_state_out = self.embedding_state(
            obs, prev_act, prev_rew, rnn_state, training
        )

        value_logits = self.critic_network(rnn_out)
        value_logits = value_logits.reshape(-1)
        return value_logits, rnn_state_out

    # def act(self, obs, action=None):
    #     return self.policy_network(obs, action)

    def step(self, obs, prev_act, prev_rew, rnn_state):
        with torch.no_grad():
            pi, _, rnn_state_out = self.pi(obs, prev_act, prev_rew, rnn_state)
            action = pi.sample()

            # Log_prob of action a
            logp_a = pi.log_prob(action)
            v, _ = self.v(obs, prev_act, prev_rew, rnn_state)

        return action, v, logp_a, rnn_state_out
