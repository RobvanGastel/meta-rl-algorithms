import os

import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
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
                layer_init(nn.Linear(dims[i], dims[i + 1]), stds[i]),
                hidden_activation(),
            ]
        )
    layers.extend(
        [
            layer_init(nn.Linear(dims[-2], dims[-1]), stds[-1]),
            output_activation(),
        ]
    )
    return nn.Sequential(*layers)
def save_state(state_dict, path, epoch=None, job_id=None):
    """Save the model and optimizer states using PyTorch"""

    model_file = os.path.join(path, f"e{epoch}_state") if epoch is not None else path

    # save the model (to temporary path if job_id is specified then rename)
    model_file_tmp = model_file if job_id is None else model_file + f"_{job_id}"
    torch.save(state_dict, model_file_tmp)
    if model_file_tmp != model_file:
        os.rename(model_file_tmp, model_file)
