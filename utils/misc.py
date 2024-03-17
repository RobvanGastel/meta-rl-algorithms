import os
import logging

import torch
import numpy as np
import torch.nn as nn
from moviepy.editor import ImageSequenceClip


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
    """
    Initialize the weights and biases of a neural network layer.

    Args:
        layer (torch.nn.Module): The neural network layer to initialize.
        std (float): The standard deviation to use for orthogonal weight initialization.
            Defaults to square root of 2.
        bias_const (float): The constant to use for initializing the biases. Defaults
            to 0.0.

    Returns:
        layer (torch.nn.Module): The initialized neural network layer.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(dims, stds, hidden_activation=nn.Tanh, output_activation=nn.Identity):
    """Construct a multi-layer perceptron (MLP) neural network.

    Args:
        dims (list of int): The dimensions of each layer of the MLP. The first value is
            the dimension of the input layer, the last value is the dimension of the
            output layer, and the values in between are the dimensions of the hidden layers.
        stds (list of float): The standard deviations to use for initializing the weights
            of each layer.
        hidden_activation (torch.nn.Module): The activation function to use for the hidden
            layers. Defaults to nn.Tanh.
        output_activation (torch.nn.Module): The activation function to use for the output
            layer. Defaults to nn.Identity.

    Returns:
        model (torch.nn.Sequential): The constructed MLP model.
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


def make_gif(agent, env, episode, config):
    """
    Generate a GIF of the agent's performance during an episode in a given environment.

    Args:
        agent (object): An instance of the agent whose performance will be recorded.
        env (object): An instance of the OpenAI gym environment.
        episode (int): The episode number to save recording.
        config (dict): A dictionary of configuration parameters.

    Returns:
        None.
    """

    obs, _ = env.reset()
    terminated, truncated = False, False

    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = (
        torch.tensor(env.action_space.sample()).to(config["device"]).view(-1)
    )
    prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)

    steps = []
    rewards = []
    while not (terminated or truncated):
        steps.append(env.render())

        obs = (
            torch.tensor(obs).float().to(config["device"]).float().unsqueeze(0)
        )
        act, _, _, rnn_state = agent.act(obs, prev_action, prev_rew, rnn_state)
        next_obs, reward, terminated, truncated, _ = env.step(act.cpu().numpy())

        # Update the observation
        obs = next_obs

        # Set previous action and reward tensors
        prev_action = act.detach()
        prev_rew = torch.tensor(reward).to(config["device"]).view(1, 1)

        rewards.append(reward)

    clip = ImageSequenceClip(steps, fps=30)
    save_dir = os.path.join(config["path"], "gifs")
    gif_name = f"{save_dir}/krazyWorld_epoch_{str(episode)}.gif"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    clip.write_gif(
        gif_name,
        fps=30,
        verbose=False,
        logger=None,
    )

    logging.info(f"Generating GIF {gif_name}")


def save_state(state_dict, path, epoch=None, job_id=None):
    """Save the model and optimizer states using PyTorch"""

    model_file = (
        os.path.join(path, f"e{epoch}_state") if epoch is not None else path
    )

    # save the model (to temporary path if job_id is specified then rename)
    model_file_tmp = model_file if job_id is None else model_file + f"_{job_id}"
    torch.save(state_dict, model_file_tmp)
    if model_file_tmp != model_file:
        os.rename(model_file_tmp, model_file)
