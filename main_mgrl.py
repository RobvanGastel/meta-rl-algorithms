# Small experiment testing meta-gradient reinforcement learning on a single environment
# to inspect the meta-gradient.

import yaml
import argparse

import torch
import torchopt
from torch import nn

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.logger import Logger
from algos.mg_a2c.agent import A2C


def main(config):
    Logger.get().info(f"Start meta-training, experiment name: {config['name']}")
    Logger.get().info(f"config: {config}")

    env = RecordEpisodeStatistics(gym.make("CartPole-v0"))

    agent = A2C(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_steps=config["num_steps"],
        device=config["device"],
    )

    # Define meta parameter
    gamma = -torch.log(
        (1 / torch.tensor(config["gamma"], requires_grad=True)) - 1
    )
    print(f"Starting value gamma: {torch.sigmoid(gamma)}")

    # Torchopt optimizers
    inner_optim = torchopt.MetaSGD(agent.ac, lr=5e-2)
    meta_optim = torchopt.SGD([gamma], lr=5e-2)
    net_state = torchopt.extract_state_dict(agent.ac)

    # TODO: Refactor after debugging
    inner_loop = 2
    debug_rews = []
    inner_loss = nn.HuberLoss()
    for epoch in range(config["epochs"]):
        for _ in range(inner_loop):
            data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))
            debug_rews.append(rews)

            # Policy loss
            a = -(torch.sum(data["action_log"] * data["advantages"].detach()))
            # Value loss
            b = inner_loss(data["return"], data["value"])

            loss = a + 0.5 * b
            inner_optim.step(loss)

        # Outer-loop
        output = agent.ac.step(data["obs"])
        data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))

        pi_loss = -(torch.sum(data["action_log"] * data["advantages"].detach()))
        v_loss = nn.HuberLoss()(data["return"], data["value"])
        meta_loss = pi_loss + 0.5 * v_loss

        meta_optim.zero_grad()
        meta_loss.backward()
        meta_optim.step()

        torchopt.recover_state_dict(agent.ac, net_state)

        # TODO: Refactor after debugging
        if epoch % 10 == 0:
            print(
                f"Average reward during episodes {epoch} is "
                f"avg: {sum(debug_rews)/len(debug_rews)} max: {max(debug_rews)} "
                f"min: {min(debug_rews)} "
                f"gamma: {torch.sigmoid(gamma):.4f}"
            )
            debug_rews = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument(
        "-d", "--debug", action="store_true", help="run in debug mode"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="configs/mg_a2c.yml"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.name is not None, "Pass a name for the experiment"
    config["name"] = args.name

    # Initialize logger
    config["debug"] = args.debug
    config["path"] = f"runs/{args.name}"
    Logger(args.name, config["path"])

    # CUDA device
    config["device"] = torch.device(config["device_id"])

    main(config)
