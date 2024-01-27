# Small experiment testing meta-gradient reinforcement learning on a single environment
# to inspect the meta-gradient.

import yaml
import argparse

import torch
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
        # ac_kwargs={},
        device=config["device"],
    )

    debug_rews = []  # Array containing total rewards
    for epoch in range(config["epochs"]):

        agent.actor_optim.zero_grad()
        agent.critic_optim.zero_grad()

        data, rews = agent.collect_rollouts(env)
        debug_rews.append(rews)

        policy_loss = -(
            torch.sum(data["action_log"] * data["advantages"].detach())
        )
        critic_loss = nn.SmoothL1Loss()(data["return"], data["value"])

        policy_loss.backward()
        critic_loss.backward()
        agent.critic_optim.step()
        agent.actor_optim.step()

        if epoch % 10 == 0:
            print(
                f"Average reward during episodes {epoch} is "
                f"{sum(debug_rews)/len(debug_rews)} {max(debug_rews)} {min(debug_rews)}"
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
