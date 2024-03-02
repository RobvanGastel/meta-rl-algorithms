import yaml
import argparse

import torch

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.logger import Logger
from algos.mg_a2c.agent import train_a2c, train_mgrl_a2c


def main(config):
    Logger.get().info(f"Start meta-training, experiment name: {config['name']}")
    Logger.get().info(f"config: {config}")

    torch.autograd.set_detect_anomaly(True)

    envs = [RecordEpisodeStatistics(gym.make("CartPole-v0"))]

    # Temporary interface to allow for one main file
    train_mgrl_a2c(config, envs)

    # train_a2c(config, envs)


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
