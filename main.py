import yaml
import argparse

import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.logger import Logger
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer
from envs.krazy_world.gym_wrapper import KrazyWorld


def main(config):
    Logger.get().info(f"Start meta-training RL2-PPO, experiment name: {config['name']}")
    Logger.get().info(f"config: {config}")

    # Discrete action distribution, same number of testing and training envs as used in E-MAML
    # envs = [
    #     RecordEpisodeStatistics(KrazyWorld(seed=s, task_seed=s**2 + 1))
    #     for s in range(32)
    # ]
    # test_envs = [
    #     RecordEpisodeStatistics(KrazyWorld(seed=s, task_seed=s**2 + 1))
    #     for s in range(100, 164)
    # ]

    # Continuous action distribution, normally distributing the gravity
    envs = [
        RecordEpisodeStatistics(
            gym.make("Pendulum-v1", g=9.81 + np.random.normal(size=1)[0])
        )
        for i in range(32)
    ]

    # Check the action space
    device = torch.device(config["device_id"])

    agent = PPO(
        obs_dim=envs[0].observation_space,
        action_dim=envs[0].action_space,
        device=device,
        ac_kwargs=config["actor_critic"],
        **config["ppo"],
    )
    buffer = RolloutBuffer(
        obs_dim=envs[0].observation_space.shape[0],
        action_dim=envs[0].action_space.shape[0],
        device=device,
        size=config["max_episode_steps"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )

    global_step = 0
    for meta_epoch in range(config["meta_epochs"]):

        # Sample new meta-training environment
        env = np.random.choice(envs, 1)[0]

        # RL^2 variables
        rnn_state = agent.actor_critic.initialize_state(batch_size=1)
        prev_action = torch.tensor(env.action_space.sample()).to(device).view(1, -1)
        prev_rew = torch.tensor(0).to(device).view(1, 1)
        ###

        # Iterate for number of episodes
        for epoch in range(config["episodes"]):
            termination, truncated = False, False
            obs, _ = env.reset()

            while not (termination or truncated):
                obs = torch.tensor(obs).to(device).float().unsqueeze(0)
                action, value, log_prob, rnn_state = agent.act(
                    obs, prev_action, prev_rew, rnn_state
                )
                next_obs, rew, termination, truncated, info = env.step(
                    action.cpu().numpy()[0]
                )

                # termination
                buffer.store(
                    obs,
                    action,
                    rew,
                    prev_action,
                    prev_rew,
                    termination,
                    value,
                    log_prob,
                )

                # Update the observation
                obs = next_obs

                # Set previous action and reward tensors
                prev_action = action.detach()
                prev_rew = torch.tensor(rew).to(device).view(1, 1)

                if termination or truncated:
                    obs = torch.tensor(obs).to(device).float().unsqueeze(0)
                    _, value, _, _ = agent.act(obs, prev_action, prev_rew, rnn_state)
                    buffer.finish_path(value, termination)

                    # Update every n episodes
                    if epoch % config["update_every_n"] == 0 and epoch != 0:
                        batch, batch_size = buffer.get()
                        agent.optimize(
                            batch, config["update_epochs"], batch_size, global_step
                        )
                        buffer.reset()

                        Logger.get().info(
                            f"meta-epoch: {meta_epoch} episode #: {epoch} "
                            f"time elapsed: {info['episode']['t']} "
                            f"episode return: {info['episode']['r']} "
                            f" and episode length: {info['episode']['l']}"
                        )

                    # Log final episode reward
                    Logger.get().writer.add_scalar(
                        "PPO/return", info["episode"]["r"], global_step
                    )

                global_step += 1

        # Store the meta-weights of the agent
        if meta_epoch % config["store_weights_n"] == 0 and meta_epoch != 0:
            agent.save_weights(config["path"], meta_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-c", "--config", type=str, default="configs/rl2_ppo.yml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    assert args.name is not None, "Pass a name for the experiment"
    config["name"] = args.name

    # Initialize logger
    config["path"] = f"runs/{args.name}"
    Logger(args.name, config["path"])

    main(config)
