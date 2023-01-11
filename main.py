import yaml
import argparse

import torch
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.logger import Logger
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer
from envs.krazy_world.gym_wrapper import KrazyWorld


def main(config):
    Logger.get().info(f"Start meta-training RL2-PPO, {config['name']}")

    # Config
    ac_kwargs = {
        "rnn_type": config["rnn_type"],
        "obs_enc_dim": config["obs_enc_dim"],
        "rnn_state_size": config["rnn_state_size"],
        "actor_hidden_sizes": config["actor_hidden_sizes"],
        "critic_hidden_sizes": config["critic_hidden_sizes"],
    }
    device = torch.device(config["device"])
    ###

    # Distribution, same number of testing and training envs as used
    # in E-MAML
    envs = [
        RecordEpisodeStatistics(KrazyWorld(seed=s, task_seed=s**2 + 1))
        for s in range(32)
    ]
    test_envs = [
        RecordEpisodeStatistics(KrazyWorld(seed=s, task_seed=s**2 + 1))
        for s in range(100, 164)
    ]

    env = np.random.choice(envs, 1)[0]
    agent = PPO(
        config,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        ac_kwargs=ac_kwargs,
        device=device,
    )
    buffer = RolloutBuffer(
        size=config["max_episode_steps"],
        gae_lambda=config["gae_lambda"],
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )

    global_step = 0
    for meta_epoch in range(config["meta_epochs"]):

        # RL^2 variables
        rnn_state = agent.actor_critic.initialize_state(batch_size=1)
        prev_action = torch.tensor(0).to(device).unsqueeze(0)
        prev_rew = torch.tensor(0).to(device).view(1, 1)
        ###

        # Sample new meta-training environment
        env = np.random.choice(envs, 1)[0]

        # Iterate for number of episodes
        for epoch in range(config["episodes"]):
            termination, truncated = False, False
            obs, _ = env.reset()

            while not (termination or truncated):
                obs = torch.tensor(obs).to(device).float().unsqueeze(0)
                action, value, log_prob, rnn_state = agent.act(
                    obs, prev_action, prev_rew, rnn_state
                )
                next_obs, rew, termination, truncated, info = env.step(action)

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

                # if termination or truncated:
                if termination or truncated:
                    obs = torch.tensor(obs).to(device).float().unsqueeze(0)

                    # The "value" argument should be 0 if the trajectory ended
                    # because the agent reached a terminal state (died).
                    if truncated:
                        value = torch.tensor(0.0)
                    else:
                        # Otherwise it should be V(s_t), the value function estimated for the
                        # last state. This allows us to bootstrap the reward-to-go calculation
                        # to account. for timesteps beyond the arbitrary episode horizon.
                        _, value, _, _ = agent.act(
                            obs, prev_action, prev_rew, rnn_state
                        )

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
                    writer = Logger.get().writer
                    writer.add_scalar("PPO/return", info["episode"]["r"], global_step)

                global_step += 1


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
    Logger(args.name, f"runs/{args.name}")

    main(config)
