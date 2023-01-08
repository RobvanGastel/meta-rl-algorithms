import yaml
import argparse

import torch
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.logger import Logger
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer
from envs.krazy_world.gym_wrapper import KrazyWorld


def main(config):
    Logger.get().info(f"Logger set. Initializing training {config['name']}")

    # Config
    ac_kwargs = {
        "obs_enc_dim": 512,
        "rnn_state_size": 128,
        "actor_hidden_sizes": [128, 128],
        "critic_hidden_sizes": [128, 128],
    }
    ###

    device = torch.device(config["device"])

    env = RecordEpisodeStatistics(KrazyWorld(seed=1))

    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        ac_kwargs=ac_kwargs,
        device=device,
    )
    buffer = RolloutBuffer(
        size=config["max_episode_steps"],
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )

    # RL^2 variables
    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = torch.tensor(0).to(device).unsqueeze(0)
    prev_rew = torch.tensor(0).to(device).view(1, 1)
    ###

    # Number of episodes
    for epoch in range(config["epochs"]):
        obs, ep_ret, ep_len = env.reset(), 0, 0
        done = False

        while not done:
            obs = torch.tensor(obs).to(device).float().unsqueeze(0)
            action, value, log_prob, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, termination, truncated, info = env.step(action)

            buffer.store(
                obs, action, rew, prev_action, prev_rew, termination, value, log_prob
            )

            # Update the observation
            obs = next_obs

            # Set previous action and reward tensors
            prev_action = action.detach()
            prev_rew = torch.tensor(rew).to(device).view(1, 1)

            ep_ret += rew
            ep_len += 1

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
                    _, value, _, _ = agent.act(obs, prev_action, prev_rew, rnn_state)

                buffer.finish_path(value, done)

                # Update every n episodes
                if epoch % config["update_every_n"] == 0 and epoch != 0:
                    batch, batch_size = buffer.get()
                    agent.optimize(batch, config["update_epochs"], batch_size)
                    buffer.reset()

                print(
                    f"time elapsed: {info['episode']['t']} "
                    f"episode return: {info['episode']['r']} "
                    f" and episode length: {info['episode']['l']}",
                    flush=True,
                )

                ep_ret, ep_len = 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-c", "--config", type=str, default="configs/rl2_ppo.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize logger
    Logger(args.name, "logs")

    main(config)
