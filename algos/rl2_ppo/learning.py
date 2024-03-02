import logging

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from utils.misc import make_gif
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer


def train_rl2_ppo(
    config,
    envs: list[gym.Env],
    test_envs: list[gym.Env] = None,
    writer: SummaryWriter = None,
):

    # Define the agent and rollout buffer
    agent = PPO(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        writer=writer,
        device=config["device"],
        ac_kwargs=config["actor_critic"],
        **config["ppo"],
    )
    buffer = RolloutBuffer(
        obs_space=envs[0].observation_space,
        action_space=envs[0].action_space,
        device=config["device"],
        size=config["max_episode_steps"],
        gae_lambda=config["ppo"]["gae_lambda"],
    )

    global_step = 0
    for meta_epoch in range(config["meta_epochs"]):

        # Sample new meta-training environment
        env = np.random.choice(envs, 1)[0]
        avg_return, avg_ep_len = [], []

        # RL^2 variables
        rnn_state = agent.actor_critic.initialize_state(batch_size=1)
        prev_action = (
            torch.tensor(env.action_space.sample())
            .to(config["device"])
            .view(-1)
        )
        prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)
        ###

        # Iterate for number of episodes
        for epoch in range(config["episodes"]):
            termination, truncated = False, False
            obs, _ = env.reset()

            while not (termination or truncated):
                obs = (
                    torch.tensor(obs).to(config["device"]).float().unsqueeze(0)
                )
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
                prev_rew = torch.tensor(rew).to(config["device"]).view(1, 1)

                if termination or truncated:
                    obs = (
                        torch.tensor(obs)
                        .to(config["device"])
                        .float()
                        .unsqueeze(0)
                    )
                    _, value, _, _ = agent.act(
                        obs, prev_action, prev_rew, rnn_state
                    )
                    buffer.finish_path(value, termination)

                    # Update every n episodes
                    if epoch % config["update_every_n"] == 0 and epoch != 0:
                        batch, batch_size = buffer.get()
                        agent.optimize(
                            batch,
                            config["update_epochs"],
                            batch_size,
                            global_step,
                        )
                        buffer.reset()

                    # Log final episode statistics
                    writer.add_scalar(
                        "env/ep_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "env/ep_length", info["episode"]["l"], global_step
                    )

                    avg_return.append(info["episode"]["r"])
                    avg_ep_len.append(info["episode"]["l"])

                global_step += 1

        # Store the meta-weights of the agent
        if meta_epoch % config["log_every_n"] == 0 and meta_epoch != 0:

            test_env = np.random.choice(test_envs, 1)[0]
            if meta_epoch % (config["log_every_n"] * 5) == 0:
                make_gif(agent, test_env, meta_epoch, config)

            # Save the weights
            if not config["debug"]:
                agent.save_weights(config["path"], meta_epoch)

            # Log test statistics
            test_return, test_ep_len = evaluate_policy(agent, test_env, config)
            writer.add_scalar("env/test_ep_return", test_return, global_step)
            writer.add_scalar("env/test_ep_length", test_ep_len, global_step)

            logging.info(
                f"meta-epoch #: {meta_epoch} "
                f"meta-train - episode return, length: ({np.mean(avg_return):.3f}, "
                f" {np.mean(avg_ep_len):.0f}) "
                f"meta-test - episode return, length: ({np.mean(test_return):.3f}, "
                f"{np.mean(test_ep_len):.0f})"
            )


def evaluate_policy(agent, env, config, episodes=10):
    """
    Evaluate the performance of an agent's policy on a given environment.

    Args:
        agent (object): An instance of the agent to be evaluated.
        env (object): An instance of the OpenAI gym environment to evaluate the agent on.
        device (str): The device to run the evaluation on (e.g. 'cpu', 'cuda').
        episodes (int): The number of episodes to run the evaluation for.

    Returns:
        A tuple of two floats, representing the average return and average episode length
        over the given number of episodes.
    """

    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = (
        torch.tensor(env.action_space.sample()).to(config["device"]).view(-1)
    )
    prev_rew = torch.tensor(0).to(config["device"]).view(1, 1)

    avg_return, avg_ep_len = [], []
    for _ in range(1, episodes):
        obs, _ = env.reset()
        termination, truncated = False, False

        while not (termination or truncated):
            obs = torch.tensor(obs).to(config["device"]).float().unsqueeze(0)
            act, _, _, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, termination, truncated, info = env.step(
                act.cpu().numpy()
            )

            # Update the observation
            obs = next_obs

            # Set previous action and reward tensors
            prev_action = act.detach()
            prev_rew = torch.tensor(rew).to(config["device"]).view(1, 1)

            if termination or truncated:
                avg_return.append(info["episode"]["r"])
                avg_ep_len.append(info["episode"]["l"])
                break

    return np.array(avg_return).mean(), np.array(avg_ep_len).mean()
