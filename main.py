import torch
import argparse

from utils.logger import Logger
from algos.rl2_ppo.agent import PPO
from algos.rl2_ppo.buffer import RolloutBuffer
from envs.krazy_world.gym_wrapper import KrazyWorld


def main(args):

    Logger("rl2_ppo", "logs")

    Logger.get().info("Logger set. Initializing training...")

    ### Config
    # TODO: Proper config
    hidden_size = 128
    epochs = 1000
    update_epochs = 15
    update_every_n = 5
    max_ep_len = 100

    ac_kwargs = {
        "obs_enc_dim": 512,
        "rnn_state_size": 128,
        "actor_hidden_sizes": [128, 128],
        "critic_hidden_sizes": [128, 128],
    }
    ###

    device = torch.device("cuda:2")

    env = KrazyWorld(seed=1, max_ep_len=max_ep_len)
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        ac_kwargs=ac_kwargs,
        device=device,
    )
    buffer = RolloutBuffer(
        size=max_ep_len,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )

    ### RL^2 variables
    rnn_state = agent.actor_critic.initialize_state(batch_size=1)
    prev_action = torch.tensor(0).to(device).unsqueeze(0)
    prev_rew = torch.tensor(0).to(device).view(1, 1)
    ###

    # Number of episodes
    for epoch in range(epochs):
        obs, ep_ret, ep_len = env.reset(), 0, 0
        done = False

        while not done:
            obs = torch.tensor(obs).to(device).float().unsqueeze(0)
            action, value, log_prob, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, done, _ = env.step(action)

            buffer.store(obs, action, rew, prev_action, prev_rew, done, value, log_prob)

            # Update the observation
            obs = next_obs

            # Set previous action and reward tensors
            prev_action = action.detach()
            prev_rew = torch.tensor(rew).to(device).view(1, 1)

            ep_ret += rew
            ep_len += 1

            if done:
                obs = torch.tensor(obs).to(device).float().unsqueeze(0)
                _, value, _, _ = agent.act(obs, prev_action, prev_rew, rnn_state)

                # TODO: Correctly, input last_termination/done.
                buffer.finish_path(value, done)

                # Update every n episodes
                if epoch % update_every_n == 0 and epoch != 0:
                    batch, batch_size = buffer.get()
                    agent.optimize(batch, update_epochs, batch_size)
                    buffer.reset()

                print(f"ep_ret: {ep_ret} and ep_len: {ep_len}")
                ep_ret, ep_len = 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
