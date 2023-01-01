import torch
import argparse

from algos.ppo.agent import PPO
from algos.ppo.buffer import RolloutBuffer
from envs.krazy_world.gym_wrapper import KrazyWorld


def main(args):

    # TODO: Proper config
    # Config
    hidden_size = 128
    epochs = 10
    ac_kwargs = {"actor_hidden_sizes": [128, 128], "critic_hidden_sizes": [128, 128]}

    device = torch.device("cuda")

    env = KrazyWorld(seed=42)
    print(env.observation_space.shape, env.action_space.n)
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        ac_kwargs=ac_kwargs,
        device=device,
    )
    buffer = RolloutBuffer(
        size=int(1e5),
        num_envs=1,
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
    )

    # RL^2 variables
    rnn_state = (
        torch.zeros([1, 1, hidden_size]).to(device),
        torch.zeros([1, 1, hidden_size]).to(device),
    )
    prev_action = 0
    prev_rew = 0

    # TODO: Why np.array?
    done = False
    # next_done = torch.zeros(1).to(device)

    global_step = 0
    # Number of episodes
    for epoch in range(epochs):
        obs, ep_ret, ep_len = env.reset(), 0, 0

        while not done:
            action, value, log_prob, rnn_state = agent.act(
                obs, prev_action, prev_rew, rnn_state
            )
            next_obs, rew, done, info = env.step(action)

            # Store: obs, action, rew, logp_a, v, prev_action, prev_rew
            # Maybe, hidden
            buffer.store(obs, action, rew, prev_action, prev_rew, done, value, log_prob)

            # Update the observation
            obs = next_obs

            # Set previous action and reward
            prev_action = action
            prev_rew = rew

            if done:
                _, value, _, _ = agent.act(obs, prev_action, prev_rew, rnn_state)
                # TODO: Correctly, input last_termination/done.
                buffer.finish_path(value, done)

                # TODO: Update every n episodes
                if epoch % 5 == 0:
                    batch = buffer.get()
                    agent.optimize(batch, 15)
                    buffer.reset()

                print(f"ep_ret: {ep_ret} and ep_len: {ep_len}")

        buffer.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Agent
    # parser.add_argument(
    #     "--rnn_",
    # )
    args = parser.parse_args()

    main(args)
