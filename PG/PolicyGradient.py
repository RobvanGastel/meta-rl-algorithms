import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np

import gym
from gym import wrappers
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Agent:
    def __init__(self, env, lr=0.01, sizes=None):
        self.env = env

        assert isinstance(env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."

        self.obs_dim = env.observation_space.shape[0]
        self.n_acts = env.action_space.n

        self.logits_net = mlp(sizes=[self.obs_dim]+sizes+[self.n_acts])
        self.optimizer = optim.Adam(self.logits_net.parameters(), lr=lr)

    def reward_to_go(self, rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    def get_policy(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, obs, act, weights):
        # make loss function whose gradient, for the right data, is policy
        # gradient
        logp = self.get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def test(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            if render:
                env.render()
            action = self.get_action(torch.as_tensor(obs,
                                                     dtype=torch.float32))
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        return ep_reward

    def train_one_epoch(self, batch_size=5000):
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = self.get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        self.optimizer.zero_grad()
        batch_loss = self.compute_loss(
            obs=torch.as_tensor(batch_obs,
                                dtype=torch.float32),
            act=torch.as_tensor(
                batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(
                batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss, batch_rets, batch_lens

    def train(self, writer, epochs=50, batch_size=5000):
        # training loop
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = self.train_one_epoch(
                batch_size=batch_size)
            print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
                  (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

            # ...log the running loss
            writer.add_scalar('Training batch loss',
                              batch_loss,
                              i)
            writer.add_scalar('Mean return',
                              np.mean(batch_rets),
                              i)
            writer.add_scalar('Mean episode length',
                              np.mean(batch_lens),
                              i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=5000)
    args = parser.parse_args()

    tbWriter = SummaryWriter(f'./runs/{args.env}', flush_secs=1)

    env = gym.make(args.env)
    agent = Agent(env, lr=args.lr, sizes=[128])
    agent.train(tbWriter, epochs=args.epochs,
                batch_size=args.batch)

    # TODO: Fix wrapper
    # env = wrappers.Monitor(
    #     env, f'./runs/{args.env}', force = True)
    agent.test(env)

    tbWriter.close()
