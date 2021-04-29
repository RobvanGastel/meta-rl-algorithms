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


"""Original Policy Gradient implementation by OpenAI
"""


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def reward_to_go(rews, obs=None, acts=None, gamma=None, v_net=None):
    n = len(rews)
    rtgs = np.zeros_like(rews)

    for i in reversed(range(n)):
        rtgs[i] = gamma * rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(obs_dim, 128)
        self.linear2 = nn.Linear(128, act_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


class Agent:
    def __init__(self, env, baseline=None,
                 gamma=0.99, lr=3e-2, hidden_size=64):
        assert baseline is not None, \
            "Give a baseline to compute the policy gradient"

        assert isinstance(env.observation_space, Box), \
            "This example only works for envs with continuous state spaces."
        assert isinstance(env.action_space, Discrete), \
            "This example only works for envs with discrete action spaces."

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.max_ep_len = env._max_episode_steps

        self.gamma = gamma
        self.baseline = baseline

        self.log_pi = Model(self.obs_dim, self.act_dim)
        self.opt = optim.Adam(self.log_pi.parameters(), lr=lr)

        self.v = Model(self.obs_dim, 1)
        self.opt_v = optim.Adam(self.v.parameters(), lr=lr)

    def update_policy(self, act, obs, baseline):
        self.opt.zero_grad()

        # Calculate the loss
        logp = self.get_policy(obs).log_prob(act)
        batch_loss = -(logp * baseline).mean()

        batch_loss.backward()
        self.opt.step()
        return batch_loss

    def get_policy(self, obs):
        logits = self.log_pi(obs)
        return Categorical(logits=logits)

    def get_action(self, obs, deterministic=False):
        return self.get_policy(obs).sample().item()

    def sample_batch(self, batch_size=5000):
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        ep_rews = []            # list for rewards accrued throughout ep

        batch_episode = []      # for episodic samples of obs
        batch_full_rets = []    # for measuring full episode returns

        done = False
        o, ep_ret, ep_len, episode_obs = env.reset(), 0, 0, []
        while True:
            batch_obs.append(o)
            episode_obs.append(o)

            a = self.get_action(torch.as_tensor(o, dtype=torch.float32))
            o, r, done, _ = env.step(a)

            # save action, reward
            batch_acts.append(a)
            ep_rews.append(r)

            if done or (ep_len == self.max_ep_len):
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # for MC value function estimation
                batch_full_rets.append(ep_rews)
                batch_episode.append(episode_obs)

                # Discounted reward
                batch_weights += list(self.baseline(ep_rews,
                                                    batch_obs,
                                                    batch_acts,
                                                    self.gamma,
                                                    self.v))

                # reset episode-specific variables
                o, done, ep_rews, episode_obs = env.reset(), False, [], []

                if len(batch_obs) > batch_size:
                    break

        # value function estimation
        value_network_loss = self.update_value_network(
            batch_episode,
            batch_full_rets,
            batch_weights)

        batch_loss = self.update_policy(
            torch.as_tensor(batch_acts, dtype=torch.float32),
            torch.as_tensor(batch_obs, dtype=torch.float32),
            torch.as_tensor(batch_weights, dtype=torch.float32))

        return batch_loss, batch_rets, batch_lens

    # Monte Carlo approach
    def update_value_network(self, obs, rews, baseline):
        MSE = torch.nn.MSELoss()

        losses = []
        targets = []
        l = torch.Tensor([])
        for i in range(len(obs)):
            state = np.array(obs[i])
            reward = rews[i]
            if type(reward[0]) == torch.Tensor:
                reward = torch.cat(tuple(reward))
            else:
                reward = torch.Tensor(reward)

            target = reward.unsqueeze(1)
            targets.append(target.mean().data.numpy())
            loss = MSE(target, self.v(torch.Tensor(state)))
            losses.append(loss)

        losses = torch.stack(losses).mean()
        self.opt_v.zero_grad()
        losses.backward()
        self.opt_v.step()
        return losses

    def train(self, writer, epochs=50, batch_size=5000):
        # training loop
        for i in range(epochs):
            batch_loss, batch_rets, batch_lens = self.sample_batch(
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
    agent = Agent(env, lr=args.lr, baseline=reward_to_go)
    agent.train(tbWriter, epochs=args.epochs,
                batch_size=args.batch)

    agent.test(env)
    tbWriter.close()
