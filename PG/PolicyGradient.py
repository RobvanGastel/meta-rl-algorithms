#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import gym
# import gym_2048

import torch
import torch.nn as nn
import torch.optim as optim


# In[79]:


class NeuralNetwork(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_space, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)
        self.activation = nn.Tanh()
#         self.linear3 = nn.Linear(hidden_size, 1)
        
        self.layers = [self.linear1, self.activation, self.linear2]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# In[80]:


class Agent:
    def __init(self, obs_space, action_space, hidden_size=128):
        self.obs_space = obs_space
        self.action_space = action_space
        
        self.network = NeuralNetwork(obs_space, action_space, hidden_size)
        self.opt = optim.Adam(self.network.parameters(), lr=0.01)
    
    def select_action(self):
        return np.random.choice(4, 1).item()
    
    # def optimize(self):
        
#         self.loss_func = nn.MSELoss()
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
        


# In[73]:





# In[ ]:





# In[ ]:

env = gym.make('2048-v0')
assert env.observation_space.shape == (4,4)
assert env.action_space.n == 4


# In[60]:


agent = Agent()

agent.select_action()


# In[62]:


n_episodes = 100


env.seed(42)

obs = env.reset()
done = False
steps = 0

for i in range(n_episodes):
    action = agent.select_action()
    obs_, reward, done, info = env.step(action)
    
#     buffer.append(obs, reward, action, obs_, done)
    
    obs = obs_

    steps += 1
    if done:
        env.render()
        print('Next Action: "{}"\n\nReward: {}'.format(
          gym_2048.Base2048Env.ACTION_STRING[action], reward))
        break
    
print('\nTotal Moves: {}'.format(steps))


# In[ ]:





# In[3]:


import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


print "hi"
# In[ ]:


# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()


# In[4]:


# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()


# In[ ]:


# for training policy
def train_one_epoch():
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

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
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

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens


# In[ ]:





# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box

from time import sleep
from IPython.display import clear_output

import gym
import gym_2048

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# 2048-v0
# CartPole-v0'
def train(env_name='2048-v0', hidden_sizes=[32], lr=1e-2, 
          epochs=200, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box),         "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete),         "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim**2]+hidden_sizes+[n_acts])

    def reward_to_go(rews):
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        c = Categorical(logits=logits)
        return c

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        a = get_policy(obs).sample()
        return a.item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
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

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.flatten().copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs.flatten(), dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            sleep(2)
            clear_output(wait=True)
            env.render()
            
            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
#                 batch_weights += [ep_ret] * ep_len                
                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env_name', '--env', type=str, default=)
#     parser.add_argument('--render', action='store_true')
#     parser.add_argument('--lr', type=float, default=1e-2)
    
print('\nUsing simplest formulation of policy gradient.\n')
train()


# In[18]:


gym.make('2048-v0').observation_space.shape


# In[ ]:




