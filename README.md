# reinforcement-learning-algorithms
This repository contains implementation and write-up of several algorithms for testing purposes.

## Install requirements
The `requirements.txt` or conda environemnt `rl.yml` can be found in the root directory. For the installation of the OpenAI gym make sure to follow the [installation](https://github.com/openai/gym#installation) instructions.

## Available Algorithms (WIP)
A good global scope of different Reinforcement Learning algorithms can be found in: [RL Taxonomy](https://github.com/bennylp/RL-Taxonomy).

### Dynamic Programming
- [x] Policy Iteration
- [x] Value Iteration

###  Temporal Difference (TD) methods
- [x] (Double) Q-learning
- [x] Sarsa
- [x] Expected Sarsa
- [ ] ~~TD(λ)~~
- [ ] ~~n-step TD~~

### Policy Gradient 
- [x] Vanilla Policy Gradient (REINFORCE)
- [ ] Advantage Actor-Critic (A2C)
- [ ] Proximal Policy Optimization (PPO)

### Deep Q-learning Networks
- [x] Deep Q-learning Networks (DQN)


**references**
* [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)
* [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
