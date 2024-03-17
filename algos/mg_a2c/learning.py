import torch
import torchopt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from algos.mg_a2c.agent import A2C


def train_mg_a2c(
    config,
    envs: list[gym.Env],
    test_envs: list[gym.Env] = None,
    writer: SummaryWriter = None,
):
    # MGRL optimizes on a single environment
    env = envs[0]

    agent = A2C(
        obs_space=env.observation_space,
        action_space=env.action_space,
        value_coeff=config["a2c"]["value_coeff"],
        writer=writer,
        ac_kwargs=config["actor_critic"],
        max_episode_steps=config["max_episode_steps"],
        device=config["device"],
    )

    # Set the meta-parameter
    gamma = nn.Parameter(
        -torch.log((1 / torch.tensor(config["gamma"])) - 1),
        requires_grad=True,
    )

    # Torchopt optimizers
    inner_optim = torchopt.MetaSGD(agent.ac, lr=config["inner_lr"])
    meta_optim = torchopt.SGD([gamma], lr=config["outer_lr"])

    for _ in range(config["epochs"]):

        for _ in range(config["inner_steps"]):
            data = agent.collect_rollouts(env, torch.sigmoid(gamma))
            loss = agent.optimize(data)

            inner_optim.step(loss)

        # Outer-loop
        data = agent.collect_rollouts(env, torch.sigmoid(gamma))
        meta_loss = agent.optimize(data)

        meta_optim.zero_grad()
        meta_loss.backward()

        # Log the gradient magnitude
        writer.add_scalar("A2C/grad_gamma", gamma.grad, agent.global_step)
        meta_optim.step()

        # Detach the graph
        torchopt.stop_gradient(agent.ac)
        torchopt.stop_gradient(inner_optim)


def train_a2c(
    config,
    envs: list[gym.Env],
    test_envs: list[gym.Env] = None,
    writer: SummaryWriter = None,
):
    assert len(envs), 1
    env = envs[0]

    agent = A2C(
        obs_space=env.observation_space,
        action_space=env.action_space,
        value_coeff=config["a2c"]["value_coeff"],
        writer=writer,
        ac_kwargs=config["actor_critic"],
        max_episode_steps=config["max_episode_steps"],
        device=config["device"],
    )

    # Define meta parameter
    gamma = nn.Parameter(-torch.log((1 / torch.tensor(config["gamma"])) - 1))

    inner_optim = torch.optim.Adam(agent.ac.parameters(), lr=config["inner_lr"])

    inner_loop = 2
    debug_rews = []
    value_loss = nn.MSELoss()

    for epoch in range(config["epochs"]):

        for _ in range(inner_loop):
            data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))
            debug_rews.append(rews)

            loss = agent.optimize(data)

            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                print(
                    f"Average reward during episodes {epoch} is "
                    f"avg: {sum(debug_rews)/len(debug_rews)} max: {max(debug_rews)} "
                    f"min: {min(debug_rews)} "
                    f"gamma: {torch.sigmoid(gamma):.4f}"
                )
            debug_rews = []
