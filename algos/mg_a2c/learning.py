import torch
import torchopt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from algos.mg_a2c.agent import A2C


def train_mgrl_a2c(
    config,
    envs: list[gym.Env],
    test_envs: list[gym.Env] = None,
    writer: SummaryWriter = None,
):
    env = envs[0]

    agent = A2C(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_steps=config["num_steps"],
        device=config["device"],
    )

    # Set the meta-parameter
    gamma = nn.Parameter(
        -torch.log((1 / torch.tensor(config["gamma"])) - 1),
        requires_grad=True,
    )

    # Torchopt optimizers
    inner_optim = torchopt.MetaSGD(agent.ac, lr=5e-3)
    meta_optim = torchopt.SGD([gamma], lr=1e-3)

    # TODO: Refactor after debugging
    debug_rews = []
    value_loss = nn.MSELoss()

    for epoch in range(config["epochs"]):

        for _ in range(config["inner_steps"]):
            data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))
            debug_rews.append(rews)

            a = -(torch.sum(data["action_log"] * data["advantages"].detach()))
            b = value_loss(data["return"], data["value"])

            loss = a + 0.5 * b
            inner_optim.step(loss)

        # Outer-loop
        data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))

        pi_loss = -(torch.sum(data["action_log"] * data["advantages"].detach()))
        v_loss = value_loss(data["return"], data["value"])
        meta_loss = pi_loss + config["a2c"]["value_coeff"] * v_loss

        meta_optim.zero_grad()
        meta_loss.backward()
        torchopt.clip_grad_norm(40)
        # print(f"gamma.grad = {gamma.grad!r}")
        meta_optim.step()

        # Detach the graph
        torchopt.stop_gradient(agent.ac)
        torchopt.stop_gradient(inner_optim)

        # TODO: Refactor after debugging
        if epoch % 10 == 0 and epoch != 0:
            with torch.no_grad():
                print(
                    f"Average reward during episodes {epoch} is "
                    f"avg: {sum(debug_rews)/len(debug_rews)} max: {max(debug_rews)} "
                    f"min: {min(debug_rews)} "
                    f"gamma: {torch.sigmoid(gamma):.4f}"
                )
            debug_rews = []


def train_a2c(config, envs: list[gym.Env]):
    assert len(envs), 1
    env = envs[0]

    agent = A2C(
        obs_space=env.observation_space,
        action_space=env.action_space,
        num_steps=config["num_steps"],
        device=config["device"],
    )

    # Define meta parameter
    gamma = nn.Parameter(-torch.log((1 / torch.tensor(config["gamma"])) - 1))

    inner_optim = torch.optim.Adam(agent.ac.parameters(), lr=5e-3)

    # TODO: Refactor after debugging
    inner_loop = 2
    debug_rews = []
    value_loss = nn.MSELoss()

    for epoch in range(config["epochs"]):

        for _ in range(inner_loop):
            data, rews = agent.collect_rollouts(env, torch.sigmoid(gamma))
            debug_rews.append(rews)

            a = -(torch.sum(data["action_log"] * data["advantages"].detach()))
            b = value_loss(data["return"], data["value"])

            loss = a + config["a2c"]["value_coeff"] * b

            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()

        # TODO: Refactor after debugging
        if epoch % 10 == 0:
            with torch.no_grad():
                print(
                    f"Average reward during episodes {epoch} is "
                    f"avg: {sum(debug_rews)/len(debug_rews)} max: {max(debug_rews)} "
                    f"min: {min(debug_rews)} "
                    f"gamma: {torch.sigmoid(gamma):.4f}"
                )
            debug_rews = []
