import torch


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of a vector using PyTorch.

    :param x: Input tensor.
    :param discount: Discount factor.
    :return: Discounted cumulative sum tensor.
    """
    result = torch.zeros_like(x)
    running_add = 0
    for t in reversed(range(len(x))):
        running_add = x[t] + discount * running_add
        result[t] = running_add
    return result


class RolloutBuffer:
    def __init__(
        self,
        size,
        obs_dim,
        device=None,
    ):
        self.device = device

        self.data = {
            "obs": torch.zeros((size, obs_dim)).to(device),
            "reward": torch.zeros((size)).to(device),
            "value": torch.zeros((size)).to(device),
            "action_log": torch.zeros((size)).to(device),
            "return": torch.zeros((size)).to(device),
            "advantages": torch.zeros((size)).to(device),
        }
        self.max_size, self.ptr, self.start = size, 0, 0

    def store(
        self,
        obs,
        action_log_prob,
        reward,
        value,
    ):
        assert self.ptr < self.max_size

        self.data["obs"][self.ptr] = torch.tensor(obs).to(self.device)
        self.data["action_log"][self.ptr] = action_log_prob
        self.data["value"][self.ptr] = value
        self.data["reward"][self.ptr] = torch.tensor(reward).to(self.device)
        self.ptr += 1

    def reset(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.ptr, self.start = 0, 0

    def calculate_discounted_rewards(
        self, last_value=None, gamma=0.99, lam=0.95
    ):
        """Called at the end of each trajectory, calculates GAE advantages
        estimates, and reward-to-go for each state in the trajectory.

        GAE-Lambda advantages for the policy update, and reward-to-go for the
        value function targets.

        The last value should be 0 if a terminal state has be reached, and
        V(S_T) otherwise the value function estimate for the last state. Used
        for bootstrapping the reward-to-go and account for timesteps beyond
        the arbitrary episode horizon, the epoch cutoff.
        """
        # traj = slice(self.start, self.ptr)

        if last_value is None:
            last_value = torch.tensor([0.0]).to(self.device)

        rews = torch.cat(
            (self.data["reward"][self.start : self.ptr], last_value)
        )
        vals = torch.cat(
            (self.data["value"][self.start : self.ptr], last_value)
        )

        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.data["advantages"][self.start : self.ptr] = discount_cumsum(
            deltas, gamma * lam
        )
        self.data["return"][self.start : self.ptr] = discount_cumsum(
            rews, gamma
        )[:-1]

        self.start = self.ptr

    def get(self) -> dict:
        """If the buffer is filled, returns the collected rollouts in a dictionary.
        Otherwise, throws an assertion exception.

        Returns:
            dict[torch.tensor]: The collected rollouts
        """
        assert self.ptr == self.max_size
        return self.data
