import torch


class RolloutBuffer:
    def __init__(
        self,
        size,
        num_envs,
        state_dim,
        action_dim,
        gamma=0.99,
        gae_lambda=0.95,
        device=None,
    ):
        """
        Buffer for storing trajectories experienced by the agent interacting with the environment.
        @param size: size of the buffer
        @param num_envs: number of parallel environments
        @param state_dim: dimension of the state space
        @param action_dim: dimension of the action space
        @param gamma: discount factor
        @param gae_lambda: lambda for GAE-Lambda computation
        @param device: device to store the buffer on
        """
        super().__init__()
        # print(size, num_envs, state_dim)
        self.data = {
            "obs": torch.zeros((size, num_envs, state_dim)).to(device),
            "action": torch.zeros((size, num_envs, action_dim)).to(device),
            "advantage": torch.zeros((size, num_envs)).to(device),
            "reward": torch.zeros((size, num_envs)).to(device),
            "return": torch.zeros((size, num_envs)).to(device),
            "value": torch.zeros((size, num_envs)).to(device),
            "prev_reward": torch.zeros((size, num_envs)).to(device),
            "prev_action": torch.zeros((size, num_envs)).to(device),
            "log_prob": torch.zeros((size, num_envs)).to(device),
            "termination": torch.zeros((size, num_envs)).to(device),
        }
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.max_size, self.curr_size = size, 0
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(
        self,
        obs,
        action,
        reward,
        prev_action,
        prev_reward,
        termination,
        value,
        log_prob,
    ):
        assert self.curr_size < self.max_size
        print(value)
        self.data["obs"][self.curr_size] = torch.tensor(obs).to(self.device)
        self.data["action"][self.curr_size] = torch.tensor(action).to(self.device)
        self.data["reward"][self.curr_size] = torch.tensor(reward).to(self.device)
        self.data["prev_action"][self.curr_size] = torch.tensor(prev_action).to(
            self.device
        )
        self.data["prev_reward"][self.curr_size] = torch.tensor(prev_reward).to(
            self.device
        )
        self.data["termination"][self.curr_size] = torch.tensor(termination).to(
            self.device
        )
        self.data["value"][self.curr_size] = torch.tensor(value).to(self.device)
        self.data["log_prob"][self.curr_size] = torch.tensor(log_prob).to(self.device)
        self.curr_size += 1

    def reset(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.curr_size = 0

    def finish_path(self, last_value, last_termination):
        # TODO: Unclear how last_termination is used in original code
        prev_advantage = 0
        for step in reversed(range(self.max_size)):
            if step == self.max_size - 1:
                next_non_terminal = 1.0 - last_termination
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.data["termination"][step + 1]
                next_value = self.data["value"][step + 1]

            delta = (
                self.data["reward"][step]
                + self.gamma * next_value * next_non_terminal
                - self.data["value"][step]
            )
            self.data["advantage"][step] = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * prev_advantage
            )
            prev_advantage = self.data["advantage"][step]

        self.data["return"] = self.data["advantage"] + self.data["value"]

    def get(self):
        # assert self.curr_size == self.max_size

        # Flatten the data from all the parallel environments (size, num_envs, *) -> (size * num_envs, *)
        return {k: v.flatten(0, 1) for k, v in self.data.items()}
