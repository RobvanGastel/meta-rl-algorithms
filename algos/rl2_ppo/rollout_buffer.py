import torch


class RolloutBuffer:
    def __init__(
        self,
        size,
        state_dim,
        action_dim,
        gamma=0.99,
        gae_lambda=0.95,
        device=None,
    ):
        self.device = device

        self.data = {
            "obs": torch.zeros((size, state_dim)).to(device),
            "action": torch.zeros((size, action_dim)).to(device),
            "advantage": torch.zeros((size)).to(device),
            "reward": torch.zeros((size)).to(device),
            "return": torch.zeros((size)).to(device),
            "value": torch.zeros((size)).to(device),
            "prev_reward": torch.zeros((size)).to(device),
            "prev_action": torch.zeros((size)).to(device),
            "log_prob": torch.zeros((size)).to(device),
            "termination": torch.zeros((size,)).to(device),
        }
        self.episodes = []

        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.max_size, self.curr_size = size, 0

    def store(
        self,
        obs,
        action,
        reward,
        prev_action,
        prev_reward,
        term,
        value,
        log_prob,
    ):
        assert self.curr_size < self.max_size
        self.data["obs"][self.curr_size] = obs
        self.data["action"][self.curr_size] = action
        self.data["reward"][self.curr_size] = torch.tensor(reward).to(self.device)
        self.data["termination"][self.curr_size] = torch.tensor(term).to(self.device)
        self.data["prev_action"][self.curr_size] = prev_action
        self.data["prev_reward"][self.curr_size] = prev_reward
        self.data["value"][self.curr_size] = value
        self.data["log_prob"][self.curr_size] = log_prob
        self.curr_size += 1

    def reset(self):
        self.empty_buffer()
        self.curr_size = 0

    def empty_buffer(self):
        for key, val in self.data.items():
            self.data[key] = torch.zeros_like(val)
        self.curr_size = 0

    def finish_path(self, last_value, last_termination):

        # Calculate advantages
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

        # Add to episode list
        episode = {k: v.clone() for k, v in self.data.items()}
        self.episodes.append(episode)

        # Empty episode buffer
        self.empty_buffer()

    def get(self):
        # format the experience to (batch_size, horizon, ...) length

        batch = {
            k: torch.stack([ep[k] for ep in self.episodes])
            for k in self.episodes[0].keys()
        }

        print(len(self.episodes))

        return batch, len(self.episodes)
