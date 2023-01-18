import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, device):
        super().__init__()

        # Both conditioned on a
        # nn.Linear(input_size, 32)

        # For an MDP T = 10N, N is the number of episodes
        # T is the horizon.

        # Value function
        # TCBlock(T, 16)
        # TCBlock(T, 16)
        # AttentionBlock(16, 16)

        # Policy function
        # TCBlock(T, 32)
        # TCBlock(T, 32)
        # AttentionBlock(32, 32)
