import torch
import numpy as np
import torch.nn as nn


from utils.logger import Logger
from algos.rl2_ppo.core import ActorCritic


class PPO(nn.Module):
    def __init__(
        self,
        config,
        obs_dim,
        action_dim,
        ac_kwargs,
        device,
        seed=0,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coeff=0.1,
        value_coeff=0.5,
        max_grad_norm=0.5,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device

        # Optimize variables
        self.clip_ratio = config["clip_ratio"]
        self.max_grad_norm = config["max_grad_norm"]
        self.value_coeff = config["value_coeff"]
        self.entropy_coeff = config["entropy_coeff"]

        self.actor_critic = ActorCritic(obs_dim, action_dim, device, **ac_kwargs)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=config["lr"], eps=1e-5
        )

    def act(self, obs, prev_action, prev_rew, rnn_state):
        return self.actor_critic.step(obs, prev_action, prev_rew, rnn_state)

    def optimize(
        self,
        batch,
        update_epochs,
        batch_size,
        global_step,
        target_kl=None,
    ):
        pi_info = dict(kl=torch.tensor(0), ent=torch.tensor(0), cf=torch.tensor(0))
        zero_rnn_state = self.actor_critic.initialize_state(batch_size=batch_size)

        for _ in range(update_epochs):

            # TODO: Potential of mini-batching the episodes

            pi_loss, pi_info = self._compute_policy_loss(batch, zero_rnn_state)
            v_loss = self._compute_value_loss(batch, zero_rnn_state)

            loss = (
                pi_loss
                - self.entropy_coeff * pi_info["ent"]
                + self.value_coeff * v_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if target_kl and pi_info["kl"] > 1.5 * target_kl:
                print(f"target kl reached {pi_info['kl']}")
                break

        # Logging of the agent variables
        y_pred, y_true = batch["value"].cpu().numpy(), batch["return"].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer = Logger.get().writer
        writer.add_scalar("PPO/explained_variance", explained_var, global_step)
        writer.add_scalar("PPO/value_loss", v_loss.item(), global_step)
        writer.add_scalar("PPO/policy_loss", pi_loss.item(), global_step)
        writer.add_scalar("PPO/entropy", pi_info["ent"].item(), global_step)
        writer.add_scalar("PPO/approx_kl", pi_info["kl"].item(), global_step)
        writer.add_scalar("PPO/clip_frac", pi_info["cf"].item(), global_step)

    def _compute_value_loss(self, batch, rnn_state):
        b_obs, b_return, b_prev_act, b_prev_rew = (
            batch["obs"],
            batch["return"],
            batch["prev_action"],
            batch["prev_reward"],
        )
        b_value = batch["value"]
        b_prev_rew = b_prev_rew.unsqueeze(-1)

        # Value loss
        v, _ = self.actor_critic.v(
            b_obs, b_prev_act, b_prev_rew, rnn_state, training=True
        )

        # Clipping the value loss
        loss_v_unclipped = (v - b_return) ** 2
        v_clipped = b_value + torch.clamp(
            v - b_value, -self.clip_ratio, self.clip_ratio
        )
        loss_v_clipped = (v_clipped - b_return) ** 2
        loss_v_max = torch.max(loss_v_unclipped, loss_v_clipped)
        loss_v = 0.5 * loss_v_max.mean()

        return loss_v

    def _compute_policy_loss(self, batch, rnn_state):

        b_obs, b_act, b_advantage, b_log_prob, b_prev_act, b_prev_rew = (
            batch["obs"],
            batch["action"],
            batch["advantage"],
            batch["log_prob"],
            batch["prev_action"],
            batch["prev_reward"],
        )
        b_prev_rew = b_prev_rew.unsqueeze(-1)

        # Normalize the advtange, done per batch to not affect the mini-batch too much
        b_advantage = (b_advantage - b_advantage.mean()) / (b_advantage.std() + 1e-8)

        pi, log_prob, _ = self.actor_critic.pi(
            b_obs, b_prev_act, b_prev_rew, rnn_state, action=b_act, training=True
        )
        ratio = torch.exp(log_prob - b_log_prob)

        # Policy loss
        clip_advantage = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advantage
        )
        loss_pi = -torch.min(ratio * b_advantage, clip_advantage).mean()

        # Debug info
        with torch.no_grad():
            approx_kl = (b_log_prob - log_prob).mean()
            clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
            clip_frac = clipped.float().mean()
            ent = pi.entropy().mean()

        return loss_pi, dict(kl=approx_kl, ent=ent, cf=clip_frac)
