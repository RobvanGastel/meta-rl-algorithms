import torch
import torch.nn as nn

from algos.ppo.core import ActorCritic


class PPO(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        ac_kwargs,
        device,
        seed=0,
        clip_ratio=0.2,
        entropy_coeff=0.0,  # TODO: Currently not used
        value_coeff=0.5,
        max_grad_norm=0.5,
        lr=3e-4,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device

        # Optimize variables
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.actor_critic = ActorCritic(obs_dim, action_dim, device, **ac_kwargs)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def step(self):
        return NotImplemented

    def act(self, obs, prev_action, prev_rew, rnn_state):

        # TODO: inputs to tensors
        obs = torch.as_tensor([obs], dtype=torch.float32).to(self.device)
        prev_action = torch.as_tensor([prev_action], dtype=torch.float32).to(
            self.device
        )
        prev_rew = torch.as_tensor([[prev_rew]], dtype=torch.float32).to(self.device)
        # rnn_state = rnn_state.to(self.device)

        print(obs.shape, prev_action.shape, prev_rew.shape)

        action, v, logp_a, rnn_state_out = self.actor_critic.step(
            obs, prev_action, prev_rew, rnn_state
        )
        return action, v, logp_a, rnn_state_out

    def optimize(
        self,
        batch,
        # global_step,
        update_epochs,
        # num_mini_batches,
        # next_rnn_state,
        target_kl=None,
    ):

        final_pi_info = dict(
            kl=torch.tensor(0), ent=torch.tensor(0), cf=torch.tensor(0)
        )
        final_pi_loss, final_v_loss = 0, 0

        # TODO: Multiple choices possible for RNN state
        # Currently, initializing to 0s
        zero_rnn_state = (
            self.actor_critic.initialize_state(batch_size=1),
            self.actor_critic.initialize_state(batch_size=1),
        )

        for epoch in range(update_epochs):
            pi_loss, pi_info = self._compute_policy_loss(batch, zero_rnn_state)
            v_loss = self._compute_value_loss(batch, zero_rnn_state)
            loss = (
                pi_loss
                - self.entropy_coeff * pi_info["ent"]
                + v_loss * self.value_coeff
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            final_pi_loss, final_v_loss, final_pi_info = (
                pi_loss,
                v_loss,
                pi_info,
            )

            if target_kl and final_pi_info["kl"] > 1.5 * target_kl:
                break

    def _compute_value_loss(self, batch, zero_rnn_state):
        obs, ret, prev_act, prev_rew = (
            batch["obs"],
            batch["return"],
            batch["prev_act"],
            batch["prev_rew"],
        )
        # TODO: Check

        v, _ = self.actor_critic.v(
            obs, prev_act, prev_rew, zero_rnn_state, training=True
        )
        loss_v = ((v - ret) ** 2).mean()

        return loss_v

    def _compute_policy_loss(self, batch, zero_rnn_state):
        b_state, b_action, b_advantage, b_log_prob = (
            batch["obs"],
            batch["action"],
            batch["advantage"],
            batch["log_prob"],
        )

        # Policy loss
        pi, _, log_prob = self.actor_critic.act(b_state, b_action)
        ratio = torch.exp(log_prob - b_log_prob)
        clip_advantage = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advantage
        )
        loss_pi = -torch.min(ratio * b_advantage, clip_advantage).mean()

        # Debug info
        approx_kl = (b_log_prob - log_prob).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clip_frac)

        return loss_pi, pi_info
