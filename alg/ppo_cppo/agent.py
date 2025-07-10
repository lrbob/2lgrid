import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from typing import Optional

from common.logger import Logger
from env.eval import Evaluator


class RolloutBuffer:
    """Simple rollout storage."""
    def __init__(self, capacity, obs_shape, action_dim):
        self.capacity = capacity
        self.obs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.extra_info = []
        self.log_probs = []

    def add(self, obs, action, reward, cost, done, extra, log_prob):
        if len(self.obs) >= self.capacity:
            return
        self.obs.append(np.array(obs))
        self.actions.append(np.array(action))
        self.rewards.append(np.array(reward))
        self.costs.append(np.array(cost))
        self.dones.append(np.array(done))
        self.extra_info.append(np.array(extra))
        self.log_probs.append(np.array(log_prob))

        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))
        if self.discrete_action:
            logits = self.actor_out(x)
            dist = torch.distributions.Categorical(logits=logits)
        else:
            mu = self.actor_mu(x)
            log_std = self.actor_logstd(x).clamp(-20, 2)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
        return dist

    def forward_critic(self, obs, thresholds=None):
        if self.use_vve and thresholds is not None:
            if self.threshold_encoder_critic:
                th_feat = self.threshold_encoder_critic(thresholds)
                x = torch.cat([obs, th_feat], dim=-1)
            else:
                x = torch.cat([obs, thresholds], dim=-1)
        else:
            x = obs
        x = F.relu(self.critic_fc1(x))
        x = F.relu(self.critic_fc2(x))
        return self.critic_out(x).squeeze(-1)

    @torch.no_grad()
    def get_eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_cvi:
            th_tensor = torch.full(
                (obs.shape[0], 1),
                getattr(self.args, "fixed_threshold", 0.0),
                device=obs.device,
            )
        else:
            th_tensor = None
        dist = self.forward_actor(obs, th_tensor)
        if self.discrete_action:
            return dist.probs.argmax(dim=-1)
        return dist.mean

    def train(
        self,
        total_timesteps: int,
        logger: Optional[Logger] = None,
        evaluator: Optional[Evaluator] = None,
        start_time: float = 0.0,
    ) -> None:
        """Run the training loop for ``total_timesteps`` steps."""

        env = self.env
        num_envs = getattr(env, "num_envs", 1)
        current_thresholds = np.zeros(num_envs)
        for i in range(num_envs):
            if getattr(self.args, "dynamic_threshold", False):
                current_thresholds[i] = np.random.uniform(
                    self.args.threshold_min, self.args.threshold_max
                )
            else:
                current_thresholds[i] = getattr(self.args, "fixed_threshold", 0.0)

        obs, _ = env.reset()
        global_step = 0
        storage = RolloutBuffer(
            self.args.n_steps,
            obs.shape[1:],
            env.action_space.shape[0] if not self.discrete_action else 1,
        )

        for step in range(1, total_timesteps + 1):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            th_tensor = (
                torch.tensor(current_thresholds, dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
                if self.use_cvi
                else None
            )
            dist = self.forward_actor(obs_tensor, th_tensor)
            action_tensor = dist.sample()
            log_prob = (
                dist.log_prob(action_tensor)
                if self.discrete_action
                else dist.log_prob(action_tensor).sum(-1)
            )
            action = action_tensor.cpu().numpy()

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, infos = step_out
                done = np.logical_or(terminated, truncated)
            else:
                next_obs, reward, done, infos = step_out

            costs = np.array([info.get("cost", 0.0) for info in infos])
            storage.add(
                obs,
                action,
                reward,
                costs,
                done,
                current_thresholds.copy(),
                log_prob.detach().cpu().numpy(),
            )
            obs = next_obs

            # Sample new thresholds at episode end when dynamic
            if getattr(self.args, "dynamic_threshold", False):
                for idx, d in enumerate(done):
                    if d:
                        current_thresholds[idx] = np.random.uniform(
                            self.args.threshold_min, self.args.threshold_max
                        )

            global_step += num_envs
            if evaluator and global_step % self.args.eval_freq == 0:
                evaluator.evaluate(global_step, self)
                if self.args.verbose:
                    print(f"SPS={int(global_step / (time() - start_time))}")

            if len(storage.obs) >= self.args.n_steps:
                batch = storage.get_all()
                states = torch.tensor(batch.obs, dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch.actions, dtype=torch.float32).to(self.device)
                rewards = torch.tensor(batch.rewards, dtype=torch.float32).to(self.device)
                costs = torch.tensor(batch.costs, dtype=torch.float32).to(self.device)
                dones = torch.tensor(batch.dones, dtype=torch.float32).to(self.device)
                thresholds = torch.tensor(batch.extra_info, dtype=torch.float32).to(self.device)
                log_probs_old = torch.tensor(batch.log_probs, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    values = self.forward_critic(
                        states, thresholds if self.use_vve else None
                    )

                adv, ret = compute_gae(
                    rewards, costs, dones, values, self.args.gamma, self.args.gae_lambda
                )
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                for _ in range(self.args.ppo_epochs):
                    dist_now = self.forward_actor(
                        states, thresholds if self.use_cvi else None
                    )
                    log_prob_now = (
                        dist_now.log_prob(actions.squeeze(-1))
                        if self.discrete_action
                        else dist_now.log_prob(actions).sum(-1)
                    )
                    ratio = (log_prob_now - log_probs_old).exp()
                    surr1 = ratio * adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.args.clip_ratio, 1.0 + self.args.clip_ratio)
                        * adv
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    values_pred = self.forward_critic(
                        states, thresholds if self.use_vve else None
                    )
                    value_loss = F.mse_loss(values_pred, ret)
                    loss = policy_loss + 0.5 * value_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                storage.clear()

    # Backwards compatibility with older interface
    def train_loop(self, total_timesteps: int) -> None:
        self.train(total_timesteps)