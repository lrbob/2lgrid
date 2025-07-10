import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def get_all(self):
        class Batch:
            pass
        b = Batch()
        b.obs = np.array(self.obs)
        b.actions = np.array(self.actions)
        b.rewards = np.array(self.rewards)
        b.costs = np.array(self.costs)
        b.dones = np.array(self.dones)
        b.extra_info = np.array(self.extra_info)
        b.log_probs = np.array(self.log_probs)
        return b

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.extra_info = []
        self.log_probs = []


def compute_gae(rewards, costs, dones, values, gamma, gae_lambda, lambda_coef=0.0, thresholds=None):
    """Compute generalized advantage estimation with optional cost penalty."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[t]
        else:
            next_value = values[t + 1]
        delta = rewards[t] - lambda_coef * costs[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
    returns = advantages + values
    return advantages, returns


class ConditionedPPO(nn.Module):
    """Conditioned PPO agent with optional threshold conditioning."""

    def __init__(self, env, args):
        super().__init__()
        self.env = env
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_shape = env.observation_space.shape
        act_space = env.action_space

        if act_space.__class__.__name__ == "Discrete":
            self.discrete_action = True
            action_dim = act_space.n
        else:
            self.discrete_action = False
            action_dim = act_space.shape[0]

        state_dim = obs_shape[0]

        self.use_cvi = getattr(args, "cvi", False)
        self.use_vve = getattr(args, "vve", False)
        th_dim = 1
        th_enc_dim = getattr(args, "threshold_embed_dim", 16)

        actor_input_dim = state_dim
        if self.use_cvi:
            if getattr(args, "threshold_encoding", "raw") == "raw":
                actor_input_dim += th_dim
                self.threshold_encoder_actor = None
            else:
                self.threshold_encoder_actor = nn.Sequential(
                    nn.Linear(th_dim, th_enc_dim), nn.ReLU(),
                    nn.Linear(th_enc_dim, th_enc_dim), nn.ReLU()
                )
                actor_input_dim += th_enc_dim
        else:
            self.threshold_encoder_actor = None

        hidden = 256
        self.actor_fc1 = nn.Linear(actor_input_dim, hidden)
        self.actor_fc2 = nn.Linear(hidden, hidden)
        if self.discrete_action:
            self.actor_out = nn.Linear(hidden, action_dim)
        else:
            self.actor_mu = nn.Linear(hidden, action_dim)
            self.actor_logstd = nn.Linear(hidden, action_dim)

        critic_input_dim = state_dim
        if self.use_vve:
            if getattr(args, "threshold_encoding", "raw") == "raw":
                critic_input_dim += th_dim
                self.threshold_encoder_critic = None
            else:
                self.threshold_encoder_critic = nn.Sequential(
                    nn.Linear(th_dim, th_enc_dim), nn.ReLU(),
                    nn.Linear(th_enc_dim, th_enc_dim), nn.ReLU()
                )
                critic_input_dim += th_enc_dim
        else:
            self.threshold_encoder_critic = None
        self.critic_fc1 = nn.Linear(critic_input_dim, hidden)
        self.critic_fc2 = nn.Linear(hidden, hidden)
        self.critic_out = nn.Linear(hidden, 1)

        self.actor_optimizer = torch.optim.Adam(self.parameters_actor(), lr=getattr(args, "lr_actor", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.parameters_critic(), lr=getattr(args, "lr_critic", 3e-4))

    def parameters_actor(self):
        params = [self.actor_fc1.parameters(), self.actor_fc2.parameters()]
        if self.discrete_action:
            params.append(self.actor_out.parameters())
        else:
            params.extend([self.actor_mu.parameters(), self.actor_logstd.parameters()])
        if self.threshold_encoder_actor:
            params.append(self.threshold_encoder_actor.parameters())
        return [p for sub in params for p in sub]

    def parameters_critic(self):
        params = [self.critic_fc1.parameters(), self.critic_fc2.parameters(), self.critic_out.parameters()]
        if self.threshold_encoder_critic:
            params.append(self.threshold_encoder_critic.parameters())
        return [p for sub in params for p in sub]

    def forward_actor(self, obs, thresholds=None):
        if self.use_cvi and thresholds is not None:
            if self.threshold_encoder_actor:
                th_feat = self.threshold_encoder_actor(thresholds)
                x = torch.cat([obs, th_feat], dim=-1)
            else:
                x = torch.cat([obs, thresholds], dim=-1)
        else:
            x = obs
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

    def train_loop(self, total_timesteps):
        env = self.env
        num_envs = getattr(env, "num_envs", 1)
        current_thresholds = np.zeros(num_envs)
        for i in range(num_envs):
            if getattr(self.args, "dynamic_threshold", False):
                current_thresholds[i] = np.random.uniform(self.args.threshold_min, self.args.threshold_max)
            else:
                current_thresholds[i] = getattr(self.args, "fixed_threshold", 0.0)

        obs, _ = env.reset()
        storage = RolloutBuffer(self.args.n_steps, obs.shape[1:], env.action_space.shape[0] if not self.discrete_action else 1)

        for global_step in range(1, total_timesteps + 1):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            if self.use_cvi:
                th_tensor = torch.tensor(current_thresholds, dtype=torch.float32).unsqueeze(1).to(self.device)
            else:
                th_tensor = None
            dist = self.forward_actor(obs_tensor, th_tensor)
            action_tensor = dist.sample()
            if self.discrete_action:
                log_prob = dist.log_prob(action_tensor)
            else:
                log_prob = dist.log_prob(action_tensor).sum(-1)
            action = action_tensor.cpu().numpy()
            next_obs, reward, done, infos = env.step(action)
            costs = np.array([info.get('cost', 0.0) for info in infos])
            storage.add(obs, action, reward, costs, done, current_thresholds.copy(), log_prob.detach().cpu().numpy())
            obs = next_obs
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
                    values = self.forward_critic(states, thresholds if self.use_vve else None)
                adv, ret = compute_gae(rewards, costs, dones, values, self.args.gamma, self.args.gae_lambda)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                for _ in range(self.args.ppo_epochs):
                    dist_now = self.forward_actor(states, thresholds if self.use_cvi else None)
                    if self.discrete_action:
                        log_prob_now = dist_now.log_prob(actions.squeeze(-1))
                    else:
                        log_prob_now = dist_now.log_prob(actions).sum(-1)
                    ratio = (log_prob_now - log_probs_old).exp()
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.args.clip_ratio, 1.0 + self.args.clip_ratio) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    values_pred = self.forward_critic(states, thresholds if self.use_vve else None)
                    value_loss = F.mse_loss(values_pred, ret)
                    loss = policy_loss + 0.5 * value_loss
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                storage.clear()