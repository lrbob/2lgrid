from torch.distributions import Categorical, Normal

from common.imports import *
from common.utils import Linear, th_act_fns

class Agent(nn.Module):
    """Neural network-based agent for policy gradient methods, supporting both discrete and continuous action spaces.

    Attributes:
        critic (nn.Sequential): Critic network for value estimation.
        actor (nn.Sequential): Actor network for action selection.
        logstd (nn.Parameter): Log standard deviation for continuous action spaces.
    """

    def __init__(self, envs: gym.Env, args: Dict[str, Any], continuous_actions: bool):
        """
        Initialize the Agent with specified environment, arguments, and action type.

        Args:
            envs: The environment.
            args: Arguments for configuration.
            continuous_actions: Flag indicating continuous or discrete action space.
        """
        super().__init__()
        self.use_threshold = args.vve or args.cvi
        self.threshold_dim = 1

        # Critic network setup
        critic_layers = args.critic_layers
        act_str, act_fn = args.critic_act_fn, th_act_fns[args.critic_act_fn]
        layers = []
        critic_input_dim = np.prod(envs.single_observation_space.shape)
        if self.use_threshold:
            critic_input_dim += self.threshold_dim
        layers.extend([
            Linear(critic_input_dim, critic_layers[0], act_str),
            act_fn
        ])

        for idx, embed_dim in enumerate(critic_layers[1:], start=1): 
            layers.extend([Linear(critic_layers[idx-1], embed_dim, act_str), act_fn])
        layers.append(Linear(critic_layers[-1], 1, 'linear'))
        self.critic = nn.Sequential(*layers)

        # Actor network setup
        actor_layers = args.actor_layers
        act_str, act_fn = args.actor_act_fn, th_act_fns[args.actor_act_fn]
        layers = []
        actor_input_dim = np.prod(envs.single_observation_space.shape)
        if self.use_threshold:
            actor_input_dim += self.threshold_dim
        layers.extend([
            Linear(actor_input_dim, actor_layers[0], act_str),
            act_fn
        ])
        for idx, embed_dim in enumerate(actor_layers[1:], start=1): 
            layers.extend([Linear(actor_layers[idx-1], embed_dim, act_str), act_fn])
        
        # Final layer differs for continuous vs. discrete actions
        if continuous_actions:
            layers.extend([Linear(actor_layers[-1], np.prod(envs.single_action_space.shape), 'sigmoid'), nn.Sigmoid()])
            self.logstd = nn.Parameter(th.zeros(1, np.prod(envs.single_action_space.shape)))
            self.get_action = self.get_continuous_action
            self.get_eval_action = self.get_eval_continuous_action
        else:
            layers.append(Linear(actor_layers[-1], np.prod(envs.single_action_space.n)))
            self.get_action = self.get_discrete_action
            self.get_eval_action = self.get_eval_discrete_action
        self.actor = nn.Sequential(*layers)

    def get_value(self, x: th.Tensor, threshold: Optional[th.Tensor] = None) -> th.Tensor:
        """Compute value estimate (critic output) for given observations.

        Args:
            x: Input observations.

        Returns:
            A tensor containing value estimates.
        """
        if self.use_threshold and threshold is not None:
            x = th.cat([x, threshold], dim=1)
        return self.critic(x)

    def get_discrete_action(self, x: th.Tensor, action: th.Tensor = None, threshold: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample discrete actions and compute log probabilities and entropy.

        Args:
            x: Input observations.
            action: Specific action to take. Defaults to None.

        Returns:
            A tuple containing tensors for the sampled discrete actions, the log probability of the sampled actions, and the entropy of the action distribution.
        """
        if self.use_threshold and threshold is not None:
            x = th.cat([x, threshold], dim=1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_continuous_action(self, x: th.Tensor, action: th.Tensor = None, threshold: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Sample continuous actions and compute log probabilities and entropy.

        Args:
            x: Input observations.
            action : Specific action to take. Defaults to None.

        Returns:
            A tuple containing tensors for the sampled continuous actions, the log probability of the sampled actions, and the entropy of the action distribution.
        """
        if self.use_threshold and threshold is not None:
            x = th.cat([x, threshold], dim=1)
        action_mu = self.actor(x)
        action_logstd = self.logstd.expand_as(action_mu)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mu, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)
    
    def get_eval_discrete_action(self, x: th.Tensor, threshold: Optional[th.Tensor] = None) -> th.Tensor:
        """Evaluate discrete actions without exploration.

        Args:
            x: Input observations.

        Returns:
            A tensor with deterministic discrete actions for evaluation.
        """
        return self.get_discrete_action(x, threshold=threshold)[0]
      
    def get_eval_continuous_action(self, x: th.Tensor, threshold: Optional[th.Tensor] = None) -> th.Tensor:
        """Evaluate continuous actions without exploration.

        Args:
            x: Input observations.

        Returns:
            A tensor with deterministic continuous actions for evaluation
        """
        if self.use_threshold and threshold is not None:
            x = th.cat([x, threshold], dim=1)
        action_mu = self.actor(x)
        return action_mu
