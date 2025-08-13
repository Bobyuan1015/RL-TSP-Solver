import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """Actor network for Actor-Critic."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0):
        super(ActorNetwork, self).__init__()
        
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.network(x)
        
        if mask is not None:
            logits = logits + (mask - 1) * 1e9
        
        probs = F.softmax(logits, dim=-1)
        return probs


class CriticNetwork(nn.Module):
    """Critic network for Actor-Critic."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0):
        super(CriticNetwork, self).__init__()
        
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Single value output
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.network(x)
        return value.squeeze(-1)


class ActorCriticAgent(BaseAgent):
    """Actor-Critic agent for TSP."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 state_components: List[str],
                 device: str = "cpu"):
        super().__init__(state_dim, action_dim, config, device)
        
        self.state_components = state_components
        
        # Training parameters
        self.actor_learning_rate = config.get('actor_learning_rate', 0.001)
        self.critic_learning_rate = config.get('critic_learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        
        # Networks
        actor_hidden_dims = config.get('actor_hidden_dims', [512, 256, 128])
        critic_hidden_dims = config.get('critic_hidden_dims', [512, 256, 128])
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        self.actor = ActorNetwork(
            state_dim, action_dim, actor_hidden_dims, activation, dropout
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim, critic_hidden_dims, activation, dropout
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_masks = []
        self.episode_log_probs = []
        self.episode_values = []
        
    def _state_to_tensor(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to numpy array."""
        state_vector = []
        
        for component in self.state_components:
            if component in state:
                if isinstance(state[component], np.ndarray):
                    if state[component].ndim == 0:  # scalar
                        state_vector.append(float(state[component]))
                    else:
                        state_vector.extend(state[component].tolist())
                else:
                    state_vector.append(float(state[component]))
        
        return np.array(state_vector, dtype=np.float32)
    
    def _get_action_mask(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action mask from state."""
        visited_mask = state.get('visited_mask', np.zeros(self.action_dim))
        action_mask = 1 - visited_mask
        
        if np.sum(action_mask) == 0:
            action_mask = np.zeros(self.action_dim)
            action_mask[0] = 1
        
        return action_mask.astype(np.float32)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """Select action using actor network."""
        state_array = self._state_to_tensor(state)
        action_mask = self._get_action_mask(state)
        
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            # Get action probabilities and state value
            probs = self.actor(state_tensor, mask_tensor)
            value = self.critic(state_tensor)
            
            if training:
                # Sample action from probability distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Store for episode update
                self.episode_states.append(state_array)
                self.episode_actions.append(action.item())
                self.episode_masks.append(action_mask)
                self.episode_log_probs.append(log_prob.item())
                self.episode_values.append(value.item())
                
                return action.item()
            else:
                # Greedy action selection for evaluation
                action = probs.argmax().item()
                return action
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent based on episode experience."""
        # Store episode reward
        self.episode_rewards.append(experience['reward'])
        
        # If episode is done, perform update
        if experience['done']:
            return self._update_networks()
        
        return {'actor_loss': float('inf'), 'critic_loss': float('inf')}
    
    def _update_networks(self) -> Dict[str, float]:
        """Update actor and critic networks."""
        if len(self.episode_rewards) == 0:
            return {'actor_loss': float('inf'), 'critic_loss': float('inf')}
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.episode_values).to(self.device)
        log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        
        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        actor_loss = -(log_probs * advantages.detach()).sum()
        critic_loss = F.mse_loss(values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Clear episode storage
        episode_return = sum(self.episode_rewards)
        self._reset_episode_storage()
        
        self.training_step += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'episode_return': episode_return,
            'episode_length': len(self.episode_rewards)
        }
    
    def _reset_episode_storage(self):
        """Clear episode storage."""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_masks = []
        self.episode_log_probs = []
        self.episode_values = []
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self._reset_episode_storage()
    
    def save_model(self, filepath: str):
        """Save agent model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint['training_step']