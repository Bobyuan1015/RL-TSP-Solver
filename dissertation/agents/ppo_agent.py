import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent


class PPOActorNetwork(nn.Module):
    """Actor network for PPO."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0):
        super(PPOActorNetwork, self).__init__()
        
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
        
        return logits


class PPOCriticNetwork(nn.Module):
    """Critic network for PPO."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0):
        super(PPOCriticNetwork, self).__init__()
        
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
        
        layers.append(nn.Linear(prev_dim, 1))
        
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


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent for TSP."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 state_components: List[str],
                 device: str = "cpu"):
        super().__init__(state_dim, action_dim, config, device)
        
        self.state_components = state_components
        
        # PPO parameters
        self.actor_learning_rate = config.get('actor_learning_rate', 0.0003)
        self.critic_learning_rate = config.get('critic_learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coefficient = config.get('entropy_coefficient', 0.01)
        self.value_loss_coefficient = config.get('value_loss_coefficient', 0.5)
        self.epochs_per_update = config.get('epochs_per_update', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # Networks
        actor_hidden_dims = config.get('actor_hidden_dims', [512, 256, 128])
        critic_hidden_dims = config.get('critic_hidden_dims', [512, 256, 128])
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        self.actor = PPOActorNetwork(
            state_dim, action_dim, actor_hidden_dims, activation, dropout
        ).to(self.device)
        
        self.critic = PPOCriticNetwork(
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
        self.episode_next_values = []
        
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
        """Select action using PPO actor network."""
        state_array = self._state_to_tensor(state)
        action_mask = self._get_action_mask(state)
        
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            # Get action logits and state value
            logits = self.actor(state_tensor, mask_tensor)
            value = self.critic(state_tensor)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
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
        
        # Store next state value if not done
        if not experience['done'] and 'next_state' in experience:
            next_state_array = self._state_to_tensor(experience['next_state'])
            next_state_tensor = torch.FloatTensor(next_state_array).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).item()
            self.episode_next_values.append(next_value)
        else:
            self.episode_next_values.append(0.0)  # Terminal state value is 0
        
        # If episode is done, perform PPO update
        if experience['done']:
            return self._update_ppo()
        
        return {'total_loss': float('inf')}
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        return advantages
    
    def _update_ppo(self) -> Dict[str, float]:
        """Update actor and critic networks using PPO."""
        if len(self.episode_rewards) == 0:
            return {'total_loss': float('inf')}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        rewards = torch.FloatTensor(self.episode_rewards).to(self.device)
        values = torch.FloatTensor(self.episode_values).to(self.device)
        next_values = torch.FloatTensor(self.episode_next_values).to(self.device)
        masks = torch.FloatTensor(np.array(self.episode_masks)).to(self.device)
        
        # Compute advantages and returns
        advantages = self._compute_gae(rewards, values, next_values)
        returns = advantages + values
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        # PPO update for multiple epochs
        for epoch in range(self.epochs_per_update):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = masks[batch_indices]
                
                # Compute current policy distribution and values
                logits = self.actor(batch_states, batch_masks)
                current_values = self.critic(batch_states)
                
                # Get current log probabilities
                dist = torch.distributions.Categorical(logits=logits)
                current_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute probability ratio
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Compute PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Total loss
                loss = (policy_loss + 
                       self.value_loss_coefficient * value_loss - 
                       self.entropy_coefficient * entropy)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_loss += loss.item()
        
        # Clear episode storage
        episode_return = sum(self.episode_rewards)
        self._reset_episode_storage()
        
        self.training_step += 1
        
        return {
            'total_loss': total_loss / (self.epochs_per_update * max(1, len(states) // self.mini_batch_size)),
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
        self.episode_next_values = []
    
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