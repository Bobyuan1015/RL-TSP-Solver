import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int],
                 activation: str = "relu",
                 dropout: float = 0.0):
        """
        Initialize policy network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (number of actions)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
        """
        super(PolicyNetwork, self).__init__()
        
        # Choose activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            mask: Action mask (1 for valid actions, 0 for invalid)
            
        Returns:
            Action probabilities
        """
        logits = self.network(x)
        
        # Apply action mask to logits
        if mask is not None:
            # Set logits of invalid actions to very negative value
            logits = logits + (mask - 1) * 1e9
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs


class ReinforceAgent(BaseAgent):
    """REINFORCE (Policy Gradient) agent for TSP."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 state_components: List[str],
                 device: str = "cpu"):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions (cities)
            config: Agent configuration
            state_components: List of state components to use
            device: Device for computations
        """
        super().__init__(state_dim, action_dim, config, device)
        
        self.state_components = state_components
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        
        # Policy network
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        self.policy_network = PolicyNetwork(
            state_dim, action_dim, hidden_dims, activation, dropout
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_masks = []
        self.episode_log_probs = []
        
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
        action_mask = 1 - visited_mask  # 1 for unvisited (valid), 0 for visited (invalid)
        
        # If all cities visited, only returning to start (city 0) is valid
        if np.sum(action_mask) == 0:
            action_mask = np.zeros(self.action_dim)
            action_mask[0] = 1
        
        return action_mask.astype(np.float32)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using policy network.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            
        Returns:
            Selected action (city index)
        """
        state_array = self._state_to_tensor(state)
        action_mask = self._get_action_mask(state)
        
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        # Get action probabilities
        with torch.no_grad() if not training else torch.enable_grad():
            probs = self.policy_network(state_tensor, mask_tensor)
            
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
                
                return action.item()
            else:
                # Greedy action selection for evaluation
                action = probs.argmax().item()
                return action
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent based on episode experience.
        
        Args:
            experience: Experience dictionary with reward information
            
        Returns:
            Dictionary containing training metrics
        """
        # Store episode reward
        self.episode_rewards.append(experience['reward'])
        
        # If episode is done, perform policy update
        if experience['done']:
            return self._update_policy()
        
        return {'loss': float('inf')}
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return {'loss': float('inf')}
        
        # Compute discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        log_probs = torch.FloatTensor(self.episode_log_probs).to(self.device)
        policy_loss = -(log_probs * returns).sum()
        
        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Clear episode storage
        episode_return = sum(self.episode_rewards)
        self._reset_episode_storage()
        
        self.training_step += 1
        
        return {
            'loss': policy_loss.item(),
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
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self._reset_episode_storage()
    
    def save_model(self, filepath: str):
        """Save agent model."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']