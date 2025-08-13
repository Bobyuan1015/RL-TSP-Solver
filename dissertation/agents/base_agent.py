import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """Base class for all RL agents."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 config: Dict[str, Any],
                 device: str = "cpu"):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of cities)
            config: Agent configuration
            device: Device to use for computations
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.episode_losses = []
        
    @abstractmethod
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action given current state.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            
        Returns:
            Selected action (city index)
        """
        pass
    
    @abstractmethod
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent based on experience.
        
        Args:
            experience: Experience data
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save agent model to file."""
        pass
    
    @abstractmethod  
    def load_model(self, filepath: str):
        """Load agent model from file."""
        pass
    
    def reset_episode(self):
        """Reset episode-specific state."""
        pass
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get training statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'avg_reward': np.mean(self.episode_rewards[-100:]),
            'avg_loss': np.mean(self.episode_losses[-100:]) if self.episode_losses else 0.0,
            'training_steps': self.training_step
        }


class BaseDQN(nn.Module):
    """Base DQN network with common functionality."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", dropout: float = 0.0):
        """
        Initialize base DQN network.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (number of actions)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
        """
        super(BaseDQN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Choose activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
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
            Q-values for each action
        """
        q_values = self.network(x)
        
        # Apply action mask
        if mask is not None:
            # Set Q-values of invalid actions to very negative value
            q_values = q_values + (mask - 1) * 1e9
        
        return q_values