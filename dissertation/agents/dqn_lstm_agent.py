import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

from .base_agent import BaseAgent


class DQNLSTMNetwork(nn.Module):
    """DQN network with LSTM for sequence processing."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 lstm_hidden_dim: int = 256,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.2,
                 mlp_hidden_dims: List[int] = [128],
                 activation: str = "relu"):
        """
        Initialize DQN LSTM network.
        
        Args:
            input_dim: Input dimension per timestep
            output_dim: Output dimension (number of actions)
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            lstm_dropout: LSTM dropout
            mlp_hidden_dims: MLP hidden dimensions after LSTM
            activation: Activation function
        """
        super(DQNLSTMNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # Choose activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # MLP layers after LSTM
        mlp_layers = []
        prev_dim = lstm_hidden_dim
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Initialize MLP weights
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                state_seq: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            state_seq: State sequence [batch_size, seq_len, input_dim]
            hidden: Previous hidden state tuple (h, c)
            mask: Action mask for final timestep
            
        Returns:
            Tuple of (q_values, new_hidden_state)
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(state_seq, hidden)
        
        # Take output from last timestep
        last_output = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # MLP forward pass
        q_values = self.mlp(last_output)
        
        # Apply action mask
        if mask is not None:
            q_values = q_values + (mask - 1) * 1e9
        
        return q_values, hidden
    
    def init_hidden(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim, device=device)
        return h, c


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Get buffer size."""
        return len(self.buffer)


class DQNLSTMAgent(BaseAgent):
    """DQN agent with LSTM for handling sequential state information."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 state_components: List[str],
                 device: str = "cpu"):
        """
        Initialize DQN LSTM agent.
        
        Args:
            state_dim: State dimension per timestep
            action_dim: Number of actions (cities)
            config: Agent configuration
            state_components: List of state components to use
            device: Device for computations
        """
        super().__init__(state_dim, action_dim, config, device)
        
        self.state_components = state_components
        
        # Network parameters
        self.lstm_hidden_dim = config.get('lstm_hidden_dim', 256)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)
        self.lstm_dropout = config.get('lstm_dropout', 0.2)
        self.mlp_hidden_dims = config.get('mlp_hidden_dims', [128])
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 100)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        
        # Initialize networks
        self.q_network = DQNLSTMNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_num_layers=self.lstm_num_layers,
            lstm_dropout=self.lstm_dropout,
            mlp_hidden_dims=self.mlp_hidden_dims,
            activation=config.get('activation', 'relu')
        ).to(self.device)
        
        self.target_network = DQNLSTMNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_num_layers=self.lstm_num_layers,
            lstm_dropout=self.lstm_dropout,
            mlp_hidden_dims=self.mlp_hidden_dims,
            activation=config.get('activation', 'relu')
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Episode state
        self.epsilon = self.epsilon_start
        self.hidden_state = None
        self.state_sequence = []
        
    def _state_to_tensor(self, state: Dict[str, Any]) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        state_vector = []
        
        for component in self.state_components:
            if component in state:
                if isinstance(state[component], np.ndarray):
                    if state[component].ndim == 0:  # scalar
                        state_vector.append([float(state[component])])
                    else:
                        state_vector.extend(state[component].tolist())
                else:
                    state_vector.append([float(state[component])])
        
        return torch.FloatTensor(state_vector).to(self.device)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with LSTM.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            
        Returns:
            Selected action (city index)
        """
        # Convert state to tensor and add to sequence
        state_tensor = self._state_to_tensor(state)
        self.state_sequence.append(state_tensor)
        
        # Get action mask
        valid_actions = state.get('valid_actions', list(range(self.action_dim)))
        action_mask = torch.zeros(self.action_dim, device=self.device)
        action_mask[valid_actions] = 1.0
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action from valid actions
            action = random.choice(valid_actions)
        else:
            # Greedy action using Q-network
            with torch.no_grad():
                self.q_network.eval()
                
                # Prepare sequence tensor
                seq_tensor = torch.stack(self.state_sequence).unsqueeze(0)  # [1, seq_len, state_dim]
                
                # Forward pass
                q_values, self.hidden_state = self.q_network(
                    seq_tensor, self.hidden_state, action_mask.unsqueeze(0)
                )
                
                # Select action with highest Q-value among valid actions
                action = q_values.argmax().item()
                
                self.q_network.train()
        
        return action
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent based on experience.
        
        Args:
            experience: Experience dictionary containing episode data
            
        Returns:
            Dictionary containing training metrics
        """
        # Add experience to replay buffer
        self.memory.push(experience)
        
        # Skip update if not enough experiences
        if len(self.memory) < self.batch_size:
            return {'loss': float('inf')}
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        
        # Compute loss and update
        loss = self._compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_step += 1
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def _compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute DQN loss for batch."""
        # This is a simplified version - in practice, you'd need to handle
        # variable-length sequences and proper batching for LSTM
        
        losses = []
        
        for experience in batch:
            state_seq = torch.stack([self._state_to_tensor(s) for s in experience['states']]).unsqueeze(0)
            action = torch.LongTensor([experience['action']]).to(self.device)
            reward = torch.FloatTensor([experience['reward']]).to(self.device)
            next_state_seq = torch.stack([self._state_to_tensor(s) for s in experience['next_states']]).unsqueeze(0)
            done = torch.BoolTensor([experience['done']]).to(self.device)
            
            # Current Q-value
            q_values, _ = self.q_network(state_seq)
            current_q = q_values.gather(1, action.unsqueeze(1))
            
            # Next Q-value from target network
            with torch.no_grad():
                next_q_values, _ = self.target_network(next_state_seq)
                next_q = next_q_values.max(1)[0].detach()
                target_q = reward + (self.gamma * next_q * ~done)
            
            # Compute loss
            loss = F.mse_loss(current_q.squeeze(), target_q)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.hidden_state = None
        self.state_sequence = []
    
    def save_model(self, filepath: str):
        """Save agent model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load agent model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']