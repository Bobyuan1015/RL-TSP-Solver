import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, BaseDQN


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, mask: np.ndarray, next_mask: np.ndarray):
        """Add experience to buffer."""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'mask': mask,
            'next_mask': next_mask
        })
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e['state'] for e in batch])
        actions = torch.LongTensor([e['action'] for e in batch])
        rewards = torch.FloatTensor([e['reward'] for e in batch])
        next_states = torch.FloatTensor([e['next_state'] for e in batch])
        dones = torch.BoolTensor([e['done'] for e in batch])
        masks = torch.FloatTensor([e['mask'] for e in batch])
        next_masks = torch.FloatTensor([e['next_mask'] for e in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'masks': masks,
            'next_masks': next_masks
        }
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for TSP."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 state_components: List[str],
                 device: str = "cpu"):
        """
        Initialize DQN agent.
        
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
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 100)
        self.batch_size = config.get('batch_size', 64)
        self.memory_size = config.get('memory_size', 10000)
        
        # Networks
        hidden_dims = config.get('hidden_dims', [512, 256, 128])
        activation = config.get('activation', 'relu')
        dropout = config.get('dropout', 0.2)
        
        self.q_network = BaseDQN(
            state_dim, action_dim, hidden_dims, activation, dropout
        ).to(self.device)
        
        self.target_network = BaseDQN(
            state_dim, action_dim, hidden_dims, activation, dropout
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Training state
        self.epsilon = self.epsilon_start
        
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
        # Use visited mask to determine valid actions
        visited_mask = state.get('visited_mask', np.zeros(self.action_dim))
        action_mask = 1 - visited_mask  # 1 for unvisited (valid), 0 for visited (invalid)
        
        # If all cities visited, only returning to start (city 0) is valid
        if np.sum(action_mask) == 0:
            action_mask = np.zeros(self.action_dim)
            action_mask[0] = 1
        
        return action_mask.astype(np.float32)
    
    def select_action(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            training: Whether in training mode
            
        Returns:
            Selected action (city index)
        """
        # Get action mask
        action_mask = self._get_action_mask(state)
        valid_actions = np.where(action_mask > 0)[0]
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action from valid actions
            action = int(np.random.choice(valid_actions))
        else:
            # Greedy action using Q-network
            state_tensor = torch.FloatTensor(self._state_to_tensor(state)).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor, mask_tensor)
                action = q_values.argmax().item()
        
        return action
    
    def update(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent based on experience.
        
        Args:
            experience: Experience dictionary with state, action, reward, next_state, done
            
        Returns:
            Dictionary containing training metrics
        """
        # Add experience to replay buffer
        state_array = self._state_to_tensor(experience['state'])
        next_state_array = self._state_to_tensor(experience['next_state'])
        mask = self._get_action_mask(experience['state'])
        next_mask = self._get_action_mask(experience['next_state'])
        
        self.memory.push(
            state_array,
            experience['action'],
            experience['reward'],
            next_state_array,
            experience['done'],
            mask,
            next_mask
        )
        
        # Skip update if not enough experiences
        if len(self.memory) < self.batch_size:
            return {'loss': float('inf'), 'epsilon': self.epsilon}
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # Compute loss
        loss = self._compute_loss(batch)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        if self.training_step % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.training_step += 1
        
        return {'loss': loss.item(), 'epsilon': self.epsilon}
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DQN loss for batch."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        masks = batch['masks']
        next_masks = batch['next_masks']
        
        # Current Q-values
        current_q_values = self.q_network(states, masks)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_masks)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        return loss
    
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