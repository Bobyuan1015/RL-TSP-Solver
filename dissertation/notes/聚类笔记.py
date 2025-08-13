把下面训练PolicyNetwork解决tsp聚类子tsp代码 按照Reinforce类的方式修改，这样可以无缝插入到项目中：
1.计算reward的时候要求考虑 子tsp的最优解，不能暴力的顺序相加，因为访问顺序不同，路径长度也不同。
2.另外子stp 与子tsp的访问顺序也存在这个问题，所以并不能武断的把所有子tsp相加为最后的reward
3.子tsp 到 另外子tsp的时候，需要策略选择子每个子tsp的点
4.要求完全按照Reinforce类的各个函数实现，比如（其中函数的参数 和返回值不能变，否则无法适配到项目代码中，因为项目代码已经适配好Reinforce类）




class PolicyNetwork(nn.Module):
    def __init__(self, node_dim=2, hidden_dim=16, n_subtours=2):
        super().__init__()
        self.gnn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.policy_head = nn.Linear(hidden_dim, n_subtours)

    def forward(self, x):
        features = self.gnn(x)
        logits = self.policy_head(features)
        return torch.softmax(logits, dim=-1)

def compute_reward(city_coords, actions):
    subtours = {}
    for i, action in enumerate(actions):
        if action not in subtours:
            subtours[action] = []
        subtours[action].append(i)

    total_length = 0
    for subtour in subtours.values():
        # 计算子路径长度（按顺序连接）
        for i in range(len(subtour)):
            x1, y1 = city_coords[subtour[i]]
            x2, y2 = city_coords[subtour[(i + 1) % len(subtour)]]
            total_length += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return -total_length  # 负号因为要最小化距离


def train(model, city_coords, optimizer, n_epochs=200):
    rewards = []
    for epoch in range(n_epochs):
        # 1. 采样动作
        probs = model(city_coords)
        actions = [torch.multinomial(p, 1).item() for p in probs]

        # 2. 计算奖励
        reward = compute_reward(city_coords, actions)
        rewards.append(reward)

        # 3. 计算损失（Policy Gradient）
        loss = 0
        for i, action in enumerate(actions):
            loss += -torch.log(probs[i][action]) * reward

        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Reward: {reward:.2f}")

    return rewards



class REINFORCE(BaseAgent):
    """
    REINFORCE agent for TSP (version 1.2).
    Uses same input and reward strategy as DQN Basic.
    """

    def __init__(self,
                 n_cities: int,
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 hidden_sizes: List[int] = [256, 256, 256],
                 device: str = 'cpu',
                 seed: Optional[int] = None,
                 save_model_diagram: bool = True):
        """
        Initialize REINFORCE agent.

        Args:
            n_cities: Number of cities
            lr: Learning rate
            gamma: Discount factor
            hidden_sizes: Hidden layer sizes for policy network
            device: Device to use
            seed: Random seed
        """
        super().__init__(n_cities, lr, device, seed, save_model_diagram)

        self.gamma = gamma

        # Input size is n_cities (one-hot encoding)
        input_size = n_cities

        # Create policy network
        self.policy_network = MLP(input_size, hidden_sizes, n_cities).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Episode storage
        self.episode_log_probs = []
        self.episode_rewards = []

        self.logger.info(
            f"Initialized REINFORCE with {sum(p.numel() for p in self.policy_network.parameters())} parameters")

        # Save architecture diagram
        config_dict = {
            'n_cities': n_cities,
            'lr': lr,
            'gamma': gamma,
            'hidden_sizes': hidden_sizes,
            'device': device,
            'seed': seed
        }
        # self.save_architecture_diagram(config_dict)

    def select_action(self, state: Dict[str, Any], valid_actions: List[int]) -> int:
        """
        Select action using policy network.

        Args:
            state: Current state containing 'current_city_onehot'
            valid_actions: List of valid actions

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state['visited_mask']).unsqueeze(0).to(self.device)

        # Get action probabilities
        logits = self.policy_network(state_tensor)

        # Mask invalid actions
        mask = torch.full_like(logits, -float('inf'))
        for action in valid_actions:
            mask[0, action] = 0.0

        masked_logits = logits + mask

        # Sample action
        probs = F.softmax(masked_logits, dim=1)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        # Store log probability for training
        log_prob = action_dist.log_prob(action)
        self.episode_log_probs.append(log_prob)

        return action.item()

    def store_reward(self, reward: float) -> None:
        """Store reward for current episode."""
        self.episode_rewards.append(reward)

    def update(self, *args, **kwargs) -> float:
        """
        Update policy network at end of episode using REINFORCE.

        Returns:
            Loss value
        """
        if len(self.episode_rewards) == 0:
            return 0.0

        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns (optional, can help with training stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.episode_log_probs = []
        self.episode_rewards = []

        return policy_loss.item()

    def save_model(self, filepath: str) -> None:
        """Save model parameters."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {filepath}")

    def set_eval_mode(self) -> None:
        """Set agent to evaluation mode."""
        self.policy_network.eval()

    def set_train_mode(self) -> None:
        """Set agent to training mode."""
        self.policy_network.train()

    def get_action_probs(self, state: Dict[str, Any], valid_actions: List[int]) -> np.ndarray:
        """Get action probabilities for current state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state['current_city_onehot']).unsqueeze(0).to(self.device)
            logits = self.policy_network(state_tensor)

            # Mask invalid actions
            mask = torch.full_like(logits, -float('inf'))
            for action in valid_actions:
                mask[0, action] = 0.0

            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=1)

            return probs.cpu().numpy()[0]