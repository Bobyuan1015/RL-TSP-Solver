import numpy as np
from typing import Dict, Any, Optional


class RewardCalculator:
    """Centralized reward calculation module for TSP."""
    
    def __init__(self, reward_type: str = "distance_based"):
        """
        Initialize reward calculator.
        
        Args:
            reward_type: Type of reward function to use
                - "distance_based": Negative distance as reward
                - "normalized_distance": Normalized negative distance
                - "sparse": Sparse reward (only at episode end)
                - "dense": Dense reward with distance and progress bonuses
                - "potential_based": Potential-based reward shaping
        """
        self.reward_type = reward_type
        
    def calculate_reward(self,
                        current_city: int,
                        next_city: int,
                        distance_matrix: np.ndarray,
                        visited: set,
                        path: list,
                        done: bool,
                        optimal_distance: Optional[float] = None,
                        step_count: int = 0,
                        n_cities: int = None) -> float:
        """
        Calculate reward for taking an action.
        
        Args:
            current_city: Current city index
            next_city: Next city index (action taken)
            distance_matrix: Distance matrix between cities
            visited: Set of visited cities
            path: Current path
            done: Whether episode is complete
            optimal_distance: Known optimal distance (if available)
            step_count: Current step count
            n_cities: Total number of cities
            
        Returns:
            Calculated reward
        """
        if self.reward_type == "distance_based":
            return self._distance_based_reward(current_city, next_city, distance_matrix)
        
        elif self.reward_type == "normalized_distance":
            return self._normalized_distance_reward(
                current_city, next_city, distance_matrix, optimal_distance
            )
        
        elif self.reward_type == "sparse":
            return self._sparse_reward(
                path, distance_matrix, done, optimal_distance
            )
        
        elif self.reward_type == "dense":
            return self._dense_reward(
                current_city, next_city, distance_matrix, visited, 
                path, done, step_count, n_cities
            )
        
        elif self.reward_type == "potential_based":
            return self._potential_based_reward(
                current_city, next_city, distance_matrix, visited, 
                path, done, n_cities
            )
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _distance_based_reward(self, 
                             current_city: int, 
                             next_city: int, 
                             distance_matrix: np.ndarray) -> float:
        """Simple negative distance reward."""
        distance = distance_matrix[current_city, next_city]
        return -distance
    
    def _normalized_distance_reward(self,
                                  current_city: int,
                                  next_city: int,
                                  distance_matrix: np.ndarray,
                                  optimal_distance: Optional[float] = None) -> float:
        """Normalized negative distance reward."""
        distance = distance_matrix[current_city, next_city]
        
        # Normalize by maximum distance in matrix
        max_distance = np.max(distance_matrix)
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        
        return -normalized_distance
    
    def _sparse_reward(self,
                      path: list,
                      distance_matrix: np.ndarray,
                      done: bool,
                      optimal_distance: Optional[float] = None) -> float:
        """Sparse reward only given at episode end."""
        if not done:
            return 0.0
        
        # Calculate total path distance
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i], path[i + 1]]
        
        if optimal_distance is not None:
            # Reward based on how close to optimal
            gap = (total_distance - optimal_distance) / optimal_distance
            return -gap  # Negative gap (closer to optimal = higher reward)
        else:
            # Just negative total distance
            return -total_distance
    
    def _dense_reward(self,
                     current_city: int,
                     next_city: int,
                     distance_matrix: np.ndarray,
                     visited: set,
                     path: list,
                     done: bool,
                     step_count: int,
                     n_cities: int) -> float:
        """Dense reward with multiple components."""
        # Base distance reward
        distance = distance_matrix[current_city, next_city]
        distance_reward = -distance
        
        # Progress bonus for visiting new cities
        progress_bonus = 0.1 if next_city not in visited else 0.0
        
        # Time penalty to encourage shorter paths
        time_penalty = -0.01
        
        # Completion bonus
        completion_bonus = 1.0 if done else 0.0
        
        # Efficiency bonus for completing quickly
        if done and n_cities is not None:
            expected_steps = n_cities  # Minimum possible steps
            efficiency_bonus = max(0, (expected_steps - step_count) * 0.05)
        else:
            efficiency_bonus = 0.0
        
        total_reward = (distance_reward + progress_bonus + 
                       time_penalty + completion_bonus + efficiency_bonus)
        
        return total_reward
    
    def _potential_based_reward(self,
                              current_city: int,
                              next_city: int,
                              distance_matrix: np.ndarray,
                              visited: set,
                              path: list,
                              done: bool,
                              n_cities: int) -> float:
        """Potential-based reward shaping."""
        # Base distance reward
        distance = distance_matrix[current_city, next_city]
        distance_reward = -distance
        
        # Potential function: negative minimum spanning tree estimate
        # of remaining unvisited cities
        unvisited = set(range(n_cities)) - visited
        
        if len(unvisited) <= 1:
            # No unvisited cities or only start city remaining
            potential_current = 0.0
            potential_next = 0.0
        else:
            potential_current = self._compute_mst_potential(
                current_city, unvisited, distance_matrix
            )
            
            # After taking action, next_city becomes visited
            unvisited_after = unvisited - {next_city}
            potential_next = self._compute_mst_potential(
                next_city, unvisited_after, distance_matrix
            )
        
        # Potential-based shaping: Φ(s') - Φ(s)
        potential_reward = potential_next - potential_current
        
        return distance_reward + potential_reward
    
    def _compute_mst_potential(self,
                             current_city: int,
                             unvisited: set,
                             distance_matrix: np.ndarray) -> float:
        """Compute minimum spanning tree estimate for potential function."""
        if len(unvisited) <= 1:
            return 0.0
        
        # Create subgraph of unvisited cities + current city
        cities = [current_city] + list(unvisited)
        
        # Use Prim's algorithm to compute MST
        visited_mst = {current_city}
        mst_cost = 0.0
        
        while len(visited_mst) < len(cities):
            min_edge = float('inf')
            
            for v in visited_mst:
                for u in cities:
                    if u not in visited_mst:
                        edge_cost = distance_matrix[v, u]
                        if edge_cost < min_edge:
                            min_edge = edge_cost
                            next_vertex = u
            
            if min_edge < float('inf'):
                visited_mst.add(next_vertex)
                mst_cost += min_edge
        
        return -mst_cost  # Negative because we want higher potential for lower cost
    
    def calculate_episode_metrics(self,
                                path: list,
                                distance_matrix: np.ndarray,
                                optimal_distance: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate episode-level metrics.
        
        Args:
            path: Complete path taken
            distance_matrix: Distance matrix
            optimal_distance: Known optimal distance
            
        Returns:
            Dictionary of metrics
        """
        # Total distance
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i], path[i + 1]]
        
        metrics = {
            'total_distance': total_distance,
            'path_length': len(path),
            'is_valid': self._is_valid_path(path, len(distance_matrix))
        }
        
        if optimal_distance is not None:
            gap = (total_distance - optimal_distance) / optimal_distance
            metrics.update({
                'gap_to_optimal': gap,
                'optimality_ratio': optimal_distance / total_distance
            })
        
        return metrics
    
    def _is_valid_path(self, path: list, n_cities: int) -> bool:
        """Check if path is a valid TSP solution."""
        if len(path) != n_cities + 1:  # Should visit all cities + return to start
            return False
        
        if path[0] != 0 or path[-1] != 0:  # Should start and end at city 0
            return False
        
        # Check all cities visited exactly once (except start/end)
        visited_cities = set(path[:-1])
        return len(visited_cities) == n_cities and visited_cities == set(range(n_cities))
    
    def get_reward_info(self) -> Dict[str, str]:
        """Get information about the current reward function."""
        reward_descriptions = {
            "distance_based": "Negative distance between cities",
            "normalized_distance": "Normalized negative distance (0 to -1 range)",
            "sparse": "Reward only at episode end based on total distance",
            "dense": "Multiple reward components: distance, progress, time, completion",
            "potential_based": "Distance reward + potential-based shaping using MST"
        }
        
        return {
            'reward_type': self.reward_type,
            'description': reward_descriptions.get(self.reward_type, "Unknown")
        }