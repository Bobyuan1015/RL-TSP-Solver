import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import time

from dissertation.env.tsp_env import TSPEnvironment
from dissertation.agents.base_agent import BaseAgent


class Evaluator:
    """Evaluation module for TSP agents."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize evaluator.
        
        Args:
            device: Device for computations
        """
        self.device = device
        
    def evaluate_agent(self, 
                      agent: BaseAgent,
                      env: TSPEnvironment,
                      num_episodes: int = 100,
                      render: bool = False,
                      save_trajectories: bool = False) -> Dict[str, Any]:
        """
        Evaluate agent performance.
        
        Args:
            agent: Agent to evaluate
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            save_trajectories: Whether to save episode trajectories
            
        Returns:
            Dictionary containing evaluation metrics
        """
        agent.q_network.eval() if hasattr(agent, 'q_network') else None
        
        episode_rewards = []
        episode_lengths = []
        episode_distances = []
        gap_to_optimal = []
        valid_solutions = []
        trajectories = [] if save_trajectories else None
        
        for episode in range(num_episodes):
            episode_reward, episode_length, episode_distance, gap, is_valid, trajectory = \
                self._run_episode(agent, env, render and episode < 5, save_trajectories)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_distances.append(episode_distance)
            if gap is not None:
                gap_to_optimal.append(gap)
            valid_solutions.append(is_valid)
            
            if save_trajectories:
                trajectories.append(trajectory)
        
        # Compute metrics
        metrics = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_distance': np.mean(episode_distances),
            'std_distance': np.std(episode_distances),
            'success_rate': np.mean(valid_solutions),
            'episode_rewards': episode_rewards,
            'episode_distances': episode_distances,
            'valid_solutions': valid_solutions
        }
        
        if gap_to_optimal:
            metrics.update({
                'mean_gap_to_optimal': np.mean(gap_to_optimal),
                'std_gap_to_optimal': np.std(gap_to_optimal),
                'gap_to_optimal': gap_to_optimal
            })
        
        if save_trajectories:
            metrics['trajectories'] = trajectories
        
        agent.q_network.train() if hasattr(agent, 'q_network') else None
        
        return metrics
    
    def _run_episode(self, 
                    agent: BaseAgent,
                    env: TSPEnvironment,
                    render: bool = False,
                    save_trajectory: bool = False) -> Tuple[float, int, float, Optional[float], bool, Optional[List]]:
        """
        Run a single evaluation episode.
        
        Returns:
            Tuple of (total_reward, episode_length, total_distance, gap_to_optimal, is_valid, trajectory)
        """
        state = env.reset()
        agent.reset_episode()
        
        total_reward = 0.0
        episode_length = 0
        trajectory = [] if save_trajectory else None
        
        while True:
            # Select action (no training)
            action = agent.select_action(state, training=False)
            
            if save_trajectory:
                trajectory.append({
                    'state': state.copy(),
                    'action': action
                })
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            if save_trajectory:
                trajectory[-1].update({
                    'reward': reward,
                    'next_state': next_state.copy(),
                    'done': done,
                    'info': info.copy()
                })
            
            if render:
                env.render()
                time.sleep(0.1)
            
            if done:
                break
            
            state = next_state
        
        # Get final metrics
        total_distance = info.get('total_distance', 0.0)
        gap_to_optimal = info.get('gap_to_optimal', None)
        is_valid = info.get('is_valid_path', False)
        
        return total_reward, episode_length, total_distance, gap_to_optimal, is_valid, trajectory
    
    def compare_agents(self, 
                      agents: Dict[str, BaseAgent],
                      env: TSPEnvironment,
                      num_episodes: int = 100) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents on the same environment.
        
        Args:
            agents: Dictionary mapping agent names to agent objects
            env: Environment to evaluate on
            num_episodes: Number of episodes per agent
            
        Returns:
            Dictionary mapping agent names to their evaluation metrics
        """
        results = {}
        
        for agent_name, agent in agents.items():
            print(f"Evaluating agent: {agent_name}")
            metrics = self.evaluate_agent(agent, env, num_episodes)
            results[agent_name] = metrics
        
        return results
    
    def evaluate_on_dataset(self,
                           agent: BaseAgent,
                           dataset: List[Dict[str, Any]],
                           state_components: List[str],
                           num_episodes_per_instance: int = 10) -> Dict[str, Any]:
        """
        Evaluate agent on a dataset of TSP instances.
        
        Args:
            agent: Agent to evaluate
            dataset: List of TSP instance data
            state_components: State components to use
            num_episodes_per_instance: Number of episodes per instance
            
        Returns:
            Aggregated evaluation metrics
        """
        all_metrics = []
        instance_metrics = {}
        
        for i, instance_data in enumerate(dataset):
            # Create environment from instance
            coordinates = np.array(instance_data['coordinates'])
            env = TSPEnvironment(
                n_cities=len(coordinates),
                coordinates=coordinates,
                use_tsplib='problem_name' in instance_data,
                tsplib_name=instance_data.get('problem_name'),
                state_components=state_components
            )
            
            # Override with cached optimal solution
            env.optimal_path = instance_data['optimal_path']
            env.optimal_distance = instance_data['optimal_distance']
            
            # Evaluate on this instance
            metrics = self.evaluate_agent(agent, env, num_episodes_per_instance)
            all_metrics.append(metrics)
            instance_metrics[f"instance_{i}"] = metrics
        
        # Aggregate metrics across all instances
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics['instance_metrics'] = instance_metrics
        
        return aggregated_metrics
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple evaluations."""
        if not metrics_list:
            return {}
        
        # Collect all episode-level data
        all_rewards = []
        all_distances = []
        all_gaps = []
        all_success = []
        
        for metrics in metrics_list:
            all_rewards.extend(metrics.get('episode_rewards', []))
            all_distances.extend(metrics.get('episode_distances', []))
            if 'gap_to_optimal' in metrics:
                all_gaps.extend(metrics['gap_to_optimal'])
            all_success.extend(metrics.get('valid_solutions', []))
        
        # Compute aggregated statistics
        aggregated = {
            'total_episodes': len(all_rewards),
            'total_instances': len(metrics_list),
            'mean_reward': np.mean(all_rewards) if all_rewards else 0.0,
            'std_reward': np.std(all_rewards) if all_rewards else 0.0,
            'mean_distance': np.mean(all_distances) if all_distances else 0.0,
            'std_distance': np.std(all_distances) if all_distances else 0.0,
            'overall_success_rate': np.mean(all_success) if all_success else 0.0
        }
        
        if all_gaps:
            aggregated.update({
                'mean_gap_to_optimal': np.mean(all_gaps),
                'std_gap_to_optimal': np.std(all_gaps),
                'median_gap_to_optimal': np.median(all_gaps)
            })
        
        # Instance-level statistics
        instance_means = [m.get('mean_reward', 0.0) for m in metrics_list]
        instance_distances = [m.get('mean_distance', 0.0) for m in metrics_list]
        instance_success = [m.get('success_rate', 0.0) for m in metrics_list]
        
        aggregated.update({
            'mean_reward_across_instances': np.mean(instance_means),
            'std_reward_across_instances': np.std(instance_means),
            'mean_distance_across_instances': np.mean(instance_distances),
            'std_distance_across_instances': np.std(instance_distances),
            'mean_success_rate_across_instances': np.mean(instance_success),
            'std_success_rate_across_instances': np.std(instance_success)
        })
        
        return aggregated
    
    def zero_shot_evaluation(self,
                           agent: BaseAgent,
                           test_instances: List[Dict[str, Any]],
                           state_components: List[str]) -> Dict[str, Any]:
        """
        Perform zero-shot evaluation on test instances.
        
        Args:
            agent: Trained agent
            test_instances: Test instances
            state_components: State components to use
            
        Returns:
            Zero-shot evaluation metrics
        """
        print("Performing zero-shot evaluation...")
        
        # Evaluate on each test instance with a fresh network initialization
        results = []
        
        for i, instance_data in enumerate(test_instances):
            print(f"Zero-shot evaluation on test instance {i+1}/{len(test_instances)}")
            
            # Create environment
            coordinates = np.array(instance_data['coordinates'])
            env = TSPEnvironment(
                n_cities=len(coordinates),
                coordinates=coordinates,
                use_tsplib='problem_name' in instance_data,
                tsplib_name=instance_data.get('problem_name'),
                state_components=state_components
            )
            
            # Override with cached optimal solution
            env.optimal_path = instance_data['optimal_path']
            env.optimal_distance = instance_data['optimal_distance']
            
            # Run single episode with untrained/random policy
            state = env.reset()
            agent.reset_episode()
            
            total_reward = 0.0
            episode_length = 0
            
            while True:
                # Use random action selection for zero-shot
                valid_actions = env.get_valid_actions()
                action = np.random.choice(valid_actions)
                
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
                state = next_state
            
            # Store results
            result = {
                'instance_id': i,
                'total_reward': total_reward,
                'episode_length': episode_length,
                'total_distance': info.get('total_distance', 0.0),
                'gap_to_optimal': info.get('gap_to_optimal', None),
                'is_valid': info.get('is_valid_path', False)
            }
            results.append(result)
        
        # Aggregate results
        total_rewards = [r['total_reward'] for r in results]
        total_distances = [r['total_distance'] for r in results]
        gaps = [r['gap_to_optimal'] for r in results if r['gap_to_optimal'] is not None]
        valid_solutions = [r['is_valid'] for r in results]
        
        metrics = {
            'num_instances': len(results),
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_distance': np.mean(total_distances),
            'std_distance': np.std(total_distances),
            'success_rate': np.mean(valid_solutions),
            'detailed_results': results
        }
        
        if gaps:
            metrics.update({
                'mean_gap_to_optimal': np.mean(gaps),
                'std_gap_to_optimal': np.std(gaps)
            })
        
        return metrics