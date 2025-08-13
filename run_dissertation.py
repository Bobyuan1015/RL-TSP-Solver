#!/usr/bin/env python3


import sys
import os
import time
import traceback
import random
import threading
import numpy as np
import torch
from typing import Dict, Any, List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager

import yaml
from itertools import combinations
# Add project root to Python path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confs.config_loader import config_loader
from dissertation.env import TSPEnvironment
from dissertation.agents import DQNAgent, DQNLSTMAgent, ReinforceAgent, ActorCriticAgent, PPOAgent
from dissertation.utils import Logger, Evaluator, RewardCalculator, DatasetGenerator, DatasetLoader, ModelManager

def build_state_combinations():
    """构建消融实验的状态组合映射"""
    base_states = ["current_city_onehot", "visited_mask", "order_embedding", "distances_from_current"]

    map_state_types = {}

    # 1. 全状态（基线）
    map_state_types['full'] = base_states.copy()

    # 2. 依次移除一种状态（4种组合）
    for i, state_to_remove in enumerate(base_states):
        remaining_states = [s for s in base_states if s != state_to_remove]
        map_state_types[f'ablation_remove_{state_to_remove.split("_")[0]}'] = remaining_states

    # 3. 依次移除两种状态（6种组合）
    for i, (state1, state2) in enumerate(combinations(base_states, 2)):
        remaining_states = [s for s in base_states if s not in [state1, state2]]
        key_name = f'ablation_remove_{state1.split("_")[0]}_{state2.split("_")[0]}'
        map_state_types[key_name] = remaining_states

    return map_state_types

class ExperimentRunner:
    """Main experiment runner for TSP RL experiments."""
    
    def __init__(self, config_file: str = "experiment_config.yaml", algorithm: str = None, mode: str = None):
        """
        Initialize experiment runner.
        
        Args:
            config_file: Configuration file name
            algorithm: Specific algorithm to run (for parallel execution)
            mode: Specific mode to run (for parallel execution)
        """
        # Load configurations
        self.experiment_config = config_loader.load_experiment_config(config_file)
        self.model_config = config_loader.load_model_config()
        
        # Set global seed
        self.seed = self.experiment_config['experiment']['seed']
        self._set_global_seed(self.seed)

        #不使用配置
        self.experiment_config['state_types'] = build_state_combinations()

        
        # Store algorithm and mode for this instance
        self.algorithm = algorithm
        self.mode = mode
        
        # Create experiment name with algorithm and mode if specified
        base_name = self.experiment_config['experiment']['name']
        if algorithm and mode:
            experiment_name = f"{base_name}_{algorithm}_{mode}"
        else:
            experiment_name = base_name
        
        # Initialize components
        self.logger = Logger(
            experiment_name=experiment_name,
            algorithm=algorithm,
            mode=mode,
            log_level=self.experiment_config['logging']['level'],
            save_to_file=self.experiment_config['logging']['save_to_file'],
            console_output=self.experiment_config['logging']['console_output']
        )
        
        self.evaluator = Evaluator()
        self.reward_calculator = RewardCalculator()
        
        # Dataset components
        self.dataset_generator = DatasetGenerator(seed=self.seed)
        self.dataset_loader = DatasetLoader()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.logger.info(f"Using device: {self.device}")
        
        # Algorithm mapping
        self.algorithm_classes = {
            'DQN': DQNAgent,
            'DQN_LSTM': DQNLSTMAgent,
            'Reinforce': ReinforceAgent,
            'ActorCritic': ActorCriticAgent,
            'PPO': PPOAgent
        }
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0

        
    def _set_global_seed(self, seed: int):
        """Set global random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def prepare_datasets(self):
        """Generate and prepare datasets for experiments."""
        self.logger.logger.info("Preparing datasets...")
        
        # Generate random datasets with train/test split
        city_nums = self.experiment_config['dataset']['city_nums']
        total_instances = self.experiment_config['dataset']['total_instances']
        train_ratio = self.experiment_config['dataset']['train_ratio']
        
        random_datasets = self.dataset_generator.generate_random_datasets(
            city_nums=city_nums,
            num_instances=total_instances,
            train_ratio=train_ratio,
            force_regenerate=False
        )
        
        # Validate datasets
        for city_num, split_data in random_datasets.items():
            # Validate both train and test splits
            all_instances = split_data['train'] + split_data['test']
            if not self.dataset_generator.validate_dataset(all_instances):
                raise ValueError(f"Dataset validation failed for {city_num} cities")
        
        # Generate TSPLIB datasets if specified
        if self.experiment_config['dataset']['use_tsplib']:
            tsplib_problems = self.experiment_config['dataset']['tsplib_problems']
            tsplib_datasets = self.dataset_generator.generate_tsplib_datasets(
                tsplib_problems=tsplib_problems,
                force_regenerate=False
            )
        
        self.logger.logger.info("Dataset preparation completed")
    
    def create_agent(self, algorithm: str, state_dim: int, action_dim: int, 
                    state_components: List[str]) -> Any:
        """Create agent instance."""
        if algorithm not in self.algorithm_classes:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        agent_config = config_loader.get_algorithm_config(algorithm)
        agent_class = self.algorithm_classes[algorithm]
        
        return agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            config=agent_config,
            state_components=state_components,
            device=str(self.device)
        )
    
    def calculate_state_dim(self, state_components: List[str], n_cities: int) -> int:
        """Calculate state dimension based on components."""
        dim = 0
        
        for component in state_components:
            if component in ['current_city_onehot', 'visited_mask', 'order_embedding', 'distances_from_current']:
                dim += n_cities
            elif component == 'step_count':
                dim += 1
        
        return dim
    
    def _calculate_total_tasks(self):
        """Calculate total number of tasks for progress tracking."""
        algorithms = self.experiment_config['algorithms']
        city_nums = self.experiment_config['dataset']['city_nums']
        state_types = list(self.experiment_config['state_types'].keys())
        repeat_runs = self.experiment_config['training']['repeat_runs']
        modes = self.experiment_config['modes']
        
        total_tasks = 0
        
        for mode in modes:
            if mode == "per_instance":
                # Load dataset to get number of training instances
                for algorithm in algorithms:
                    for city_num in city_nums:
                        dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                        train_instances = len(dataset['train'])
                        test_instances = len(dataset['test'])

                        for state_type in state_types:
                            total_tasks += train_instances * repeat_runs*(1+test_instances)


            
            elif mode == "cross_instance":
                for algorithm in algorithms:
                    for city_num in city_nums:
                        for state_type in state_types:
                            # Cross-instance training: repeat_runs tasks
                            # Testing on test instances
                            dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                            total_tasks += repeat_runs + len(dataset['test'])*repeat_runs

        
        self.total_tasks = total_tasks
        self.logger.logger.info(f"Total tasks calculated: {total_tasks}")
    
    def run_mode(self, mode: str):
        """Run experiments in specified mode (per_instance or cross_instance)."""
        self.logger.logger.info(f"Starting {mode} mode experiments")
        
        algorithms = self.experiment_config['algorithms']
        city_nums = self.experiment_config['dataset']['city_nums']
        state_types = list(self.experiment_config['state_types'].keys())
        repeat_runs = self.experiment_config['training']['repeat_runs']
        save_model_every = self.experiment_config['training'].get('save_model_every', 100)
        evaluate_every = self.experiment_config['training'].get('evaluate_every', 100)
        
        if mode == "per_instance":
            episodes_training = self.experiment_config['training']['episodes_per_instance']
        else:  # cross_instance
            episodes_training = self.experiment_config['training']['episodes_cross_instance']
        
        for algorithm in algorithms:
            self.logger.logger.info(f"Running algorithm: {algorithm}")
            
            for city_num in city_nums:
                self.logger.logger.info(f"Processing {city_num} cities")
                
                # Load pre-split dataset
                dataset = self.dataset_loader.load_random_dataset(city_num, split='both')
                train_data = dataset['train']

                test_data = dataset['test']
                
                for state_type in state_types:
                    self.logger.logger.info(f"State type: {state_type}")
                    
                    state_components = config_loader.get_state_type_config(state_type)
                    state_dim = self.calculate_state_dim(state_components, city_num)
                    
                    # Use unified training function for both modes
                    self._run_unified_training(
                        algorithm, city_num, state_type, state_components, state_dim,
                        train_data, test_data, repeat_runs, episodes_training,
                        save_model_every, evaluate_every, mode
                    )

    def _run_unified_training(self, algorithm: str, city_num: int, state_type: str,
                            state_components: List[str], state_dim: int, train_data: List,
                            test_data: List, repeat_runs: int, episodes_training: int,
                            save_model_every: int, evaluate_every: int, mode: str):
        """Handle unified training logic for both per_instance and cross_instance modes."""


        for index_instance_id, instance_data in enumerate(train_data):

            for run_id in range(repeat_runs):
                self.completed_tasks += 1
                agent = self.create_agent(algorithm, state_dim, city_num, state_components)
                # Calculate local progress
                if mode == "per_instance":
                    instance_id = instance_data.get('instance_id', '无')

                    total_instance_runs = len(train_data) * repeat_runs
                    current_instance_run = index_instance_id * repeat_runs + run_id + 1
                    local_progress = (current_instance_run / total_instance_runs) * 100
                    # Create environment and agent
                    env = self.dataset_loader.create_env_from_instance(instance_data, state_components)
                else:  # cross_instance
                    if index_instance_id >0:
                        print("CROSS 此轮结束")
                        return
                    instance_id = str([d['instance_id'] for d in train_data])
                    current_instance_run = run_id + 1
                    local_progress = (current_instance_run / repeat_runs) * 100
                    env = train_data

                total_instances_for_log = len(train_data)


                self.logger.log_experiment_progress(
                    algorithm, city_num, mode,
                    instance_id, total_instances_for_log, run_id + 1, repeat_runs, index_instance_id,
                    self.completed_tasks, self.total_tasks
                )
                self.logger.logger.info(f"Local Progress: {run_id + 1 if mode == 'cross_instance' else current_instance_run}/{repeat_runs if mode == 'cross_instance' else total_instance_runs} ({local_progress:.1f}%)")

                # Create model manager
                model_dir = os.path.join(
                    self.logger.get_experiment_dir(), "models",
                    algorithm, f"{city_num}cities", state_type, mode,
                    f"instance_{instance_id}_run_{run_id}"
                )
                model_manager = ModelManager(model_dir, keep_recent=10)

                # Train agent
                self._train_agent(
                    agent, env, episodes_training, algorithm,
                    city_num, mode, instance_id, run_id,
                    state_type, "train", model_manager,
                    save_model_every, evaluate_every, state_components
                )
                
                # Zero-shot testing on test instances
                for index_test_instance_id in range(len(test_data)):
                    test_instance_data = test_data[index_test_instance_id]
                    test_instance_id = test_instance_data.get('instance_id', '无')
                    # print(f'{algorithm} {mode} {state_type}  {city_num} index_instance_id={index_instance_id} run_id={run_id} index_test_instance_id={index_test_instance_id}     instance_id={instance_id}   test_instance_id={test_instance_id}')

                    self.completed_tasks += 1

                    test_progress = ((index_test_instance_id + 1) / len(test_data)) * 100
                    self.logger.logger.info(f"Testing instance {index_test_instance_id + 1}/{len(test_data)} ({test_progress:.1f}%)")

                    # Log testing progress
                    self.logger.log_experiment_progress(
                        algorithm, city_num, f"{mode}_testing",
                        test_instance_id, len(test_data), 1, 1, index_test_instance_id,
                        self.completed_tasks, self.total_tasks
                    )

                    # Create fresh environment for zero-shot test
                    test_env = self.dataset_loader.create_env_from_instance(test_instance_data, state_components)

                    # Zero-shot test (single episode with trained agent using best model)
                    self._zero_shot_test(
                        agent, test_env, algorithm, city_num, mode,
                        f"train_{instance_id}_test_{test_instance_id}", run_id, state_type, "test", model_manager
                    )


    def run_per_instance_mode(self):
        """Run per-instance training mode."""
        return self.run_mode("per_instance")

    def run_cross_instance_mode(self):
        """Run cross-instance training mode."""
        return self.run_mode("cross_instance")

    def _train_agent(self, agent, env_or_data, num_episodes: int, algorithm: str,
                    city_num: int, mode: str, instance_id: int, run_id: int, 
                    state_type: str, train_test: str, model_manager: ModelManager = None,
                    save_model_every: int = 100, evaluate_every: int = 100,
                    state_components: List[str] = None):
        """Train agent on environment or across multiple instances."""
        for episode in range(num_episodes):
            # Handle cross-instance vs single instance training
            if mode == "cross_instance" and isinstance(env_or_data, list):
                # Randomly sample instance for cross-instance training
                instance_data = random.choice(env_or_data)
                env = self.dataset_loader.create_env_from_instance(
                    instance_data, state_components
                )

            else:
                # Single instance training
                env = env_or_data
            current_instance_id = instance_id
            
            state = env.reset()
            agent.reset_episode()
            
            total_reward = 0.0
            step = 0

            while True:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                }
                
                update_info = agent.update(experience)
                
                # Extract coordinates and state values for CSV logging
                coordinates = env.coordinates.tolist() if hasattr(env, 'coordinates') else None
                state_values = state_components
                
                # Log step to CSV for both modes
                # 当前state，根据state选择a，对应reward
                self.logger.log_training_step(
                    algorithm, city_num, mode, current_instance_id, run_id, state_type,
                    train_test, episode, step, action, state, done, reward,
                    update_info.get('loss', None), total_reward + reward, 
                    env.total_distance, env.optimal_distance, env.optimal_path,
                    coordinates, state_values
                )

                total_reward += reward
                step += 1
                
                if done:

                    # if episode <10:
                    #     print(f'yf episode={episode}',state['path_sequence'],env.total_distance)
                    break
                
                state = next_state
            
            # Log episode summary periodically
            if episode % evaluate_every == 0:
                self.logger.log_episode_summary(
                    algorithm, city_num, mode, current_instance_id, run_id, state_type,
                    train_test, episode, total_reward, step, update_info
                )
            
            # Save model periodically
            if episode % save_model_every == 0 and model_manager is not None:
                model_manager.save_model(
                    agent, episode, total_reward, 
                    metric_name="reward", is_higher_better=True
                )
    
    def _extract_state_values(self, state: Dict[str, Any], state_components: List[str]) -> Dict[str, Any]:
        """
        Extract state values based on state_type configuration.
        
        Args:
            state: Current state dictionary
            state_components: List of state components to extract
            
        Returns:
            Dictionary with state component values
        """
        state_values = {}
        
        for component in state_components:
            if component in state:
                value = state[component]
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(value, 'tolist'):
                    state_values[component] = value.tolist()
                elif isinstance(value, (int, float, bool)):
                    state_values[component] = value
                else:
                    # For other types, convert to string
                    state_values[component] = str(value)
            else:
                # Component not found in state
                state_values[component] = None
                
        return state_values

    def _zero_shot_test(self, agent, env, algorithm: str, city_num: int, 
                       mode: str, instance_id: int, run_id: int, state_type: str, 
                       train_test: str, model_manager=None):
        """Perform test with agent (trained or untrained) and log detailed steps to CSV."""
        # Load the best model if model_manager is provided and has a best model
        if model_manager is not None and model_manager.get_best_model_path() is not None:
            self.logger.logger.info(f"Loading best model for zero-shot testing: {model_manager.get_best_model_path()}")
            model_manager.load_best_model(agent)
        
        state = env.reset()
        agent.reset_episode()
        
        total_reward = 0.0
        step = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            # Extract coordinates and state values for CSV logging
            coordinates = env.coordinates.tolist() if hasattr(env, 'coordinates') else None
            # For _zero_shot_test, we don't have state_components passed, so we need to get them
            state_components = config_loader.get_state_type_config(state_type)


            self.logger.log_training_step(
                algorithm, city_num, mode, instance_id, run_id, state_type,
                train_test, 0, step, action, state, done, reward,
                None, total_reward + reward, env.total_distance, env.optimal_distance, env.optimal_path,
                coordinates, state_components
            )

            total_reward += reward
            step += 1
            # Log step

            
            if done:
                break
            
            state = next_state
        
        # Log episode summary
        self.logger.log_episode_summary(
            algorithm, city_num, mode, instance_id, run_id, state_type,
            train_test, 0, total_reward, step
        )




def run_single_experiment(algorithm: str, mode: str, config_file: str = "experiment_config.yaml", progress_dict=None):
    """Run a single experiment for a specific algorithm and mode."""
    try:
        print(f"[{algorithm}-{mode}] Starting experiment...")
        
        # Initialize progress tracking
        if progress_dict is not None:
            task_key = f"{algorithm}-{mode}"
            progress_dict[task_key] = {"current": 0, "total": 0, "status": "starting"}

        # Create experiment runner for this specific algorithm/mode combination
        runner = ExperimentRunner(config_file=config_file, algorithm=algorithm, mode=mode)
        if progress_dict is not None:
            runner.progress_dict = progress_dict
            runner.task_key = task_key
            # Set up progress callback in logger
            runner.logger.progress_dict = progress_dict
            runner.logger.task_key = task_key
        
        # Filter the runner to only run specific algorithm and mode
        original_algorithms = runner.experiment_config['algorithms']
        original_modes = runner.experiment_config['modes']
        
        # Override config for this specific run
        runner.experiment_config['algorithms'] = [algorithm]
        runner.experiment_config['modes'] = [mode]
        
        print(f"[{algorithm}-{mode}] Preparing datasets...")
        # Prepare datasets
        runner.prepare_datasets()
        
        print(f"[{algorithm}-{mode}] Calculating total tasks...")
        # Calculate total tasks for this specific experiment
        runner._calculate_total_tasks()
        
        # Update progress with total tasks
        if progress_dict is not None:
            progress_dict[task_key]["total"] = runner.total_tasks
            progress_dict[task_key]["status"] = "running"
        
        print(f"[{algorithm}-{mode}] Starting {mode} mode with {runner.total_tasks} total tasks...")
        # Run the specific mode
        if mode == "per_instance":
            runner.run_per_instance_mode()
        elif mode == "cross_instance":
            runner.run_cross_instance_mode()
        
        print(f"[{algorithm}-{mode}] Analyzing results...")
        # Analyze results for this specific experiment
        analysis = runner.logger.analyze_results()
        runner.logger.save_experiment_summary(analysis)
        
        # Mark as completed
        if progress_dict is not None:
            progress_dict[task_key]["status"] = "completed"
            progress_dict[task_key]["current"] = progress_dict[task_key]["total"]
        
        print(f"[{algorithm}-{mode}] Experiment completed successfully!")
        return f"Completed: {algorithm} - {mode}"
        
    except Exception as e:
        # Mark as failed
        if progress_dict is not None:
            progress_dict[task_key]["status"] = "failed"
        
        error_msg = f"Failed: {algorithm} - {mode}: {str(e)}"
        print(f"[{algorithm}-{mode}] ERROR: {str(e)} {traceback.format_exc()}")
        return error_msg


def print_progress_table(progress_dict):
    """Print a formatted progress table for all running experiments."""
    if not progress_dict:
        return
    
    print("\n" + "="*120)
    print(f"{'Task':<15} {'Status':<10} {'Overall':<12} {'Details':<75}")
    print("-"*120)
    
    for task_id, info in progress_dict.items():
        status = info.get('status', 'Unknown')
        current = info.get('current', 0)
        total = info.get('total', 0)
        details_info = info.get('details', {})
        
        # Overall progress
        if total > 0:
            percentage = (current / total) * 100
            overall_str = f"{current}/{total} ({percentage:.1f}%)"
        else:
            overall_str = "N/A"
        
        # Detailed progress from log_experiment_progress
        if details_info:
            city_num = details_info.get('city_num', '?')
            mode = details_info.get('mode', '?')
            curr_inst = details_info.get('current_instance', 0)
            total_inst = details_info.get('total_instances', 0)
            curr_run = details_info.get('current_run', 0)
            total_run = details_info.get('total_runs', 0)
            inst_pct = details_info.get('instance_pct', 0)
            run_pct = details_info.get('run_pct', 0)
            total_pct = details_info.get('total_pct', 0)
            
            details_str = (f"[{city_num}cities] {mode} | "
                          f"Inst {curr_inst}/{total_inst} ({inst_pct:.1f}%) | "
                          f"Run {curr_run}/{total_run} ({run_pct:.1f}%) | "
                          f"Total {total_pct:.1f}%")
        else:
            details_str = "No details available"
        
        print(f"{task_id:<15} {status:<10} {overall_str:<12} {details_str[:75]:<75}")
    
    print("="*120 + "\n")


def monitor_progress(progress_dict, stop_event):
    """Background thread to monitor and display progress."""
    while not stop_event.is_set():
        time.sleep(20)  # Update every 10 seconds
        if progress_dict:
            print_progress_table(progress_dict)


def main():
    """Main function to run experiments with parallel execution by algorithm and mode."""
    
    # 进程模式控制开关: True=单进程, False=多进程
    USE_SINGLE_PROCESS = False
    
    # Load configuration from YAML file (for backward compatibility)
    config_path = "confs/experiment_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    
    try:
        # Get algorithms and modes from config
        algorithms = config.get('algorithms', ['DQN'])
        modes = config.get('modes', ['per_instance'])
        
        # Create list of all algorithm-mode combinations
        experiment_combinations = []
        for algorithm in algorithms:
            for mode in modes:
                experiment_combinations.append((algorithm, mode))
        
        print(f"\n" + "="*60)
        print("PARALLEL EXPERIMENT EXECUTION")
        print("="*60)
        print(f"Total experiment combinations: {len(experiment_combinations)}")
        print(f"Algorithm-Mode combinations: {experiment_combinations}")
        
        # Determine number of parallel workers based on mode
        if USE_SINGLE_PROCESS:
            max_workers = 1
            print(f"Using single-process mode (max_workers = 1)")
        else:
            max_workers = min(len(experiment_combinations), cpu_count())
            print(f"Using multi-process mode (max_workers = {max_workers})")
        
        print(f"Available CPU cores: {cpu_count()}")
        print(f"Total experiment combinations: {len(experiment_combinations)}")
        print("\nStarting execution...\n")
        
        # Create shared progress dictionary for monitoring
        manager = Manager()
        progress_dict = manager.dict()
        
        # Start progress monitoring thread
        stop_event = threading.Event()
        progress_thread = threading.Thread(target=monitor_progress, args=(progress_dict, stop_event))
        progress_thread.daemon = True
        progress_thread.start()
        
        # Run experiments in parallel
        completed_experiments = []
        failed_experiments = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments with startup notification
                future_to_experiment = {}
                for algorithm, mode in experiment_combinations:
                    print(f"[SUBMIT] Queuing experiment: {algorithm} - {mode}")
                    future = executor.submit(run_single_experiment, algorithm, mode, "experiment_config.yaml", progress_dict)
                    future_to_experiment[future] = (algorithm, mode)
                
                print(f"\n[STATUS] All {len(experiment_combinations)} experiments queued and starting...\n")
                
                # Process completed experiments with progress tracking
                completed_count = 0
                for future in as_completed(future_to_experiment):
                    algorithm, mode = future_to_experiment[future]
                    try:
                        result = future.result()
                        completed_count += 1
                        
                        # Remove completed task from progress dict
                        task_id = f"{algorithm}-{mode}"
                        if task_id in progress_dict:
                            del progress_dict[task_id]
                        
                        progress_pct = (completed_count / len(experiment_combinations)) * 100
                        status_msg = f"[PROGRESS] {completed_count}/{len(experiment_combinations)} ({progress_pct:.1f}%) - {result}"
                        print(status_msg)
                        
                        if "Completed" in result:
                            completed_experiments.append((algorithm, mode))
                        else:
                            failed_experiments.append((algorithm, mode, result))
                    except Exception as e:
                        print(f"主进程异常 {traceback.format_exc()}")
                        completed_count += 1
                        
                        # Remove failed task from progress dict
                        task_id = f"{algorithm}-{mode}"
                        if task_id in progress_dict:
                            del progress_dict[task_id]
                        
                        progress_pct = (completed_count / len(experiment_combinations)) * 100
                        error_msg = f"Exception in {algorithm} - {mode}: {str(e)}"
                        status_msg = f"[PROGRESS] {completed_count}/{len(experiment_combinations)} ({progress_pct:.1f}%) - {error_msg}"
                        print(status_msg)
                        failed_experiments.append((algorithm, mode, error_msg))
        
        finally:
            # Stop progress monitoring
            stop_event.set()
            progress_thread.join(timeout=1)
        
        # Print final progress table
        print_progress_table(progress_dict)
        
        # Report final results
        print("\n" + "="*60)
        print("PARALLEL EXECUTION SUMMARY")
        print("="*60)
        print(f"Total experiments: {len(experiment_combinations)}")
        print(f"Completed successfully: {len(completed_experiments)}")
        print(f"Failed: {len(failed_experiments)}")
        
        if completed_experiments:
            print("\nCompleted experiments:")
            for algo, mode in completed_experiments:
                print(f"  ✓ {algo} - {mode}")
        
        if failed_experiments:
            print("\nFailed experiments:")
            for algo, mode, error in failed_experiments:
                print(f"  ✗ {algo} - {mode}: {error}")
        
        if failed_experiments:
            print("\nSome experiments failed. Check logs for details.")
            sys.exit(1)
        else:
            print("\nAll experiments completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Experiment failed with error: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()