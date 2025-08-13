import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dissertation.env.tsp_env import TSPEnvironment
from confs.path import project_root
from confs.config_loader import config_loader


class DatasetLoader:
    """Load and manage TSP datasets."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing generated datasets
        """
        self.data_dir = data_dir or os.path.join(project_root, "data", "generated")
        
    def load_random_dataset(self, city_num: int, num_instances: int = None, split: str = 'both') -> Dict[str, List[Dict[str, Any]]]:
        """
        Load random TSP dataset.
        
        Args:
            city_num: Number of cities
            num_instances: Number of instances (if None, uses config value)
            split: Which split to load ('train', 'test', or 'both')
            
        Returns:
            Dictionary with train/test splits or specific split data
        """
        # Load config and use total_instances if num_instances not provided
        if num_instances is None:
            config = config_loader.load_experiment_config()
            num_instances = config['dataset']['total_instances']
        train_file = os.path.join(self.data_dir, f"random_tsp_{city_num}cities_{num_instances}instances_train.json")
        test_file = os.path.join(self.data_dir, f"random_tsp_{city_num}cities_{num_instances}instances_test.json")
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Split dataset files not found for {city_num} cities. Please generate them first.")
        
        if split == 'train':
            with open(train_file, 'r') as f:
                return json.load(f)
        elif split == 'test':
            with open(test_file, 'r') as f:
                return json.load(f)
        else:  # split == 'both'
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            return {'train': train_data, 'test': test_data}
    
    def load_tsplib_dataset(self, problem_name: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Load TSPLIB dataset.
        
        Args:
            problem_name: Specific problem name, or None to load all
            
        Returns:
            Dictionary mapping problem name to instance data
        """
        cache_file = os.path.join(self.data_dir, "tsplib_datasets.json")
        
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"TSPLIB datasets not found: {cache_file}. Please generate them first.")
        
        with open(cache_file, 'r') as f:
            datasets = json.load(f)
        
        if problem_name:
            if problem_name not in datasets:
                raise KeyError(f"Problem {problem_name} not found in datasets")
            return {problem_name: datasets[problem_name]}
        
        return datasets
    
    def create_env_from_instance(self, 
                               instance_data: Dict[str, Any], 
                               state_components: List[str] = None) -> TSPEnvironment:
        """
        Create TSP environment from instance data.
        
        Args:
            instance_data: Instance data dictionary
            state_components: List of state components to include
            
        Returns:
            TSP environment initialized with instance data
        """
        coordinates = np.array(instance_data['coordinates'])
        
        # Determine if this is TSPLIB data
        is_tsplib = 'problem_name' in instance_data
        
        if is_tsplib:
            # For TSPLIB data, we need to handle it specially
            tsplib_data_dir = os.path.join(project_root, "data", "tsplib")
            env = TSPEnvironment(
                n_cities=len(coordinates),
                coordinates=coordinates,
                use_tsplib=True,
                tsplib_name=instance_data['problem_name'],
                tsplib_path=tsplib_data_dir,
                state_components=state_components
            )
            # Override with cached optimal solution
            env.optimal_path = instance_data['optimal_path']
            env.optimal_distance = instance_data['optimal_distance']
        else:
            # For random data
            env = TSPEnvironment(
                n_cities=len(coordinates),
                coordinates=coordinates,
                use_tsplib=False,
                state_components=state_components
            )
            # Override with cached optimal solution
            env.optimal_path = instance_data['optimal_path']
            env.optimal_distance = instance_data['optimal_distance']
        
        return env
    
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Get list of available datasets.
        
        Returns:
            Dictionary with 'random' and 'tsplib' keys listing available datasets
        """
        available = {'random': [], 'tsplib': []}
        
        if not os.path.exists(self.data_dir):
            return available
        
        # Check for random datasets
        for filename in os.listdir(self.data_dir):
            if filename.startswith('random_tsp_') and filename.endswith('.json'):
                available['random'].append(filename)
        
        # Check for TSPLIB datasets
        tsplib_file = os.path.join(self.data_dir, "tsplib_datasets.json")
        if os.path.exists(tsplib_file):
            with open(tsplib_file, 'r') as f:
                tsplib_data = json.load(f)
            available['tsplib'] = list(tsplib_data.keys())
        
        return available