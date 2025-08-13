import os
import numpy as np
from typing import List, Tuple, Dict, Any
import json
from tqdm import tqdm

from dissertation.env.tsp_env import TSPEnvironment
from confs.path import project_root


class DatasetGenerator:
    """Generate and cache TSP datasets."""
    
    def __init__(self, data_dir: str = None, seed: int = 42):
        """
        Initialize dataset generator.
        
        Args:
            data_dir: Directory to save generated datasets
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir or os.path.join(project_root, "data", "generated")
        self.seed = seed
        os.makedirs(self.data_dir, exist_ok=True)
        
    def generate_random_datasets(self, 
                                city_nums: List[int],
                                num_instances: int = 100,
                                train_ratio: float = 0.8,
                                force_regenerate: bool = False) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """
        Generate random TSP datasets for different city numbers.
        
        Args:
            city_nums: List of city numbers to generate datasets for
            num_instances: Number of instances per city number
            train_ratio: Ratio of instances for training (0.8 = 80% train, 20% test)
            force_regenerate: Whether to regenerate even if cached data exists
            
        Returns:
            Dictionary mapping city_num to {'train': [instances], 'test': [instances]}
        """
        datasets = {}
        
        for city_num in city_nums:
            # Check for pre-split dataset files
            train_file = os.path.join(self.data_dir, f"random_tsp_{city_num}cities_{num_instances}instances_train.json")
            test_file = os.path.join(self.data_dir, f"random_tsp_{city_num}cities_{num_instances}instances_test.json")
            
            if os.path.exists(train_file) and os.path.exists(test_file) and not force_regenerate:
                print(f"Loading cached split dataset for {city_num} cities")
                with open(train_file, 'r') as f:
                    train_instances = json.load(f)
                with open(test_file, 'r') as f:
                    test_instances = json.load(f)
                datasets[city_num] = {'train': train_instances, 'test': test_instances}
            else:
                print(f"Generating new dataset for {city_num} cities...")
                instances = []
                
                for instance_id in tqdm(range(num_instances), desc=f"Generating {city_num}-city instances"):
                    # Use different seed for each instance to ensure diversity
                    instance_seed = self.seed + instance_id * 1000 + city_num
                    
                    # Generate environment with unique coordinates
                    env = TSPEnvironment(n_cities=city_num, seed=instance_seed, use_tsplib=False)
                    
                    # Store instance data
                    instance_data = {
                        'instance_id': instance_id,
                        'city_num': city_num,
                        'coordinates': env.coordinates.tolist(),
                        'distance_matrix': env.distance_matrix.tolist(),
                        'optimal_path': env.optimal_path,
                        'optimal_distance': float(env.optimal_distance),
                        'seed': instance_seed
                    }
                    instances.append(instance_data)
                
                # Split dataset using configured ratio
                train_instances, test_instances = self.split_dataset(instances, train_ratio)
                
                # Save split datasets
                with open(train_file, 'w') as f:
                    json.dump(train_instances, f, indent=2)
                with open(test_file, 'w') as f:
                    json.dump(test_instances, f, indent=2)
                
                # Also save metadata
                metadata_file = os.path.join(self.data_dir, f"random_tsp_{city_num}cities_{num_instances}instances_metadata.json")
                metadata = {
                    'city_num': city_num,
                    'num_instances': num_instances,
                    'train_instances': len(train_instances),
                    'test_instances': len(test_instances),
                    'train_ratio': train_ratio,
                    'generation_seed': self.seed,
                    'instance_seeds': [inst['seed'] for inst in instances],
                    'avg_optimal_distance': float(np.mean([inst['optimal_distance'] for inst in instances])),
                    'std_optimal_distance': float(np.std([inst['optimal_distance'] for inst in instances]))
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                datasets[city_num] = {'train': train_instances, 'test': test_instances}
                print(f"Generated and cached {len(train_instances)} train + {len(test_instances)} test instances for {city_num} cities")
        
        return datasets
    
    def generate_tsplib_datasets(self, 
                               tsplib_problems: List[str],
                               force_regenerate: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Generate TSPLIB datasets.
        
        Args:
            tsplib_problems: List of TSPLIB problem names
            force_regenerate: Whether to regenerate even if cached data exists
            
        Returns:
            Dictionary mapping problem name to instance data
        """
        datasets = {}
        cache_file = os.path.join(self.data_dir, "tsplib_datasets.json")
        
        if os.path.exists(cache_file) and not force_regenerate:
            print(f"Loading cached TSPLIB datasets from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_datasets = json.load(f)
            
            # Filter to requested problems
            for problem in tsplib_problems:
                if problem in cached_datasets:
                    datasets[problem] = cached_datasets[problem]
                else:
                    print(f"Problem {problem} not found in cache, generating...")
                    datasets.update(self._generate_single_tsplib(problem))
        else:
            print("Generating new TSPLIB datasets...")
            for problem in tsplib_problems:
                datasets.update(self._generate_single_tsplib(problem))
        
        # Save/update cache as JSON
        with open(cache_file, 'w') as f:
            json.dump(datasets, f, indent=2)
            
        return datasets
    
    def _generate_single_tsplib(self, problem_name: str) -> Dict[str, Dict[str, Any]]:
        """Generate single TSPLIB dataset."""
        try:
            tsplib_data_dir = os.path.join(project_root, "data", "tsplib")
            env = TSPEnvironment(use_tsplib=True, tsplib_name=problem_name, tsplib_path=tsplib_data_dir)
            
            instance_data = {
                'problem_name': problem_name,
                'city_num': env.n_cities,
                'coordinates': env.coordinates.tolist(),
                'distance_matrix': env.distance_matrix.tolist(),
                'optimal_path': env.optimal_path,
                'optimal_distance': float(env.optimal_distance)
            }
            
            print(f"Generated TSPLIB dataset for {problem_name} ({env.n_cities} cities)")
            return {problem_name: instance_data}
            
        except Exception as e:
            print(f"Failed to generate TSPLIB dataset for {problem_name}: {e}")
            return {}
    
    def split_dataset(self, 
                     instances: List[Dict[str, Any]], 
                     train_ratio: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split dataset into train and test sets.
        
        Args:
            instances: List of instance data
            train_ratio: Ratio of instances for training
            
        Returns:
            Tuple of (train_instances, test_instances)
        """
        np.random.seed(self.seed)
        indices = np.random.permutation(len(instances))
        
        train_size = int(len(instances) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        train_instances = [instances[i] for i in train_indices]
        test_instances = [instances[i] for i in test_indices]
        
        return train_instances, test_instances
    
    def validate_dataset(self, instances: List[Dict[str, Any]]) -> bool:
        """
        Validate generated dataset.
        
        Args:
            instances: List of instance data to validate
            
        Returns:
            True if dataset is valid, False otherwise
        """
        for instance in instances:
            # Check required fields
            required_fields = ['coordinates', 'distance_matrix', 'optimal_path', 'optimal_distance']
            if not all(field in instance for field in required_fields):
                print(f"Missing required fields in instance {instance.get('instance_id', 'unknown')}")
                return False
            
            # Check coordinate uniqueness
            coordinates = np.array(instance['coordinates'])
            n_cities = len(coordinates)
            
            # Check for duplicate coordinates
            for i in range(n_cities):
                for j in range(i + 1, n_cities):
                    if np.allclose(coordinates[i], coordinates[j], atol=1e-6):
                        print(f"Duplicate coordinates found in instance {instance.get('instance_id', 'unknown')}: "
                             f"cities {i} and {j}")
                        return False
            
            # Validate optimal path
            optimal_path = instance['optimal_path']
            if len(optimal_path) != n_cities + 1:
                print(f"Invalid optimal path length in instance {instance.get('instance_id', 'unknown')}")
                return False
            
            if optimal_path[0] != 0 or optimal_path[-1] != 0:
                print(f"Optimal path doesn't start/end at city 0 in instance {instance.get('instance_id', 'unknown')}")
                return False
            
            # Check that all cities are visited exactly once (except start/end)
            visited_cities = set(optimal_path[:-1])
            if len(visited_cities) != n_cities:
                print(f"Not all cities visited in optimal path for instance {instance.get('instance_id', 'unknown')}")
                return False
        
        print(f"Dataset validation passed for {len(instances)} instances")
        return True