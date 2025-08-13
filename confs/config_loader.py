import yaml
import os
from typing import Dict, Any
from .path import project_root

class ConfigLoader:
    def __init__(self):
        self.config_dir = os.path.join(project_root, "confs")
        self.experiment_config = None
        self.model_config = None
        
    def load_experiment_config(self, config_file: str = "experiment_config.yaml") -> Dict[str, Any]:
        """Load experiment configuration from YAML file."""
        config_path = os.path.join(self.config_dir, config_file)
        with open(config_path, 'r') as f:
            self.experiment_config = yaml.safe_load(f)
        return self.experiment_config
    
    def load_model_config(self, config_file: str = "model_config.yaml") -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        config_path = os.path.join(self.config_dir, config_file)
        with open(config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        return self.model_config
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """Get configuration for specific algorithm."""
        if self.model_config is None:
            self.load_model_config()
        return self.model_config.get(algorithm, {})
    
    def get_state_type_config(self, state_type: str) -> list:
        """Get state components for specific state type."""
        if self.experiment_config is None:
            self.load_experiment_config()
        return self.experiment_config["state_types"].get(state_type, [])

# Global config loader instance
config_loader = ConfigLoader()