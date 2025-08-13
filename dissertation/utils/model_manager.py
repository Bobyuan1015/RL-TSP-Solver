import os
import glob
import json
import traceback
from typing import Dict, Any, Optional
import torch


class ModelManager:
    """Manage model saving with best model and recent models retention."""
    
    def __init__(self, save_dir: str, keep_recent: int = 10):
        """
        Initialize model manager.
        
        Args:
            save_dir: Directory to save models
            keep_recent: Number of recent models to keep (default: 10)
        """
        self.save_dir = save_dir
        self.keep_recent = keep_recent
        os.makedirs(save_dir, exist_ok=True)
        
        # Track best model info
        self.best_metric = float('-inf')
        self.best_model_path = None
        
        # Load existing best model info if available
        self._load_best_model_info()
    
    def _load_best_model_info(self):
        """Load best model information from metadata file."""
        metadata_file = os.path.join(self.save_dir, "model_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.best_metric = metadata.get('best_metric', float('-inf'))
                self.best_model_path = metadata.get('best_model_path')
            except Exception as e:
                print(f"Warning: Could not load model metadata: {e}")
    
    def _save_metadata(self):
        """Save best model metadata."""
        metadata = {
            'best_metric': float(self.best_metric),
            'best_model_path': self.best_model_path
        }
        metadata_file = os.path.join(self.save_dir, "model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_model(self, 
                   agent: Any, 
                   episode: int, 
                   metric_value: float, 
                   metric_name: str = "reward",
                   is_higher_better: bool = True) -> str:
        """
        Save model with smart retention policy.
        
        Args:
            agent: Agent object with save_model method
            episode: Current episode number
            metric_value: Value of the metric to track (e.g., reward, success rate)
            metric_name: Name of the metric for filename
            is_higher_better: Whether higher metric value is better
            
        Returns:
            Path to saved model file
        """
        # Create model filename
        model_filename = f"model_episode_{episode}_{metric_name}_{metric_value:.4f}.pth"
        model_path = os.path.join(self.save_dir, model_filename)
        
        # Save the model
        agent.save_model(model_path)
        
        # Check if this is the best model
        is_best = False
        if is_higher_better:
            if metric_value > self.best_metric:
                is_best = True
                self.best_metric = metric_value
        else:
            if metric_value < self.best_metric or self.best_metric == float('-inf'):
                is_best = True
                self.best_metric = metric_value
        
        # If this is the best model, save it as best and update metadata
        if is_best:
            best_model_path = os.path.join(self.save_dir, f"best_model_{metric_name}_{metric_value:.4f}.pth")
            
            # Remove old best model if exists
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                except OSError:
                    pass
            
            # Copy current model as best model
            import shutil
            shutil.copy2(model_path, best_model_path)
            self.best_model_path = best_model_path
            
            # Save metadata
            self._save_metadata()
            
            print(f"New best model saved: {best_model_path} ({metric_name}: {metric_value:.4f})")
        
        # Clean up old models (keep only recent ones + best)
        self._cleanup_old_models()
        
        return model_path
    
    def _cleanup_old_models(self):
        """Remove old models, keeping only recent ones and the best model."""
        # Get all model files (excluding best model and metadata)
        model_pattern = os.path.join(self.save_dir, "model_episode_*.pth")
        model_files = glob.glob(model_pattern)
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        # Keep only the most recent models
        if len(model_files) > self.keep_recent:
            files_to_remove = model_files[self.keep_recent:]
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    # print(f"Removed old model: {os.path.basename(file_path)}")
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e} {traceback.format_exc()}")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get path to the best model."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            return self.best_model_path
        return None
    
    def get_best_metric(self) -> float:
        """Get the best metric value."""
        return self.best_metric
    
    def load_best_model(self, agent: Any) -> bool:
        """
        Load the best model into the agent.
        
        Args:
            agent: Agent object with load_model method
            
        Returns:
            True if successfully loaded, False otherwise
        """
        best_path = self.get_best_model_path()
        if best_path:
            try:
                agent.load_model(best_path)
                print(f"Loaded best model from: {best_path}")
                return True
            except Exception as e:
                print(f"Error loading best model: {e}")
        return False
    
    def get_recent_models(self) -> list:
        """Get list of recent model files."""
        model_pattern = os.path.join(self.save_dir, "model_episode_*.pth")
        model_files = glob.glob(model_pattern)
        model_files.sort(key=os.path.getmtime, reverse=True)
        return model_files[:self.keep_recent]