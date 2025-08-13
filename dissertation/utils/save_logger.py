import os
import logging
import csv
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from confs.path import project_root


class Logger:
    """Comprehensive logging system for TSP experiments."""
    
    def __init__(self, 
                 experiment_name: str,
                 algorithm: str = None,
                 mode: str = None,
                 log_level: str = "INFO",
                 save_to_file: bool = True,
                 console_output: bool = True):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of the experiment
            algorithm: Algorithm name for file naming
            mode: Mode name for file naming
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_to_file: Whether to save logs to file
            console_output: Whether to output logs to console
        """
        self.buffer_size = 100  # 缓存1000条记录再写入
        self.csv_buffer = []

        self.experiment_name = experiment_name
        self.algorithm = algorithm
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create experiment directory with algorithm and mode in path

        dir_name = f"{experiment_name}_{self.timestamp}"
        
        self.experiment_dir = os.path.join(project_root, "results", dir_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Setup standard logger
        self.logger = self._setup_logger(log_level, save_to_file, console_output)
        
        # CSV data storage
        self.csv_data = []
        self.csv_headers = [
            'algorithm', 'city_num', 'mode', 'instance_id', 'run_id', 'state_type',
            'train_test', 'episode', 'step', 'action', 'state', 'done', 'reward', 
            'loss', 'total_reward', 'current_distance','optimal_distance','optimal_path',
            'state_values', 'timestamp'
        ]
        
        # Initialize CSV file with algorithm and mode in filename
        csv_filename = f"experiment_data_{self.timestamp}.csv"
        
        self.csv_file = os.path.join(self.experiment_dir, csv_filename)
        self._init_csv_file()
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.timestamp,
            'experiment_dir': self.experiment_dir
        }
        
        if self.algorithm and self.mode:
            self.logger.info(f"Initialized logger for experiment: {experiment_name} [{self.algorithm}-{self.mode}]")
        else:
            self.logger.info(f"Initialized logger for experiment: {experiment_name}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_logger(self, log_level: str, save_to_file: bool, console_output: bool) -> logging.Logger:
        """Setup logger with file and console handlers."""
        # Create unique logger name with algorithm and mode
        if self.algorithm and self.mode:
            logger_name = f"tsp_{self.algorithm}_{self.mode}_{self.timestamp}"
        else:
            logger_name = f"tsp_experiment_{self.experiment_name}_{self.timestamp}"
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter with algorithm and mode prefix
        if self.algorithm and self.mode:
            format_str = f'[%(asctime)s][{self.algorithm}-{self.mode}][%(levelname)s]: %(message)s'
        else:
            format_str = '[%(asctime)s][%(name)s][%(levelname)s]: %(message)s'
        
        formatter = logging.Formatter(
            format_str,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with algorithm and mode in filename
        if save_to_file:
            if self.algorithm and self.mode:
                log_filename = f"{self.algorithm}_{self.mode}_experiment_{self.timestamp}.log"
            else:
                log_filename = f"experiment_{self.timestamp}.log"
            
            log_file = os.path.join(self.experiment_dir, log_filename)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def log_training_step(self, 
                         algorithm: str,
                         city_num: int,
                         mode: str,
                         instance_id: int,
                         run_id: int,
                         state_type: str,
                         train_test: str,
                         episode: int,
                         step: int,
                         action: int,
                         state: Dict[str, Any],
                         done: bool,
                         reward: float,
                         loss: Optional[float] = None,
                         total_reward: Optional[float] = None,
                         current_distance=None,
                         optimal_distance=None,
                         optimal_path=[],
                         coordinates=None,
                         state_values=None):
        """
        Log a single training/testing step.
        
        Args:
            algorithm: Algorithm name
            city_num: Number of cities
            mode: Training mode (per_instance or cross_instance)
            instance_id: Instance ID
            run_id: Run ID
            state_type: State representation type
            train_test: Whether this is training or testing data
            episode: Episode number
            step: Step number within episode
            action: Action taken
            state: State representation
            done: Whether episode is done
            reward: Reward received
            loss: Loss value (if available)
            total_reward: Total reward so far in episode
            current_distance: Current tour length
            optimal_distance: Optimal distance for this instance
            optimal_path: Optimal path for this instance
            coordinates: City coordinates for this instance
            state_values: State type components values
        """
        # Format state as JSON string (only include numeric values for CSV)
        state_json = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                state_json[key] = float(value)
            elif hasattr(value, 'tolist'):  # numpy array
                state_json[key] = value.tolist()
        
        # Format coordinates and state_values as JSON strings
        state_values_json = json.dumps(state_values) if state_values is not None else ''
        
        # Create row data
        row_data = [
            algorithm, city_num, mode, instance_id, run_id, state_type,
            train_test, episode, step, action, json.dumps(state_json), 
            int(done), reward, loss or '', total_reward or '', current_distance,optimal_distance, optimal_path,
            state_values_json, datetime.now().isoformat()
        ]
        self.csv_buffer.append(row_data)
        # 达到缓存大小时批量写入
        if len(self.csv_buffer) >= self.buffer_size:
            self._flush_buffer()


        
        # Log to standard logger with simplified format (algorithm/mode already in logger prefix)
        log_msg = (f"[{city_num}cities][{state_type}][Episode {episode}][Step {step}]: "
                  f"Action={action}, Reward={reward:.4f}, Done={done}")
        
        if loss is not None:
            log_msg += f", Loss={loss:.6f}"
        
        self.logger.debug(log_msg)

    def _flush_buffer(self):
        if self.csv_buffer:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.csv_buffer)
            self.csv_buffer.clear()

    def log_episode_summary(self,
                           algorithm: str,
                           city_num: int,
                           mode: str,
                           instance_id: int,
                           run_id: int,
                           state_type: str,
                           train_test: str,
                           episode: int,
                           total_reward: float,
                           episode_length: int,
                           metrics: Dict[str, float] = None):
        """
        Log episode summary.
        
        Args:
            algorithm: Algorithm name
            city_num: Number of cities
            mode: Training mode
            instance_id: Instance ID
            run_id: Run ID
            state_type: State representation type
            train_test: Training or testing
            episode: Episode number
            total_reward: Total episode reward
            episode_length: Number of steps in episode
            metrics: Additional metrics (loss, etc.)
        """
        # Simplified format since algorithm/mode already in logger prefix
        log_msg = (f"[{city_num}cities][{state_type}][Episode {episode} SUMMARY]: "
                  f"Reward={total_reward:.4f}, Steps={episode_length}")
        
        if metrics:
            metric_strs = [f"{k}={v:.6f}" for k, v in metrics.items()]
            log_msg += f", {', '.join(metric_strs)}"
        
        self.logger.info(log_msg)
    
    def log_experiment_progress(self,
                              algorithm: str,
                              city_num: int,
                              mode: str,
                              current_instance: int,
                              total_instances: int,
                              current_run: int,
                              total_runs: int,
                              index_instance:int,
                              current_task: int = None,
                              total_tasks: int = None):
        """Log overall experiment progress."""
        # Calculate progress percentages
        instance_pct = ((index_instance+1) / total_instances) * 100 if total_instances > 0 else 0
        run_pct = (current_run / total_runs) * 100 if total_runs > 0 else 0

        progress_msg = (f"[{city_num}cities] Progress: "
                       f"Instance {index_instance+1}/{total_instances} ({instance_pct:.1f}%), "
                       f"Run {current_run}/{total_runs} ({run_pct:.1f}%)")
        
        # Add total progress if provided
        if current_task is not None and total_tasks is not None and total_tasks > 0:
            total_pct = (current_task / total_tasks) * 100
            progress_msg += f" | Overall: {current_task}/{total_tasks} ({total_pct:.1f}%)"
        
        # Update progress_dict if available
        if hasattr(self, 'progress_dict') and self.progress_dict is not None and hasattr(self, 'task_key') and self.task_key:
            try:
                # Update progress details in a way that works with multiprocessing Manager
                progress_data = dict(self.progress_dict.get(self.task_key, {}))
                progress_data.update({
                    "current": current_task if current_task is not None else 0,
                    "total": total_tasks if total_tasks is not None else 0,
                    "status": "running",
                    "details": {
                        "city_num": city_num,
                        "mode": mode,
                        "current_instance": current_instance,
                        "total_instances": total_instances,
                        "current_run": current_run,
                        "total_runs": total_runs,
                        "instance_pct": instance_pct,
                        "run_pct": run_pct,
                        "total_pct": total_pct if current_task is not None and total_tasks is not None and total_tasks > 0 else 0
                    }
                })
                self.progress_dict[self.task_key] = progress_data
            except Exception as e:
                self.logger.warning(f"Failed to update progress_dict: {e} {traceback.format_exc()}")
        
        self.logger.info(progress_msg)
    
    def log_performance_metrics(self,
                              algorithm: str,
                              city_num: int,
                              state_type: str,
                              mode: str,
                              metrics: Dict[str, float]):
        """Log performance metrics."""
        # Simplified format since algorithm/mode already in logger prefix
        metrics_msg = (f"[{city_num}cities][{state_type}] Performance Metrics: ")
        
        metric_strs = [f"{k}={v:.6f}" for k, v in metrics.items()]
        metrics_msg += ", ".join(metric_strs)
        
        self.logger.info(metrics_msg)
    
    def save_experiment_summary(self, summary_data: Dict[str, Any]):
        """Save experiment summary to JSON file."""
        if self.algorithm and self.mode:
            summary_filename = f"{self.algorithm}_{self.mode}_experiment_summary_{self.timestamp}.json"
        else:
            summary_filename = f"experiment_summary_{self.timestamp}.json"
        
        summary_file = os.path.join(self.experiment_dir, summary_filename)
        
        summary_data.update(self.metadata)
        summary_data['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_data['algorithm'] = self.algorithm
        summary_data['mode'] = self.mode
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Experiment summary saved to {summary_file}")
    
    def create_algorithm_csv(self, algorithm: str) -> str:
        """Create separate CSV file for specific algorithm."""
        algorithm_dir = os.path.join(self.experiment_dir, algorithm)
        os.makedirs(algorithm_dir, exist_ok=True)
        
        csv_filename = f"{algorithm}_data_{self.timestamp}.csv"
        algorithm_csv = os.path.join(algorithm_dir, csv_filename)
        
        # Initialize with headers
        with open(algorithm_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
        
        return algorithm_csv
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze experiment results from CSV data."""
        if not os.path.exists(self.csv_file):
            self.logger.warning("No CSV data file found for analysis")
            return {}
        
        try:
            df = pd.read_csv(self.csv_file)
            
            # Basic statistics
            analysis = {
                'total_steps': len(df),
                'algorithms': df['algorithm'].unique().tolist(),
                'city_nums': df['city_num'].unique().tolist(),
                'state_types': df['state_type'].unique().tolist(),
                'modes': df['mode'].unique().tolist()
            }
            
            # Performance by algorithm
            if 'total_reward' in df.columns:
                # Filter for episode completion (done=True)
                episode_data = df[df['done'] == 1]
                
                if not episode_data.empty:
                    analysis['performance_by_algorithm'] = {}
                    for algorithm in episode_data['algorithm'].unique():
                        alg_data = episode_data[episode_data['algorithm'] == algorithm]
                        analysis['performance_by_algorithm'][algorithm] = {
                            'mean_reward': float(alg_data['total_reward'].mean()),
                            'std_reward': float(alg_data['total_reward'].std()),
                            'episodes_completed': len(alg_data)
                        }
            
            self.logger.info("Results analysis completed")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e} {traceback.format_exc()}")
            return {}
    
    def get_experiment_dir(self) -> str:
        """Get experiment directory path."""
        return self.experiment_dir
    
    def get_csv_file(self) -> str:
        """Get CSV file path."""
        return self.csv_file