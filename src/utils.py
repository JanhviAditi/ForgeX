# -*- coding: utf-8 -*-
"""
Utility functions for the document forgery detection project.

This module contains helper functions, logging setup, file operations,
and other utility functions used throughout the project.
"""

import os
import sys
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import functools
from contextlib import contextmanager


def setup_logging(log_level: str = "INFO", 
                 log_format: Optional[str] = None,
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string
        log_file: Optional file to save logs
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler if specified
    handlers = [console_handler]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Get root logger and add handlers
    logger = logging.getLogger()
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger


def get_file_hash(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read at a time
        
    Returns:
        Hexadecimal hash string
    """
    hash_sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)
    
    return hash_sha256.hexdigest()


def create_directory_structure(base_path: Union[str, Path], 
                             directories: List[str]) -> None:
    """
    Create a directory structure.
    
    Args:
        base_path: Base directory path
        directories: List of directory paths to create
    """
    base_path = Path(base_path)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)


def find_files(directory: Union[str, Path], 
               extensions: List[str], 
               recursive: bool = True) -> List[Path]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Directory to search in
        extensions: List of file extensions (with or without dots)
        recursive: Whether to search recursively
        
    Returns:
        List of Path objects for found files
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    # Normalize extensions
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    files = []
    pattern = '**/*' if recursive else '*'
    
    for ext in extensions:
        files.extend(directory.glob(f'{pattern}{ext}'))
        files.extend(directory.glob(f'{pattern}{ext.upper()}'))
    
    return sorted(list(set(files)))


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded data
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {duration:.4f} seconds")
        
        return result
    
    return wrapper


def memory_usage(func: Callable) -> Callable:
    """
    Decorator to measure memory usage of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_diff = mem_after - mem_before
            
            logger = logging.getLogger(func.__module__)
            logger.info(f"{func.__name__} memory usage: {mem_diff:+.2f} MB")
            
            return result
            
        except ImportError:
            logger = logging.getLogger(func.__module__)
            logger.warning("psutil not available for memory monitoring")
            return func(*args, **kwargs)
    
    return wrapper


@contextmanager
def timer(name: str):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the timed operation
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info(f"Starting {name}...")
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{name} completed in {duration:.4f} seconds")


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Ensure reproducibility by setting random seeds.
    
    Args:
        seed: Random seed value
    """
    import random
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set TensorFlow random seed if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Set PyTorch random seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging and debugging.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'hostname': platform.node(),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
        info['disk_usage'] = {
            'total_gb': psutil.disk_usage('/').total / (1024**3),
            'free_gb': psutil.disk_usage('/').free / (1024**3)
        }
    except ImportError:
        pass
    
    return info


def validate_image_file(file_path: Union[str, Path], 
                       supported_formats: List[str] = None) -> bool:
    """
    Validate if a file is a supported image format.
    
    Args:
        file_path: Path to the image file
        supported_formats: List of supported file extensions
        
    Returns:
        True if valid image file, False otherwise
    """
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False
    
    # Check extension
    if file_path.suffix.lower() not in [ext.lower() for ext in supported_formats]:
        return False
    
    # Try to open with PIL to verify it's a valid image
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def create_experiment_directory(base_path: Union[str, Path], 
                              experiment_name: Optional[str] = None) -> Path:
    """
    Create a directory for an experiment with timestamp.
    
    Args:
        base_path: Base directory for experiments
        experiment_name: Optional experiment name
        
    Returns:
        Path to the created experiment directory
    """
    base_path = Path(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = base_path / f"{experiment_name}_{timestamp}"
    else:
        exp_dir = base_path / f"experiment_{timestamp}"
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['models', 'results', 'logs', 'plots']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(exist_ok=True)
    
    return exp_dir


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human readable format.
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds into human readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    
    days = hours / 24
    return f"{days:.1f}d"


def safe_division(numerator: float, denominator: float, 
                 default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def create_progress_callback(total_items: int, 
                           description: str = "Processing") -> Callable:
    """
    Create a progress callback function for long-running operations.
    
    Args:
        total_items: Total number of items to process
        description: Description of the operation
        
    Returns:
        Callback function
    """
    logger = logging.getLogger(__name__)
    
    def callback(current_item: int):
        if current_item % max(1, total_items // 10) == 0 or current_item == total_items:
            percentage = (current_item / total_items) * 100
            logger.info(f"{description}: {current_item}/{total_items} ({percentage:.1f}%)")
    
    return callback


class ExperimentLogger:
    """Logger for machine learning experiments."""
    
    def __init__(self, experiment_dir: Union[str, Path]):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Directory to save experiment logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.experiment_dir / "experiment.log"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.config_file = self.experiment_dir / "config.json"
        
        self.logger = setup_logging(log_file=str(self.log_file))
        self.metrics = {}
        self.start_time = time.time()
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        save_json(config, self.config_file)
        self.logger.info("Experiment configuration saved")
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a metric value."""
        timestamp = time.time()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'timestamp': timestamp,
            'step': step
        })
        
        # Save metrics
        save_json(self.metrics, self.metrics_file)
        
        # Log to console
        if step is not None:
            self.logger.info(f"Step {step} - {name}: {value}")
        else:
            self.logger.info(f"{name}: {value}")
    
    def log_artifact(self, artifact_name: str, artifact_path: Union[str, Path]) -> None:
        """Log an artifact (model, plot, etc.)."""
        artifact_path = Path(artifact_path)
        dest_path = self.experiment_dir / artifact_name
        
        if artifact_path.is_file():
            import shutil
            shutil.copy2(artifact_path, dest_path)
            self.logger.info(f"Artifact saved: {artifact_name}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary."""
        duration = time.time() - self.start_time
        
        summary = {
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'total_metrics': len(self.metrics),
            'metrics_names': list(self.metrics.keys()),
            'experiment_dir': str(self.experiment_dir)
        }
        
        return summary


# Global utility functions
def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def chunks(lst: List, chunk_size: int):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]