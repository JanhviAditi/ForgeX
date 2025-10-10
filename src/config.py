# -*- coding: utf-8 -*-
"""
Configuration module for document forgery detection project.

This module contains configuration settings, constants, and helper functions
used throughout the project.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import yaml
import json


@dataclass
class DataConfig:
    """Configuration for data handling."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    interim_data_path: str = "data/interim"
    external_data_path: str = "data/external"
    
    # Data splits
    train_split: float = 0.7
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Image processing
    target_image_size: Tuple[int, int] = (224, 224)
    supported_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    
    # Data augmentation
    augmentation_factor: int = 2
    apply_augmentation: bool = False


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Texture features
    gabor_frequencies: List[float] = None
    gabor_orientations: List[int] = None  # in degrees
    
    # Edge detection
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    sobel_kernel_size: int = 3
    
    # Local Binary Pattern
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    # GLCM parameters
    glcm_distances: List[int] = None
    glcm_angles: List[float] = None
    glcm_levels: int = 8
    
    def __post_init__(self):
        if self.gabor_frequencies is None:
            self.gabor_frequencies = [0.1, 0.3, 0.5]
        if self.gabor_orientations is None:
            self.gabor_orientations = [0, 45, 90, 135]
        if self.glcm_distances is None:
            self.glcm_distances = [1, 2, 3]
        if self.glcm_angles is None:
            self.glcm_angles = [0, 0.785, 1.571, 2.356]  # 0, 45, 90, 135 degrees in radians


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # General
    random_state: int = 42
    models_path: str = "models"
    
    # Traditional ML
    traditional_ml_models: Dict = None
    use_grid_search: bool = False
    cv_folds: int = 5
    
    # Deep Learning
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.2
    
    # CNN Architecture
    cnn_dropout_rate: float = 0.5
    cnn_batch_norm: bool = True
    
    # Transfer Learning
    transfer_learning_models: List[str] = None
    freeze_base_model: bool = True
    fine_tune_epochs: int = 20
    fine_tune_lr: float = 0.0001
    
    def __post_init__(self):
        if self.traditional_ml_models is None:
            self.traditional_ml_models = {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                },
                'gradient_boosting': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                },
                'svm': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                },
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000
                }
            }
        
        if self.transfer_learning_models is None:
            self.transfer_learning_models = ['vgg16', 'resnet50', 'efficientnet']


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str] = None
    cross_validation: bool = True
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_feature_importance: bool = True
    save_plots: bool = True
    plots_dpi: int = 300
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']


@dataclass
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = None
    features: FeatureConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None
    
    # Paths
    project_root: str = "."
    src_path: str = "src"
    notebooks_path: str = "notebooks"
    reports_path: str = "reports"
    figures_path: str = "reports/figures"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


# Global configuration instance
CONFIG = ProjectConfig()


def load_config_from_file(config_path: str) -> ProjectConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ProjectConfig instance
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration data
    if config_file.suffix.lower() in ['.yaml', '.yml']:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
    elif config_file.suffix.lower() == '.json':
        with open(config_file, 'r') as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
    
    # Create configuration objects
    data_config = DataConfig(**config_data.get('data', {}))
    features_config = FeatureConfig(**config_data.get('features', {}))
    model_config = ModelConfig(**config_data.get('model', {}))
    evaluation_config = EvaluationConfig(**config_data.get('evaluation', {}))
    
    # Create main config
    main_config_data = {k: v for k, v in config_data.items() 
                       if k not in ['data', 'features', 'model', 'evaluation']}
    
    config = ProjectConfig(
        data=data_config,
        features=features_config,
        model=model_config,
        evaluation=evaluation_config,
        **main_config_data
    )
    
    return config


def save_config_to_file(config: ProjectConfig, config_path: str) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: ProjectConfig instance
        config_path: Path to save configuration file
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    config_dict = {
        'data': config.data.__dict__,
        'features': config.features.__dict__,
        'model': config.model.__dict__,
        'evaluation': config.evaluation.__dict__,
        'project_root': config.project_root,
        'src_path': config.src_path,
        'notebooks_path': config.notebooks_path,
        'reports_path': config.reports_path,
        'figures_path': config.figures_path,
        'log_level': config.log_level,
        'log_format': config.log_format
    }
    
    # Save configuration
    if config_file.suffix.lower() in ['.yaml', '.yml']:
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif config_file.suffix.lower() == '.json':
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    
    # Go up the directory tree until we find a directory with setup.py or README.md
    for parent in current_file.parents:
        if (parent / 'setup.py').exists() or (parent / 'README.md').exists():
            return parent
    
    # If not found, return the parent directory of src
    return current_file.parent.parent


def setup_directories(config: Optional[ProjectConfig] = None) -> None:
    """
    Create necessary project directories.
    
    Args:
        config: ProjectConfig instance. If None, uses global CONFIG.
    """
    if config is None:
        config = CONFIG
    
    project_root = get_project_root()
    
    directories = [
        config.data.raw_data_path,
        config.data.processed_data_path,
        config.data.interim_data_path,
        config.data.external_data_path,
        config.model.models_path,
        config.reports_path,
        config.figures_path
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)


def get_data_paths(config: Optional[ProjectConfig] = None) -> Dict[str, Path]:
    """
    Get all data paths as Path objects.
    
    Args:
        config: ProjectConfig instance. If None, uses global CONFIG.
        
    Returns:
        Dictionary mapping data type to Path objects
    """
    if config is None:
        config = CONFIG
    
    project_root = get_project_root()
    
    return {
        'raw': project_root / config.data.raw_data_path,
        'processed': project_root / config.data.processed_data_path,
        'interim': project_root / config.data.interim_data_path,
        'external': project_root / config.data.external_data_path,
        'models': project_root / config.model.models_path,
        'reports': project_root / config.reports_path,
        'figures': project_root / config.figures_path
    }


def validate_config(config: ProjectConfig) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: ProjectConfig instance to validate
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Validate data splits
    total_split = config.data.train_split + config.data.validation_split + config.data.test_split
    if abs(total_split - 1.0) > 1e-6:
        errors.append("Data splits must sum to 1.0")
    
    if any(split <= 0 for split in [config.data.train_split, 
                                   config.data.validation_split, 
                                   config.data.test_split]):
        errors.append("All data splits must be positive")
    
    # Validate image size
    if not all(isinstance(s, int) and s > 0 for s in config.data.target_image_size):
        errors.append("Target image size must be positive integers")
    
    # Validate model parameters
    if config.model.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    if config.model.epochs <= 0:
        errors.append("Number of epochs must be positive")
    
    if config.model.learning_rate <= 0:
        errors.append("Learning rate must be positive")
    
    # Validate feature parameters
    if not all(f > 0 for f in config.features.gabor_frequencies):
        errors.append("Gabor frequencies must be positive")
    
    if config.features.lbp_radius <= 0:
        errors.append("LBP radius must be positive")
    
    return errors


# Constants
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_RANDOM_STATE = 42

# Model type constants
MODEL_TYPES = {
    'TRADITIONAL_ML': 'traditional_ml',
    'CNN': 'cnn',
    'TRANSFER_LEARNING': 'transfer_learning'
}

# Evaluation metrics
CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'roc_auc', 'average_precision'
]

# Feature categories
FEATURE_CATEGORIES = {
    'texture': ['gabor', 'lbp', 'glcm'],
    'edge': ['sobel', 'canny', 'laplacian'],
    'statistical': ['mean', 'std', 'skewness', 'kurtosis'],
    'frequency': ['fft', 'dct'],
    'compression': ['block', 'jpeg'],
    'noise': ['wavelet', 'gradient']
}