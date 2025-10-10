#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command Line Interface for Document Forgery Detection.

This script provides a unified interface for all project operations including
data preprocessing, feature extraction, model training, and prediction.
"""

import sys
import click
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import load_config_from_file, CONFIG, setup_directories, get_data_paths
from utils import setup_logging, timing_decorator, ExperimentLogger
from data.make_dataset import create_dataset_structure, augment_images
from features.build_features import extract_features_from_directory
from models.train_model import DocumentForgeryDetector
from models.predict_model import DocumentForgeryPredictor
from visualization.visualize import create_comprehensive_report


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Document Forgery Detection CLI - Detect forged documents using ML/AI."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    if config:
        try:
            global_config = load_config_from_file(config)
            click.echo(f"Loaded configuration from {config}")
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            global_config = CONFIG
    else:
        global_config = CONFIG
    
    # Setup directories
    setup_directories(global_config)
    
    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj['config'] = global_config
    ctx.obj['paths'] = get_data_paths(global_config)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Output directory for processed data')
@click.option('--train-split', default=0.7, help='Training split ratio')
@click.option('--val-split', default=0.2, help='Validation split ratio')
@click.option('--test-split', default=0.1, help='Test split ratio')
@click.option('--augment', is_flag=True, help='Apply data augmentation')
@click.pass_context
@timing_decorator
def preprocess_data(ctx, input_dir, output_dir, train_split, val_split, test_split, augment):
    """Preprocess raw document images for training."""
    
    config = ctx.obj['config']
    paths = ctx.obj['paths']
    
    if not output_dir:
        output_dir = str(paths['processed'])
    
    click.echo(f"Preprocessing data from {input_dir} to {output_dir}")
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        click.echo("Error: Train, validation, and test splits must sum to 1.0", err=True)
        return
    
    try:
        # Create dataset structure
        create_dataset_structure(
            input_dir=input_dir,
            output_dir=output_dir,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split
        )
        
        # Apply augmentation if requested
        if augment:
            click.echo("Applying data augmentation...")
            train_dir = Path(output_dir) / "train"
            augment_dir = Path(output_dir) / "train_augmented"
            
            augment_images(str(train_dir), str(augment_dir), 
                         augmentation_factor=config.data.augmentation_factor)
        
        click.echo("✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during preprocessing: {e}", err=True)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-file', '-o', help='Output CSV file for features')
@click.option('--image-size', nargs=2, type=int, help='Target image size (width height)')
@click.pass_context
@timing_decorator
def extract_features(ctx, input_dir, output_file, image_size):
    """Extract features from document images."""
    
    paths = ctx.obj['paths']
    
    if not output_file:
        output_file = str(paths['processed'] / 'features.csv')
    
    if image_size:
        target_size = tuple(image_size)
    else:
        target_size = (224, 224)
    
    click.echo(f"Extracting features from {input_dir}")
    click.echo(f"Target image size: {target_size}")
    
    try:
        extract_features_from_directory(
            input_dir=input_dir,
            output_file=output_file,
            image_size=target_size
        )
        
        click.echo(f"✅ Features extracted successfully to {output_file}")
        
    except Exception as e:
        click.echo(f"❌ Error during feature extraction: {e}", err=True)


@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--model-type', type=click.Choice(['traditional_ml', 'cnn', 'transfer_learning']),
              default='traditional_ml', help='Type of model to train')
@click.option('--model-name', default='random_forest', help='Name of traditional ML model')
@click.option('--features-path', type=click.Path(exists=True), help='Path to features CSV')
@click.option('--epochs', default=50, help='Number of training epochs (for deep learning)')
@click.option('--batch-size', default=32, help='Batch size')
@click.option('--experiment-name', help='Name of the experiment')
@click.pass_context
@timing_decorator
def train_model(ctx, data_path, model_type, model_name, features_path, 
               epochs, batch_size, experiment_name):
    """Train a document forgery detection model."""
    
    config = ctx.obj['config']
    paths = ctx.obj['paths']
    
    # Create experiment directory
    if experiment_name:
        from utils import create_experiment_directory
        exp_dir = create_experiment_directory(paths['reports'], experiment_name)
        exp_logger = ExperimentLogger(exp_dir)
    else:
        exp_logger = None
    
    click.echo(f"Training {model_type} model on {data_path}")
    
    try:
        # Initialize detector
        detector = DocumentForgeryDetector(
            model_type=model_type, 
            random_state=config.model.random_state
        )
        
        # Load data
        if model_type == 'traditional_ml' and features_path:
            X, y = detector.load_data(features_path)
        else:
            X, y = detector.load_data(data_path)
        
        # Preprocess data
        X_processed, y_processed = detector.preprocess_data(X, y)
        
        # Log experiment config
        if exp_logger:
            exp_config = {
                'model_type': model_type,
                'model_name': model_name,
                'data_path': data_path,
                'features_path': features_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'data_shape': X_processed.shape,
                'labels_shape': y_processed.shape
            }
            exp_logger.log_config(exp_config)
        
        # Train model
        if model_type == 'traditional_ml':
            results = detector.train_model(
                X_processed, y_processed,
                model_name=model_name,
                use_grid_search=config.model.use_grid_search
            )
        else:
            results = detector.train_model(
                X_processed, y_processed,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Log results
        if exp_logger:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    exp_logger.log_metric(key, value)
        
        # Save model
        if model_type == 'traditional_ml':
            model_path = paths['models'] / f"{model_name}_model.joblib"
        else:
            model_path = paths['models'] / f"{model_type}_model.h5"
        
        detector.save_model(str(model_path), metadata=results)
        
        if exp_logger:
            exp_logger.log_artifact("model", model_path)
        
        # Plot training history for deep learning
        if model_type != 'traditional_ml' and hasattr(detector, 'history'):
            plot_path = paths['figures'] / f"{model_type}_training_history.png"
            detector.plot_training_history(str(plot_path))
            
            if exp_logger:
                exp_logger.log_artifact("training_history.png", plot_path)
        
        click.echo("✅ Model training completed successfully!")
        click.echo(f"Model saved to: {model_path}")
        
        if exp_logger:
            summary = exp_logger.get_experiment_summary()
            click.echo(f"Experiment logged to: {summary['experiment_dir']}")
        
    except Exception as e:
        click.echo(f"❌ Error during training: {e}", err=True)
        if exp_logger:
            exp_logger.log_metric("error", str(e))


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output CSV file for results')
@click.option('--model-type', type=click.Choice(['traditional_ml', 'cnn', 'transfer_learning', 'auto']),
              default='auto', help='Type of model')
@click.option('--batch-size', default=32, help='Batch size for processing')
@click.option('--visualize', is_flag=True, help='Create visualization of results')
@click.pass_context
@timing_decorator
def predict(ctx, model_path, input_path, output, model_type, batch_size, visualize):
    """Make predictions on document images."""
    
    paths = ctx.obj['paths']
    
    if not output:
        output = str(paths['reports'] / 'predictions.csv')
    
    click.echo(f"Making predictions with model: {model_path}")
    click.echo(f"Input: {input_path}")
    
    try:
        # Initialize predictor
        predictor = DocumentForgeryPredictor(model_path, model_type)
        
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Single image prediction
            result = predictor.predict_single_image(input_path)
            
            click.echo(f"\nPrediction for {input_path_obj.name}:")
            click.echo(f"  Class: {result.get('prediction', 'Unknown')}")
            click.echo(f"  Confidence: {result.get('confidence', 0):.4f}")
            
            if 'probabilities' in result:
                click.echo("  Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    click.echo(f"    {class_name}: {prob:.4f}")
        
        elif input_path_obj.is_dir():
            # Directory prediction
            df_results = predictor.predict_directory(input_path, output)
            
            if not df_results.empty:
                click.echo(f"\nProcessed {len(df_results)} images")
                
                # Summary statistics
                pred_counts = df_results['prediction'].value_counts()
                click.echo("\nPrediction Summary:")
                for pred, count in pred_counts.items():
                    percentage = (count / len(df_results)) * 100
                    click.echo(f"  {pred}: {count} ({percentage:.1f}%)")
                
                click.echo(f"\nConfidence Statistics:")
                click.echo(f"  Mean: {df_results['confidence'].mean():.4f}")
                click.echo(f"  Min:  {df_results['confidence'].min():.4f}")
                click.echo(f"  Max:  {df_results['confidence'].max():.4f}")
                
                click.echo(f"\nResults saved to: {output}")
                
                # Create visualizations
                if visualize:
                    from visualization.visualize import DocumentForgeryVisualizer
                    
                    visualizer = DocumentForgeryVisualizer()
                    
                    # Convert results to list of dicts for visualization
                    predictions_list = df_results.to_dict('records')
                    
                    # Plot confidence distribution
                    conf_plot_path = paths['figures'] / 'prediction_confidence_distribution.png'
                    visualizer.plot_prediction_confidence_distribution(
                        predictions_list, str(conf_plot_path)
                    )
                    
                    # Visualize sample predictions
                    image_paths = df_results['image_path'].tolist()[:12]
                    sample_predictions = predictions_list[:12]
                    
                    pred_plot_path = paths['figures'] / 'sample_predictions.png'
                    visualizer.visualize_image_predictions(
                        image_paths, sample_predictions, save_path=str(pred_plot_path)
                    )
                    
                    click.echo(f"Visualizations saved to: {paths['figures']}")
            else:
                click.echo("No results generated")
        
        else:
            click.echo(f"Error: {input_path} is not a valid file or directory", err=True)
    
    except Exception as e:
        click.echo(f"❌ Error during prediction: {e}", err=True)


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), help='Path to trained model')
@click.option('--test-data', type=click.Path(exists=True), help='Path to test data')
@click.option('--output-dir', help='Output directory for evaluation results')
@click.pass_context
def evaluate(ctx, model_path, test_data, output_dir):
    """Evaluate model performance on test data."""
    
    paths = ctx.obj['paths']
    
    if not output_dir:
        output_dir = str(paths['reports'])
    
    click.echo("Model evaluation functionality")
    click.echo("This command would evaluate the model and generate comprehensive reports")
    click.echo(f"Model: {model_path}")
    click.echo(f"Test data: {test_data}")
    click.echo(f"Output: {output_dir}")
    
    # TODO: Implement comprehensive model evaluation
    click.echo("⚠️  Full evaluation functionality coming soon!")


@cli.command()
@click.pass_context
def setup(ctx):
    """Setup the project directory structure and configuration."""
    
    config = ctx.obj['config']
    paths = ctx.obj['paths']
    
    click.echo("Setting up Document Forgery Detection project...")
    
    # Create directories
    setup_directories(config)
    
    click.echo("📁 Created directory structure:")
    for name, path in paths.items():
        if path.exists():
            click.echo(f"  ✅ {name}: {path}")
        else:
            click.echo(f"  ❌ {name}: {path} (failed to create)")
    
    # Create sample config if it doesn't exist
    config_file = Path("config.yaml")
    if not config_file.exists():
        from config import save_config_to_file
        save_config_to_file(config, config_file)
        click.echo(f"📝 Created default configuration: {config_file}")
    
    click.echo("\n🚀 Project setup completed!")
    click.echo("\nNext steps:")
    click.echo("1. Add your document images to data/raw/")
    click.echo("2. Run: python cli.py preprocess-data data/raw/")
    click.echo("3. Run: python cli.py extract-features data/processed/train/")
    click.echo("4. Run: python cli.py train-model data/processed/ --features-path data/processed/features.csv")


@cli.command()
@click.pass_context
def info(ctx):
    """Display project and system information."""
    
    config = ctx.obj['config']
    paths = ctx.obj['paths']
    
    click.echo("📋 Document Forgery Detection - Project Information")
    click.echo("=" * 60)
    
    # Project structure
    click.echo("\n📁 Project Structure:")
    for name, path in paths.items():
        status = "✅" if path.exists() else "❌"
        click.echo(f"  {status} {name}: {path}")
    
    # Configuration summary
    click.echo(f"\n⚙️  Configuration:")
    click.echo(f"  Image size: {config.data.target_image_size}")
    click.echo(f"  Data splits: train={config.data.train_split}, val={config.data.validation_split}, test={config.data.test_split}")
    click.echo(f"  Random seed: {config.model.random_state}")
    click.echo(f"  Log level: {config.log_level}")
    
    # System information
    from utils import get_system_info
    sys_info = get_system_info()
    
    click.echo(f"\n💻 System Information:")
    click.echo(f"  Python: {sys_info['python_version'].split()[0]}")
    click.echo(f"  Platform: {sys_info['platform']}")
    if 'cpu_count' in sys_info:
        click.echo(f"  CPU cores: {sys_info['cpu_count']}")
    if 'memory_gb' in sys_info:
        click.echo(f"  Memory: {sys_info['memory_gb']:.1f} GB")
    
    # Check dependencies
    click.echo(f"\n📦 Key Dependencies:")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            click.echo(f"  ✅ {name}")
        except ImportError:
            click.echo(f"  ❌ {name} (not installed)")
    
    # Optional dependencies
    click.echo(f"\n📦 Optional Dependencies:")
    
    optional_deps = [
        ('tensorflow', 'TensorFlow'),
        ('torch', 'PyTorch'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            click.echo(f"  ✅ {name}")
        except ImportError:
            click.echo(f"  ⚠️  {name} (optional)")


if __name__ == '__main__':
    cli()