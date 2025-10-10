# -*- coding: utf-8 -*-
"""
Visualization module for document forgery detection.

This module contains functions to create various visualizations for
analyzing model performance, data distribution, and prediction results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import logging

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

# Machine Learning
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DocumentForgeryVisualizer:
    """
    A comprehensive visualization class for document forgery detection analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str] = None, 
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = ['Authentic', 'Forged']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy information
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f}', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: List[str] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot classification report as a heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = ['Authentic', 'Forged']
        
        # Get classification report as dict
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Convert to DataFrame for visualization
        df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'accuracy' row
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.iloc[:-2, :-1], annot=True, cmap='RdYlBu_r', fmt='.3f')
        plt.title('Classification Report', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      class_names: List[str] = None,
                      save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels (binary)
            y_scores: Prediction scores/probabilities
            class_names: Names of classes
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = ['Authentic', 'Forged']
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot training history for deep learning models.
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training & validation loss
        axes[0, 1].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='red')
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis('off')
        
        # Plot accuracy difference
        if 'accuracy' in history and 'val_accuracy' in history:
            acc_diff = np.array(history['accuracy']) - np.array(history['val_accuracy'])
            axes[1, 1].plot(acc_diff, linewidth=2, color='purple')
            axes[1, 1].set_title('Training vs Validation Accuracy Difference', 
                                fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy Difference')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], importance_values: np.ndarray,
                              top_k: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance for traditional ML models.
        
        Args:
            feature_names: Names of features
            importance_values: Importance values
            top_k: Number of top features to display
            save_path: Path to save the plot
        """
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1]
        
        # Select top k features
        top_indices = indices[:top_k]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance_values[top_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_importance[::-1])
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance', fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_importance[::-1])):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{value:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_data_distribution(self, df: pd.DataFrame, target_column: str = 'class',
                             save_path: Optional[str] = None) -> None:
        """
        Plot data distribution analysis.
        
        Args:
            df: DataFrame containing the data
            target_column: Name of the target column
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        class_counts = df[target_column].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Class distribution bar plot
        sns.countplot(data=df, x=target_column, ax=axes[0, 1])
        axes[0, 1].set_title('Class Counts', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Feature correlation heatmap (if numerical features exist)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, ax=axes[1, 0], cmap='coolwarm', center=0, 
                       square=True, annot=False)
            axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        else:
            axes[1, 0].axis('off')
        
        # Missing values heatmap
        if df.isnull().sum().sum() > 0:
            sns.heatmap(df.isnull(), ax=axes[1, 1], cbar=True, yticklabels=False)
            axes[1, 1].set_title('Missing Values', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                           transform=axes[1, 1].transAxes, ha='center', va='center',
                           fontsize=16, fontweight='bold')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distribution(self, df: pd.DataFrame, features: List[str],
                                target_column: str = 'class',
                                save_path: Optional[str] = None) -> None:
        """
        Plot distribution of selected features by class.
        
        Args:
            df: DataFrame containing the data
            features: List of feature names to plot
            target_column: Name of the target column
            save_path: Path to save the plot
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(features):
            if i >= len(axes):
                break
            
            # Box plot for each class
            sns.boxplot(data=df, x=target_column, y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution by Class', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_dimensionality_reduction(self, features: np.ndarray, labels: np.ndarray,
                                    method: str = 'tsne', class_names: List[str] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot 2D dimensionality reduction visualization.
        
        Args:
            features: Feature matrix
            labels: Class labels
            method: Reduction method ('tsne' or 'pca')
            class_names: Names of classes
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in np.unique(labels)]
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title = 't-SNE Visualization'
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            title = 'PCA Visualization'
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        # Handle NaN values
        features_clean = np.nan_to_num(features)
        
        reduced_features = reducer.fit_transform(features_clean)
        
        plt.figure(figsize=(10, 8))
        
        # Plot each class
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = labels == label
            plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                       c=[color], label=class_names[i] if i < len(class_names) else f'Class {label}',
                       alpha=0.6, s=60)
        
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_confidence_distribution(self, predictions: List[Dict],
                                              save_path: Optional[str] = None) -> None:
        """
        Plot distribution of prediction confidences.
        
        Args:
            predictions: List of prediction dictionaries
            save_path: Path to save the plot
        """
        # Extract confidence values and predictions
        confidences = [pred['confidence'] for pred in predictions if 'confidence' in pred]
        pred_classes = [pred['prediction'] for pred in predictions if 'prediction' in pred]
        
        if not confidences:
            self.logger.warning("No confidence values found in predictions")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall confidence distribution
        axes[0, 0].hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 0].legend()
        
        # Confidence by class
        df_pred = pd.DataFrame({'confidence': confidences, 'prediction': pred_classes})
        sns.boxplot(data=df_pred, x='prediction', y='confidence', ax=axes[0, 1])
        axes[0, 1].set_title('Confidence by Predicted Class', fontsize=14, fontweight='bold')
        
        # Confidence statistics
        conf_stats = {
            'Mean': np.mean(confidences),
            'Median': np.median(confidences),
            'Std Dev': np.std(confidences),
            'Min': np.min(confidences),
            'Max': np.max(confidences)
        }
        
        axes[1, 0].axis('off')
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in conf_stats.items()])
        axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes,
                       fontsize=14, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 0].set_title('Confidence Statistics', fontsize=14, fontweight='bold')
        
        # Low confidence predictions
        low_conf_threshold = 0.6
        low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)
        
        labels = ['High Confidence', 'Low Confidence']
        sizes = [len(confidences) - low_conf_count, low_conf_count]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title(f'Predictions by Confidence (threshold={low_conf_threshold})', 
                           fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_image_predictions(self, image_paths: List[str], predictions: List[Dict],
                                  max_images: int = 12, save_path: Optional[str] = None) -> None:
        """
        Visualize images with their predictions.
        
        Args:
            image_paths: List of paths to images
            predictions: List of prediction dictionaries
            max_images: Maximum number of images to display
            save_path: Path to save the plot
        """
        n_images = min(len(image_paths), max_images)
        n_cols = 4
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i in range(n_images):
            try:
                # Load and display image
                image = Image.open(image_paths[i])
                axes[i].imshow(image)
                axes[i].axis('off')
                
                # Add prediction information
                pred = predictions[i]
                title = f"Pred: {pred.get('prediction', 'Unknown')}\n"
                title += f"Conf: {pred.get('confidence', 0):.3f}"
                
                # Color based on prediction
                color = 'green' if pred.get('prediction', '').lower() == 'authentic' else 'red'
                axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
                
            except Exception as e:
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f'Error loading\n{Path(image_paths[i]).name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_comprehensive_report(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_scores: Optional[np.ndarray] = None,
                              class_names: List[str] = None,
                              output_dir: str = 'reports/figures') -> None:
    """
    Create a comprehensive visualization report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (for ROC curve)
        class_names: Names of classes
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = DocumentForgeryVisualizer()
    
    # Confusion Matrix
    visualizer.plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=output_path / 'confusion_matrix.png'
    )
    
    # Classification Report
    visualizer.plot_classification_report(
        y_true, y_pred, class_names,
        save_path=output_path / 'classification_report.png'
    )
    
    # ROC Curve (if scores available)
    if y_scores is not None:
        visualizer.plot_roc_curve(
            y_true, y_scores, class_names,
            save_path=output_path / 'roc_curve.png'
        )
    
    logging.info(f"Comprehensive report saved to {output_dir}")


def main():
    """Main function for testing visualization functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations for document forgery detection')
    parser.add_argument('--demo', action='store_true', help='Run demonstration with sample data')
    
    args = parser.parse_args()
    
    if args.demo:
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        # Sample predictions
        y_true = np.random.choice([0, 1], n_samples)
        y_pred = np.random.choice([0, 1], n_samples)
        y_scores = np.random.random(n_samples)
        
        # Sample features
        features = np.random.randn(n_samples, 50)
        
        visualizer = DocumentForgeryVisualizer()
        
        print("Creating sample visualizations...")
        
        # Confusion Matrix
        visualizer.plot_confusion_matrix(y_true, y_pred)
        
        # ROC Curve
        visualizer.plot_roc_curve(y_true, y_scores)
        
        # Feature importance (sample)
        feature_names = [f'Feature_{i}' for i in range(20)]
        importance_values = np.random.random(20)
        visualizer.plot_feature_importance(feature_names, importance_values)
        
        # Dimensionality reduction
        visualizer.plot_dimensionality_reduction(features, y_true, method='tsne')
        
        print("Demo completed!")


if __name__ == '__main__':
    main()
