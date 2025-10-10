# -*- coding: utf-8 -*-
"""
Model training module for document forgery detection.

This module contains classes and functions to train machine learning models
for detecting document forgery using both traditional ML and deep learning approaches.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import joblib

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = keras = layers = models = optimizers = callbacks = None
    ImageDataGenerator = VGG16 = ResNet50 = EfficientNetB0 = None

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class DocumentForgeryDetector:
    """
    Main class for training document forgery detection models.
    """
    
    def __init__(self, model_type: str = 'cnn', random_state: int = 42):
        """
        Initialize the detector.
        
        Args:
            model_type: Type of model ('traditional_ml', 'cnn', 'transfer_learning')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.history = None
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        np.random.seed(random_state)
        if HAS_TENSORFLOW:
            tf.random.set_seed(random_state)
    
    def load_data(self, data_path: str, features_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data.
        
        Args:
            data_path: Path to image data directory or features CSV
            features_path: Path to features CSV (if using traditional ML)
            
        Returns:
            Tuple of (X, y) where X is features/images and y is labels
        """
        if self.model_type == 'traditional_ml':
            return self._load_feature_data(features_path or data_path)
        else:
            return self._load_image_data(data_path)
    
    def _load_feature_data(self, features_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-extracted features for traditional ML."""
        df = pd.read_csv(features_path)
        
        # Assume 'class' or 'label' column contains the target
        if 'class' in df.columns:
            target_col = 'class'
        elif 'label' in df.columns:
            target_col = 'label'
        else:
            raise ValueError("No 'class' or 'label' column found in features CSV")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in [target_col, 'filename', 'filepath']]
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Handle missing values
        X = np.nan_to_num(X)
        
        self.logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    
    def _load_image_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image data for deep learning models."""
        data_dir = Path(data_path)
        
        # Expected structure: data_path/train/{authentic,forged}/images
        train_dir = data_dir / 'train'
        if not train_dir.exists():
            train_dir = data_dir  # Assume direct class folders
        
        # Use ImageDataGenerator for loading
        datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            validation_split=0.2
        )
        
        # Load training data
        train_generator = datagen.flow_from_directory(
            str(train_dir),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        val_generator = datagen.flow_from_directory(
            str(train_dir),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Convert to numpy arrays (for smaller datasets)
        X_train, y_train = [], []
        X_val, y_val = [], []
        
        # Extract training data
        for i in range(len(train_generator)):
            batch_x, batch_y = train_generator[i]
            X_train.extend(batch_x)
            y_train.extend(batch_y)
        
        # Extract validation data
        for i in range(len(val_generator)):
            batch_x, batch_y = val_generator[i]
            X_val.extend(batch_x)
            y_val.extend(batch_y)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        # Combine for consistency
        X = np.vstack([X_train, X_val])
        y = np.vstack([y_train, y_val])
        
        self.logger.info(f"Loaded {X.shape[0]} images with shape {X.shape[1:]}")
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data for training.
        
        Args:
            X: Feature matrix or image array
            y: Target labels
            
        Returns:
            Preprocessed (X, y)
        """
        if self.model_type == 'traditional_ml':
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k='all')
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            return X_selected, y_encoded
        
        else:
            # For deep learning, images are already preprocessed
            # Convert labels to categorical if needed
            if y.ndim == 1:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                y_categorical = keras.utils.to_categorical(y_encoded)
                return X, y_categorical
            
            return X, y
    
    def build_traditional_model(self, model_name: str = 'random_forest') -> object:
        """
        Build a traditional machine learning model.
        
        Args:
            model_name: Name of the model to build
            
        Returns:
            Sklearn model instance
        """
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return models[model_name]
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                       num_classes: int = 2):
        """
        Build a CNN model for image classification.
        
        Args:
            input_shape: Shape of input images
            num_classes: Number of output classes
            
        Returns:
            Keras model instance
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_transfer_learning_model(self, base_model_name: str = 'vgg16',
                                    input_shape: Tuple[int, int, int] = (224, 224, 3),
                                    num_classes: int = 2):
        """
        Build a transfer learning model.
        
        Args:
            base_model_name: Name of the base model ('vgg16', 'resnet50', 'efficientnet')
            input_shape: Shape of input images
            num_classes: Number of output classes
            
        Returns:
            Keras model instance
        """
        # Load pre-trained base model
        if base_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        elif base_model_name == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classifier
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """
        Train the model.
        
        Args:
            X: Training features/images
            y: Training labels
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary containing training results
        """
        if self.model_type == 'traditional_ml':
            return self._train_traditional_model(X, y, **kwargs)
        else:
            return self._train_deep_model(X, y, **kwargs)
    
    def _train_traditional_model(self, X: np.ndarray, y: np.ndarray, 
                               model_name: str = 'random_forest',
                               use_grid_search: bool = False) -> Dict:
        """Train traditional ML model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Build model
        self.model = self.build_traditional_model(model_name)
        
        # Grid search for hyperparameter tuning
        if use_grid_search:
            self.model = self._perform_grid_search(self.model, X_train, y_train)
        
        # Train model
        self.logger.info("Training traditional ML model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred),
            'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
        }
        
        self.logger.info(f"Training completed. Test accuracy: {results['test_accuracy']:.4f}")
        return results
    
    def _train_deep_model(self, X: np.ndarray, y: np.ndarray,
                         epochs: int = 50, batch_size: int = 32,
                         validation_split: float = 0.2) -> Dict:
        """Train deep learning model."""
        # Build model
        if self.model_type == 'cnn':
            self.model = self.build_cnn_model(X.shape[1:], y.shape[1])
        elif self.model_type == 'transfer_learning':
            self.model = self.build_transfer_learning_model(input_shape=X.shape[1:], num_classes=y.shape[1])
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.logger.info("Training deep learning model...")
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate
        final_accuracy = max(self.history.history['val_accuracy'])
        final_loss = min(self.history.history['val_loss'])
        
        results = {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'history': self.history.history
        }
        
        self.logger.info(f"Training completed. Best validation accuracy: {final_accuracy:.4f}")
        return results
    
    def _perform_grid_search(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Perform grid search for hyperparameter tuning."""
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVC': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly']
            }
        }
        
        model_name = type(model).__name__
        if model_name in param_grids:
            self.logger.info(f"Performing grid search for {model_name}")
            grid_search = GridSearchCV(
                model, param_grids[model_name],
                cv=StratifiedKFold(n_splits=5),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def save_model(self, model_path: str, metadata: Optional[Dict] = None):
        """
        Save the trained model and associated components.
        
        Args:
            model_path: Path to save the model
            metadata: Additional metadata to save
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'traditional_ml':
            # Save sklearn model
            joblib.dump(self.model, model_path)
            
            # Save preprocessors
            if self.scaler:
                joblib.dump(self.scaler, model_path.replace('.joblib', '_scaler.joblib'))
            if self.feature_selector:
                joblib.dump(self.feature_selector, model_path.replace('.joblib', '_selector.joblib'))
            if self.label_encoder:
                joblib.dump(self.label_encoder, model_path.replace('.joblib', '_encoder.joblib'))
        
        else:
            # Save Keras model
            self.model.save(model_path)
            
            # Save preprocessors
            if self.label_encoder:
                joblib.dump(self.label_encoder, str(model_path).replace('.h5', '_encoder.joblib'))
        
        # Save metadata
        if metadata:
            metadata_path = str(model_path).replace('.joblib', '_metadata.json').replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history for deep learning models.
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            self.logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function for training models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train document forgery detection model')
    parser.add_argument('data_path', help='Path to training data')
    parser.add_argument('--model-type', choices=['traditional_ml', 'cnn', 'transfer_learning'],
                       default='cnn', help='Type of model to train')
    parser.add_argument('--model-name', default='random_forest',
                       help='Name of traditional ML model')
    parser.add_argument('--features-path', help='Path to features CSV (for traditional ML)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--output-dir', default='models/', help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize detector
    detector = DocumentForgeryDetector(model_type=args.model_type)
    
    # Load data
    X, y = detector.load_data(args.data_path, args.features_path)
    
    # Preprocess data
    X_processed, y_processed = detector.preprocess_data(X, y)
    
    # Train model
    if args.model_type == 'traditional_ml':
        results = detector.train_model(X_processed, y_processed, model_name=args.model_name)
    else:
        results = detector.train_model(X_processed, y_processed, 
                                     epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model_type == 'traditional_ml':
        model_path = output_dir / f"{args.model_name}_model.joblib"
    else:
        model_path = output_dir / f"{args.model_type}_model.h5"
    
    detector.save_model(str(model_path), metadata=results)
    
    # Plot training history for deep learning models
    if args.model_type != 'traditional_ml':
        plot_path = output_dir / f"{args.model_type}_training_history.png"
        detector.plot_training_history(str(plot_path))
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
