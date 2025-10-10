# -*- coding: utf-8 -*-
"""
Prediction module for document forgery detection.

This module contains classes and functions to make predictions on new document images
using trained models for forgery detection.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import joblib
from PIL import Image

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = keras = None

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.build_features import DocumentFeatureExtractor


class DocumentForgeryPredictor:
    """
    Class for making predictions on document images using trained models.
    """
    
    def __init__(self, model_path: str, model_type: str = 'auto'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model ('traditional_ml', 'cnn', 'transfer_learning', 'auto')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.feature_extractor = None
        self.metadata = None
        self.logger = logging.getLogger(__name__)
        
        # Auto-detect model type if not specified
        if model_type == 'auto':
            self.model_type = self._detect_model_type()
        
        # Load model and components
        self._load_model()
        self._load_components()
    
    def _detect_model_type(self) -> str:
        """Auto-detect the model type based on file extension."""
        if self.model_path.suffix == '.joblib':
            return 'traditional_ml'
        elif self.model_path.suffix == '.h5':
            if 'transfer' in str(self.model_path).lower():
                return 'transfer_learning'
            else:
                return 'cnn'
        else:
            raise ValueError(f"Unknown model file extension: {self.model_path.suffix}")
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if self.model_type == 'traditional_ml':
            self.model = joblib.load(self.model_path)
            self.feature_extractor = DocumentFeatureExtractor()
        else:
            self.model = keras.models.load_model(self.model_path)
        
        self.logger.info(f"Loaded {self.model_type} model from {self.model_path}")
    
    def _load_components(self):
        """Load associated components like scalers and encoders."""
        base_path = str(self.model_path).replace(self.model_path.suffix, '')
        
        # Load scaler
        scaler_path = base_path + '_scaler.joblib'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Loaded scaler")
        
        # Load feature selector
        selector_path = base_path + '_selector.joblib'
        if os.path.exists(selector_path):
            self.feature_selector = joblib.load(selector_path)
            self.logger.info("Loaded feature selector")
        
        # Load label encoder
        encoder_path = base_path + '_encoder.joblib'
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
            self.logger.info("Loaded label encoder")
        
        # Load metadata
        metadata_path = base_path + '_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.logger.info("Loaded metadata")
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Make a prediction on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            if self.model_type == 'traditional_ml':
                return self._predict_traditional_ml(image_path)
            else:
                return self._predict_deep_learning(image_path)
                
        except Exception as e:
            self.logger.error(f"Error predicting image {image_path}: {str(e)}")
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }
    
    def _predict_traditional_ml(self, image_path: str) -> Dict:
        """Make prediction using traditional ML model."""
        # Extract features
        features = self.feature_extractor.extract_all_features(image_path)
        
        if not features:
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': 'Could not extract features'
            }
        
        # Convert to array
        feature_names = list(features.keys())
        feature_values = np.array([list(features.values())])
        
        # Handle missing values
        feature_values = np.nan_to_num(feature_values)
        
        # Apply preprocessing
        if self.scaler:
            feature_values = self.scaler.transform(feature_values)
        
        if self.feature_selector:
            feature_values = self.feature_selector.transform(feature_values)
        
        # Make prediction
        prediction_encoded = self.model.predict(feature_values)[0]
        probabilities = self.model.predict_proba(feature_values)[0]
        
        # Decode prediction
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            class_names = self.label_encoder.classes_
        else:
            prediction = prediction_encoded
            class_names = ['authentic', 'forged']  # Default class names
        
        # Get confidence (max probability)
        confidence = np.max(probabilities)
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = float(probabilities[i])
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'feature_count': len(features)
        }
    
    def _predict_deep_learning(self, image_path: str) -> Dict:
        """Make prediction using deep learning model."""
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path)
        
        if image is None:
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': 'Could not load image'
            }
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_batch, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])
        
        # Decode prediction
        if self.label_encoder:
            class_names = self.label_encoder.classes_
            prediction = class_names[predicted_class_idx]
        else:
            class_names = ['authentic', 'forged']  # Default class names
            prediction = class_names[predicted_class_idx]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = float(probabilities[i])
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': prob_dict,
            'image_shape': image.shape
        }
    
    def _load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Load and preprocess image for deep learning models."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize
            image = image.resize(target_size)
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing (only used for deep learning models)
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        if self.model_type == 'traditional_ml':
            # Process individually for traditional ML
            for image_path in image_paths:
                result = self.predict_single_image(image_path)
                result['image_path'] = image_path
                results.append(result)
        
        else:
            # Process in batches for deep learning
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_results = self._predict_batch_deep_learning(batch_paths)
                results.extend(batch_results)
        
        return results
    
    def _predict_batch_deep_learning(self, image_paths: List[str]) -> List[Dict]:
        """Process a batch of images with deep learning model."""
        # Load and preprocess all images
        images = []
        valid_paths = []
        
        for path in image_paths:
            image = self._load_and_preprocess_image(path)
            if image is not None:
                images.append(image)
                valid_paths.append(path)
        
        if not images:
            return []
        
        # Convert to batch
        images_batch = np.array(images)
        
        # Make predictions
        predictions = self.model.predict(images_batch, verbose=0)
        
        # Process results
        results = []
        for i, (path, prediction) in enumerate(zip(valid_paths, predictions)):
            # Get predicted class
            predicted_class_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_class_idx])
            
            # Decode prediction
            if self.label_encoder:
                class_names = self.label_encoder.classes_
                predicted_class = class_names[predicted_class_idx]
            else:
                class_names = ['authentic', 'forged']
                predicted_class = class_names[predicted_class_idx]
            
            # Create probability dictionary
            prob_dict = {}
            for j, class_name in enumerate(class_names):
                prob_dict[class_name] = float(prediction[j])
            
            results.append({
                'image_path': path,
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': prob_dict,
                'image_shape': images[i].shape
            })
        
        return results
    
    def predict_directory(self, directory_path: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            output_file: Optional path to save results CSV
            
        Returns:
            DataFrame containing prediction results
        """
        directory = Path(directory_path)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(directory.glob(f'**/*{ext}'))
            image_paths.extend(directory.glob(f'**/*{ext.upper()}'))
        
        if not image_paths:
            self.logger.warning(f"No image files found in {directory_path}")
            return pd.DataFrame()
        
        # Convert to strings
        image_paths = [str(path) for path in image_paths]
        
        self.logger.info(f"Processing {len(image_paths)} images...")
        
        # Make predictions
        results = self.predict_batch(image_paths)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add additional information
        if not df.empty:
            df['filename'] = df['image_path'].apply(lambda x: Path(x).name)
            df['directory'] = directory_path
        
        # Save results if requested
        if output_file and not df.empty:
            df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to {output_file}")
        
        return df
    
    def explain_prediction(self, image_path: str) -> Dict:
        """
        Provide explanation for a prediction (for traditional ML models).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing explanation information
        """
        if self.model_type != 'traditional_ml':
            return {'error': 'Explanations only available for traditional ML models'}
        
        try:
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            else:
                feature_importance = None
            
            # Extract features for this image
            features = self.feature_extractor.extract_all_features(image_path)
            
            if not features:
                return {'error': 'Could not extract features'}
            
            # Get prediction
            prediction_result = self.predict_single_image(image_path)
            
            # Create explanation
            explanation = {
                'prediction': prediction_result,
                'feature_count': len(features),
                'extracted_features': features
            }
            
            if feature_importance is not None:
                # Get top important features
                feature_names = list(features.keys())
                if len(feature_importance) == len(feature_names):
                    importance_dict = dict(zip(feature_names, feature_importance))
                    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                    explanation['top_important_features'] = top_features
            
            return explanation
            
        except Exception as e:
            return {'error': f'Error generating explanation: {str(e)}'}


def main():
    """Main function for making predictions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions on document images')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('input_path', help='Path to image file or directory')
    parser.add_argument('--model-type', choices=['traditional_ml', 'cnn', 'transfer_learning', 'auto'],
                       default='auto', help='Type of model')
    parser.add_argument('--output', '-o', help='Output CSV file for results')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--explain', action='store_true', help='Provide explanation (traditional ML only)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize predictor
    predictor = DocumentForgeryPredictor(args.model_path, args.model_type)
    
    input_path = Path(args.input_path)
    
    if input_path.is_file():
        # Single image prediction
        result = predictor.predict_single_image(str(input_path))
        
        print(f"\nPrediction for {input_path.name}:")
        print(f"Class: {result.get('prediction', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.4f}")
        
        if 'probabilities' in result:
            print("\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")
        
        # Provide explanation if requested
        if args.explain:
            explanation = predictor.explain_prediction(str(input_path))
            if 'error' not in explanation:
                print(f"\nExtracted {explanation['feature_count']} features")
                if 'top_important_features' in explanation:
                    print("\nTop 5 important features:")
                    for feature, importance in explanation['top_important_features'][:5]:
                        print(f"  {feature}: {importance:.4f}")
    
    elif input_path.is_dir():
        # Directory prediction
        df = predictor.predict_directory(str(input_path), args.output)
        
        if not df.empty:
            print(f"\nProcessed {len(df)} images")
            print(f"Predictions summary:")
            print(df['prediction'].value_counts().to_string())
            
            print(f"\nConfidence statistics:")
            print(f"Mean confidence: {df['confidence'].mean():.4f}")
            print(f"Min confidence: {df['confidence'].min():.4f}")
            print(f"Max confidence: {df['confidence'].max():.4f}")
            
            if args.output:
                print(f"\nDetailed results saved to {args.output}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == '__main__':
    main()
