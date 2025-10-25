"""
Document Forgery Detection - Model Testing Script
===================================================
This script allows you to test the trained model on:
1. Single image
2. Multiple images
3. Entire folder
4. Interactive testing
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
import sys
import argparse
sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

class ForgeryDetector:
    """Class to detect document forgeries using the trained model."""
    
    def __init__(self, model_path='models/final_ensemble_model.joblib'):
        """Load the trained model and preprocessing components."""
        print("🔧 Loading model and preprocessing components...")
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load('models/final_scaler.joblib')
            self.selector = joblib.load('models/final_feature_selector.joblib')
            self.extractor = DocumentFeatureExtractor()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure you have trained the model first!")
            sys.exit(1)
    
    def extract_enhanced_features(self, img_path):
        """Extract enhanced features from an image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        # Get base features
        features = self.extractor.extract_all_features(str(img_path))
        if not features:
            return None
        
        # Add enhanced features (same as training)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # DCT features
        dct = cv2.dct(np.float32(gray))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        features['dct_max'] = np.max(np.abs(dct))
        
        # Compression artifacts
        _, compressed = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, decompressed)
        features['compression_diff_mean'] = np.mean(diff)
        features['compression_diff_std'] = np.std(diff)
        
        # Noise analysis
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        features['noise_level'] = np.mean(np.abs(high_freq))
        features['noise_variance'] = np.var(high_freq)
        
        # Edge consistency
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        features['edge_consistency'] = np.sum(dilated) / (gray.shape[0] * gray.shape[1])
        
        # Contrast variation
        block_size = 32
        h, w = gray.shape
        contrasts = []
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                contrasts.append(np.std(block))
        features['contrast_variation'] = np.std(contrasts) if contrasts else 0
        features['contrast_mean'] = np.mean(contrasts) if contrasts else 0
        
        # Color histogram entropy
        hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
        
        def entropy(hist):
            hist = hist / (hist.sum() + 1e-7)
            return -np.sum(hist * np.log2(hist + 1e-7))
        
        features['color_hist_entropy'] = (entropy(hist_b) + entropy(hist_g) + entropy(hist_r)) / 3
        
        return list(features.values())
    
    def predict(self, img_path):
        """
        Predict if an image is authentic or forged.
        
        Returns:
            dict: {
                'prediction': 'Authentic' or 'Forged',
                'confidence': float (0-100),
                'probabilities': {'Authentic': float, 'Forged': float}
            }
        """
        # Extract features
        features = self.extract_enhanced_features(img_path)
        if features is None:
            return None
        
        # Preprocess
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        result = {
            'prediction': 'Authentic' if prediction == 0 else 'Forged',
            'confidence': max(probabilities) * 100,
            'probabilities': {
                'Authentic': probabilities[0] * 100,
                'Forged': probabilities[1] * 100
            }
        }
        
        return result
    
    def display_result(self, img_path, result):
        """Display prediction result with colored output."""
        filename = Path(img_path).name
        
        if result['prediction'] == 'Authentic':
            emoji = "✅"
            color = "\033[92m"  # Green
        else:
            emoji = "⚠️"
            color = "\033[91m"  # Red
        
        reset = "\033[0m"
        
        print(f"\n{'='*70}")
        print(f"📄 File: {filename}")
        print(f"{'='*70}")
        print(f"{color}{emoji} Prediction: {result['prediction']}{reset}")
        print(f"🎯 Confidence: {result['confidence']:.2f}%")
        print(f"\n📊 Probabilities:")
        print(f"   ✅ Authentic: {result['probabilities']['Authentic']:.2f}%")
        print(f"   ⚠️  Forged:    {result['probabilities']['Forged']:.2f}%")
        print(f"{'='*70}")


def test_single_image(detector, img_path):
    """Test a single image."""
    print(f"\n🔍 Testing image: {img_path}")
    
    if not Path(img_path).exists():
        print(f"❌ Error: File not found - {img_path}")
        return
    
    result = detector.predict(img_path)
    if result:
        detector.display_result(img_path, result)
    else:
        print(f"❌ Error: Could not process image")


def test_folder(detector, folder_path, limit=None):
    """Test all images in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Error: Folder not found - {folder_path}")
        return
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(folder.glob(ext))
    
    if not images:
        print(f"❌ No images found in {folder_path}")
        return
    
    # Limit if specified
    if limit:
        images = images[:limit]
    
    print(f"\n🔍 Testing {len(images)} images from: {folder_path}")
    print("="*70)
    
    results = {
        'authentic': 0,
        'forged': 0,
        'errors': 0
    }
    
    for img_path in images:
        result = detector.predict(img_path)
        if result:
            if result['prediction'] == 'Authentic':
                results['authentic'] += 1
                status = "✅"
            else:
                results['forged'] += 1
                status = "⚠️"
            
            print(f"{status} {img_path.name:50s} -> {result['prediction']:12s} ({result['confidence']:.1f}%)")
        else:
            results['errors'] += 1
            print(f"❌ {img_path.name:50s} -> Error processing")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"✅ Authentic: {results['authentic']}")
    print(f"⚠️  Forged:    {results['forged']}")
    print(f"❌ Errors:    {results['errors']}")
    print(f"📁 Total:     {len(images)}")
    print("="*70)


def interactive_mode(detector):
    """Interactive testing mode."""
    print("\n" + "="*70)
    print("🎮 INTERACTIVE TESTING MODE")
    print("="*70)
    print("Enter image path to test (or 'quit' to exit)")
    
    while True:
        try:
            img_path = input("\n📄 Image path: ").strip()
            
            if img_path.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not img_path:
                continue
            
            # Remove quotes if present
            img_path = img_path.strip('"').strip("'")
            
            test_single_image(detector, img_path)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Test Document Forgery Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single image
  python test_model.py --image path/to/image.jpg
  
  # Test all images in a folder
  python test_model.py --folder data/test_images
  
  # Test first 10 images in a folder
  python test_model.py --folder data/test_images --limit 10
  
  # Interactive mode
  python test_model.py --interactive
  
  # Use different model
  python test_model.py --image test.jpg --model models/random_forest_optimized.joblib
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to a single image to test')
    parser.add_argument('--folder', type=str, help='Path to folder containing images')
    parser.add_argument('--limit', type=int, help='Limit number of images to test from folder')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model', type=str, default='models/final_ensemble_model.joblib', 
                        help='Path to model file (default: final_ensemble_model.joblib)')
    
    args = parser.parse_args()
    
    # Load model
    print("="*70)
    print("🚀 DOCUMENT FORGERY DETECTION - MODEL TESTING")
    print("="*70)
    
    detector = ForgeryDetector(args.model)
    
    # Execute based on arguments
    if args.image:
        test_single_image(detector, args.image)
    elif args.folder:
        test_folder(detector, args.folder, args.limit)
    elif args.interactive:
        interactive_mode(detector)
    else:
        # Default: interactive mode
        print("\n💡 No arguments provided. Starting interactive mode...")
        print("   (Use --help to see all options)")
        interactive_mode(detector)


if __name__ == '__main__':
    main()
