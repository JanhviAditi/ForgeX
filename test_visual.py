"""
Visual Testing Demo - Shows predictions with image display
===========================================================
This script tests images and displays them with prediction overlays.
Requires: matplotlib for visualization
"""

import cv2
import numpy as np
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

class VisualForgeryDetector:
    """Visual forgery detector with image display."""
    
    def __init__(self, model_path='models/final_ensemble_model.joblib'):
        """Load the trained model."""
        print("🔧 Loading model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/final_scaler.joblib')
        self.selector = joblib.load('models/final_feature_selector.joblib')
        self.extractor = DocumentFeatureExtractor()
        print("✅ Model loaded!")
    
    def extract_enhanced_features(self, img_path):
        """Extract enhanced features from an image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        
        features = self.extractor.extract_all_features(str(img_path))
        if not features:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced features (matching training)
        dct = cv2.dct(np.float32(gray))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        features['dct_max'] = np.max(np.abs(dct))
        
        _, compressed = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(img, decompressed)
        features['compression_diff_mean'] = np.mean(diff)
        features['compression_diff_std'] = np.std(diff)
        
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        features['noise_level'] = np.mean(np.abs(high_freq))
        features['noise_variance'] = np.var(high_freq)
        
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        features['edge_consistency'] = np.sum(dilated) / (gray.shape[0] * gray.shape[1])
        
        block_size = 32
        h, w = gray.shape
        contrasts = []
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                contrasts.append(np.std(block))
        features['contrast_variation'] = np.std(contrasts) if contrasts else 0
        features['contrast_mean'] = np.mean(contrasts) if contrasts else 0
        
        hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
        
        def entropy(hist):
            hist = hist / (hist.sum() + 1e-7)
            return -np.sum(hist * np.log2(hist + 1e-7))
        
        features['color_hist_entropy'] = (entropy(hist_b) + entropy(hist_g) + entropy(hist_r)) / 3
        
        return list(features.values())
    
    def predict(self, img_path):
        """Predict if image is authentic or forged."""
        features = self.extract_enhanced_features(img_path)
        if features is None:
            return None
        
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        return {
            'prediction': 'Authentic' if prediction == 0 else 'Forged',
            'confidence': max(probabilities) * 100,
            'probabilities': {
                'Authentic': probabilities[0] * 100,
                'Forged': probabilities[1] * 100
            }
        }
    
    def visualize_prediction(self, img_path, result):
        """Display image with prediction overlay."""
        # Read image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img_rgb)
        ax.axis('off')
        
        # Determine colors
        if result['prediction'] == 'Authentic':
            color = 'green'
            emoji = '✅'
        else:
            color = 'red'
            emoji = '⚠️'
        
        # Add prediction box
        filename = Path(img_path).name
        text = f"{emoji} {result['prediction']}\nConfidence: {result['confidence']:.1f}%"
        
        # Add semi-transparent box at the top
        props = dict(boxstyle='round', facecolor=color, alpha=0.8)
        ax.text(0.5, 0.05, text, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', horizontalalignment='center',
                bbox=props, color='white', weight='bold')
        
        # Add filename at bottom
        ax.text(0.5, 0.98, filename, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def test_multiple_images(self, image_paths, save_output=False):
        """Test and visualize multiple images in a grid."""
        n_images = len(image_paths)
        
        # Calculate grid size
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx, img_path in enumerate(image_paths):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # Read and predict
            img = cv2.imread(str(img_path))
            if img is None:
                ax.text(0.5, 0.5, f'Error loading\n{Path(img_path).name}', 
                       ha='center', va='center')
                ax.axis('off')
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.predict(img_path)
            
            if result is None:
                ax.text(0.5, 0.5, f'Error processing\n{Path(img_path).name}', 
                       ha='center', va='center')
                ax.axis('off')
                continue
            
            # Display
            ax.imshow(img_rgb)
            ax.axis('off')
            
            # Colors
            color = 'green' if result['prediction'] == 'Authentic' else 'red'
            emoji = '✅' if result['prediction'] == 'Authentic' else '⚠️'
            
            # Title
            title = f"{emoji} {result['prediction']} ({result['confidence']:.1f}%)"
            ax.set_title(title, fontsize=12, color=color, weight='bold', pad=10)
            
            # Filename
            filename = Path(img_path).name
            ax.text(0.5, -0.05, filename, transform=ax.transAxes, fontsize=8,
                   ha='center', va='top', style='italic')
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_output:
            output_path = 'reports/figures/test_results.png'
            Path('reports/figures').mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Results saved to: {output_path}")
        
        return fig


def main():
    """Demo: Test some sample images."""
    print("="*70)
    print("🎨 VISUAL FORGERY DETECTION DEMO")
    print("="*70)
    
    detector = VisualForgeryDetector()
    
    # Get sample images
    authentic_dir = Path('data/consolidated/authentic')
    forged_dir = Path('data/consolidated/forged')
    
    # Sample authentic images
    authentic_samples = list(authentic_dir.glob('*.jpg'))[:3]
    # Sample forged images
    forged_samples = list(forged_dir.glob('*.jpg'))[:3]
    
    if not authentic_samples and not forged_samples:
        print("❌ No sample images found!")
        print("Please ensure data/consolidated/ contains images.")
        return
    
    # Combine samples
    sample_images = authentic_samples + forged_samples
    
    print(f"\n🔍 Testing {len(sample_images)} sample images...")
    print(f"   - {len(authentic_samples)} authentic")
    print(f"   - {len(forged_samples)} forged")
    
    # Test and visualize
    fig = detector.test_multiple_images(sample_images, save_output=True)
    
    print("\n✅ Visualization complete!")
    print("📊 Displaying results...")
    plt.show()


if __name__ == '__main__':
    main()
