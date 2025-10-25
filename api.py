"""
ForgeX - REST API Backend
=========================
Flask-based REST API for document forgery detection.
Provides endpoints for single and batch document analysis.
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
import sys
import io
import base64
from datetime import datetime
import json

sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

class ForgeryDetector:
    """Forgery detection backend"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selector = None
        self.extractor = DocumentFeatureExtractor()
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load trained models"""
        try:
            self.model = joblib.load('models/final_ensemble_model.joblib')
            self.scaler = joblib.load('models/final_scaler.joblib')
            self.selector = joblib.load('models/final_feature_selector.joblib')
            self.model_loaded = True
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def extract_enhanced_features(self, img):
        """Extract features from image array"""
        # Save temporarily
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Get base features
        features = self.extractor.extract_all_features(temp_path)
        if not features:
            return None
        
        # Enhanced features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        dct = cv2.dct(np.float32(gray))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        features['dct_max'] = np.max(np.abs(dct))
        
        _, compressed = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        decompressed_rgb = cv2.cvtColor(decompressed, cv2.COLOR_BGR2RGB)
        diff = cv2.absdiff(img, decompressed_rgb)
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
        
        Path(temp_path).unlink(missing_ok=True)
        return list(features.values())
    
    def predict(self, image_array):
        """Predict document authenticity"""
        if not self.model_loaded:
            return None
        
        features = self.extract_enhanced_features(image_array)
        if features is None:
            return None
        
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        return {
            'prediction': 'authentic' if prediction == 0 else 'forged',
            'confidence': float(max(probabilities) * 100),
            'probabilities': {
                'authentic': float(probabilities[0] * 100),
                'forged': float(probabilities[1] * 100)
            },
            'timestamp': datetime.now().isoformat()
        }

# Initialize detector
detector = ForgeryDetector()

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        'name': 'ForgeX API',
        'version': '1.0.0',
        'description': 'Document Forgery Detection API',
        'accuracy': '94.94%',
        'endpoints': {
            '/api/predict': 'POST - Single document analysis',
            '/api/batch': 'POST - Batch document analysis',
            '/api/health': 'GET - Health check',
            '/api/stats': 'GET - Model statistics'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if detector.model_loaded else 'model not loaded',
        'model_loaded': detector.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    """Model statistics endpoint"""
    return jsonify({
        'model': {
            'type': 'Ensemble (Random Forest + Gradient Boosting + SVM + LR)',
            'accuracy': 94.94,
            'precision': 97.24,
            'recall': 92.50,
            'roc_auc': 98.22,
            'features': 32
        },
        'training': {
            'total_images': 8000,
            'authentic': 4000,
            'forged': 4000,
            'test_size': 1600
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single document prediction endpoint"""
    if not detector.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read image
        image = Image.open(file.stream)
        img_array = np.array(image)
        
        # Predict
        result = detector.predict(img_array)
        
        if result is None:
            return jsonify({'error': 'Error processing image'}), 500
        
        return jsonify({
            'success': True,
            'result': result,
            'filename': file.filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    """Batch document prediction endpoint"""
    if not detector.model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        try:
            image = Image.open(file.stream)
            img_array = np.array(image)
            result = detector.predict(img_array)
            
            if result:
                results.append({
                    'filename': file.filename,
                    'result': result
                })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    # Summary statistics
    successful = [r for r in results if 'result' in r]
    authentic_count = sum(1 for r in successful if r['result']['prediction'] == 'authentic')
    forged_count = sum(1 for r in successful if r['result']['prediction'] == 'forged')
    
    return jsonify({
        'success': True,
        'total_processed': len(results),
        'successful': len(successful),
        'failed': len(results) - len(successful),
        'summary': {
            'authentic': authentic_count,
            'forged': forged_count
        },
        'results': results
    })

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*70)
    print("🚀 ForgeX API Server Starting...")
    print("="*70)
    print(f"Model Status: {'✅ Loaded' if detector.model_loaded else '❌ Not Loaded'}")
    print("\nEndpoints:")
    print("  - GET  /              - API information")
    print("  - GET  /api/health    - Health check")
    print("  - GET  /api/stats     - Model statistics")
    print("  - POST /api/predict   - Single prediction")
    print("  - POST /api/batch     - Batch prediction")
    print("\nStarting server on http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
