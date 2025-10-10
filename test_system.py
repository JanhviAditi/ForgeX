#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to validate the Document Forgery Detection system.

This script runs basic tests to ensure all components are working correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        from config import CONFIG, load_config_from_file
        print("  ✅ Config module imported successfully")
    except Exception as e:
        print(f"  ❌ Config import failed: {e}")
        return False
    
    try:
        from utils import setup_logging, timing_decorator
        print("  ✅ Utils module imported successfully")
    except Exception as e:
        print(f"  ❌ Utils import failed: {e}")
        return False
    
    try:
        from data.make_dataset import create_dataset_structure
        print("  ✅ Data module imported successfully")
    except Exception as e:
        print(f"  ❌ Data import failed: {e}")
        return False
    
    try:
        from features.build_features import DocumentFeatureExtractor
        print("  ✅ Features module imported successfully")
    except Exception as e:
        print(f"  ❌ Features import failed: {e}")
        return False
    
    try:
        from models.train_model import DocumentForgeryDetector
        print("  ✅ Training module imported successfully")
    except Exception as e:
        print(f"  ❌ Training import failed: {e}")
        return False
    
    try:
        from models.predict_model import DocumentForgeryPredictor
        print("  ✅ Prediction module imported successfully")
    except Exception as e:
        print(f"  ❌ Prediction import failed: {e}")
        return False
    
    try:
        from visualization.visualize import DocumentForgeryVisualizer
        print("  ✅ Visualization module imported successfully")
    except Exception as e:
        print(f"  ❌ Visualization import failed: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import CONFIG, setup_directories, get_data_paths
        
        # Test config loading
        print(f"  ✅ Default config loaded: project root = {CONFIG.project_root}")
        
        # Test directory setup
        paths = get_data_paths(CONFIG)
        print(f"  ✅ Data paths configured: {len(paths)} paths")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction with synthetic data."""
    print("\n🧪 Testing feature extraction...")
    
    try:
        import numpy as np
        from features.build_features import DocumentFeatureExtractor
        
        # Create synthetic image
        synthetic_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        
        extractor = DocumentFeatureExtractor()
        features = extractor.extract_features(synthetic_image)
        
        print(f"  ✅ Extracted {len(features)} features from synthetic image")
        print(f"  ✅ Feature types: {list(features.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature extraction test failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\n🧪 Testing model initialization...")
    
    try:
        from models.train_model import DocumentForgeryDetector
        
        # Test traditional ML model
        detector_ml = DocumentForgeryDetector(model_type='traditional_ml')
        print("  ✅ Traditional ML detector initialized")
        
        # Test CNN model (without TensorFlow dependency)
        try:
            detector_cnn = DocumentForgeryDetector(model_type='cnn')
            print("  ✅ CNN detector initialized")
        except ImportError:
            print("  ⚠️  CNN detector requires TensorFlow (optional)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model initialization test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🧪 Testing utilities...")
    
    try:
        from utils import setup_logging, get_system_info
        
        # Test logging setup
        logger = setup_logging("INFO")
        print("  ✅ Logging setup successful")
        
        # Test system info
        sys_info = get_system_info()
        print(f"  ✅ System info retrieved: Python {sys_info['python_version'].split()[0]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utilities test failed: {e}")
        return False


def check_dependencies():
    """Check if key dependencies are available."""
    print("\n🧪 Checking dependencies...")
    
    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas', 
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }
    
    optional_dependencies = {
        'cv2': 'OpenCV',
        'tensorflow': 'TensorFlow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
    }
    
    all_available = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (required)")
            all_available = False
    
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name} (optional)")
        except ImportError:
            print(f"  ⚠️  {name} (optional, not installed)")
    
    return all_available


def main():
    """Run all tests."""
    print("🚀 Document Forgery Detection System Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration), 
        ("Feature Extraction Tests", test_feature_extraction),
        ("Model Initialization Tests", test_model_initialization),
        ("Utility Tests", test_utilities),
        ("Dependency Check", check_dependencies),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())