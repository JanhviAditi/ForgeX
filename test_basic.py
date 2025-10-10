#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify the basic system components without optional dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_system():
    """Test basic system components without OpenCV/TensorFlow."""
    print("🧪 Testing Document Forgery Detection - Basic Components")
    print("=" * 60)
    
    # Test 1: Config system
    try:
        from config import CONFIG, load_config_from_file, setup_directories, get_data_paths
        print("✅ Configuration system works")
        
        paths = get_data_paths(CONFIG)
        print(f"✅ Data paths configured: {len(paths)} directories")
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False
    
    # Test 2: Utilities
    try:
        from utils import setup_logging, get_system_info
        logger = setup_logging("INFO")
        sys_info = get_system_info()
        print(f"✅ Utilities work: Python {sys_info['python_version'].split()[0]}")
    except Exception as e:
        print(f"❌ Utilities failed: {e}")
        return False
    
    # Test 3: Basic feature extraction (without OpenCV)
    try:
        import numpy as np
        from features.build_features import DocumentFeatureExtractor
        
        # Create synthetic image
        synthetic_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        extractor = DocumentFeatureExtractor()
        
        # Test statistical features (don't need OpenCV)
        try:
            stats = extractor.extract_statistical_features(synthetic_image)
            print(f"✅ Statistical features work: {len(stats)} features")
        except ImportError:
            print("⚠️  Advanced features need OpenCV (optional)")
        
    except Exception as e:
        print(f"❌ Basic feature extraction failed: {e}")
        return False
    
    # Test 4: Traditional ML models (don't need TensorFlow)
    try:
        from models.train_model import DocumentForgeryDetector
        
        detector = DocumentForgeryDetector(model_type='traditional_ml')
        print("✅ Traditional ML models work")
        
    except ImportError as e:
        if "tensorflow" in str(e).lower():
            print("⚠️  Deep learning needs TensorFlow (optional)")
        else:
            print(f"❌ Traditional ML failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False
    
    # Test 5: CLI availability
    try:
        import cli
        print("✅ CLI interface available")
    except Exception as e:
        print(f"❌ CLI failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Basic system components are working!")
    print("\nℹ️  To use full functionality, install optional dependencies:")
    print("   pip install opencv-python tensorflow")
    print("\n🚀 Ready to use! Try: python cli.py info")
    
    return True

if __name__ == "__main__":
    success = test_basic_system()
    sys.exit(0 if success else 1)