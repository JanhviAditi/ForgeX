# Document Forgery Detection - Setup Guide

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended - Windows)
Double-click or run:
```bash
SETUP_PROJECT.bat
```

### Option 2: Manual Setup
```bash
# 1. Run setup script
python setup_project.py

# 2. Activate virtual environment
# Windows CMD:
activate.bat
# PowerShell:
.\activate.ps1
# Unix/Mac:
source venv/bin/activate

# 3. Verify setup
python verify_setup.py
```

## 📦 What Gets Installed

### Required Packages
- **Data Processing**: numpy, pandas
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Image Processing**: Pillow
- **Utilities**: joblib, pyyaml, click, tqdm

### Optional Packages (for full functionality)
- **opencv-python** - For advanced image processing and feature extraction
- **tensorflow** - For deep learning models (CNN, Transfer Learning)

To install optional packages after setup:
```bash
# Activate venv first
activate.bat

# Install optional packages
pip install opencv-python tensorflow
```

## 📁 Directory Structure Created

```
Document-Forgery-Detection/
├── venv/                              # Virtual environment (created by setup)
├── activate.bat                       # Windows CMD activation script
├── activate.ps1                       # PowerShell activation script
│
├── data/
│   ├── raw/
│   │   ├── authentic/                # ← Place real documents here
│   │   └── forged/                   # ← Place fake documents here
│   ├── processed/
│   │   ├── train/                    # Training data (70%)
│   │   ├── val/                      # Validation data (20%)
│   │   └── test/                     # Test data (10%)
│   ├── interim/                      # Intermediate processing
│   └── external/                     # External datasets
│
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   ├── features/                     # Feature extraction
│   ├── models/                       # Training and prediction
│   └── visualization/                # Plotting and analysis
│
├── models/                           # ← Trained models saved here
├── reports/
│   └── figures/                      # ← Visualizations saved here
├── logs/                             # Training logs
├── notebooks/                        # Jupyter notebooks
└── tests/                            # Unit tests
```

## ✅ Verification

After running setup, verify everything is working:

```bash
# Check project structure
python check_project.py

# Verify installation
python verify_setup.py

# Show project info
python cli.py info

# List all commands
python cli.py --help
```

## 📋 Next Steps After Setup

### 1. Prepare Your Dataset

You need document images organized as:
- **Authentic documents** in `data/raw/authentic/`
- **Forged documents** in `data/raw/forged/`

**Minimum recommended**: 100-200 images per class for testing
**For production**: 1000+ images per class

**Supported formats**: .jpg, .jpeg, .png, .bmp, .tiff

### 2. Preprocess Data

```bash
# Split data into train/val/test sets with augmentation
python cli.py preprocess-data data/raw/ --augment

# Without augmentation
python cli.py preprocess-data data/raw/
```

This creates:
- `data/processed/train/` (70% of data)
- `data/processed/val/` (20% of data)
- `data/processed/test/` (10% of data)

### 3. Train Models

#### Option A: Traditional Machine Learning (Faster, requires less data)

```bash
# Extract features
python cli.py extract-features data/processed/train/ --output-file data/processed/features.csv

# Train Random Forest
python cli.py train-model data/processed/ \
    --model-type traditional_ml \
    --model-name random_forest \
    --features-path data/processed/features.csv

# Train SVM
python cli.py train-model data/processed/ \
    --model-type traditional_ml \
    --model-name svm \
    --features-path data/processed/features.csv
```

#### Option B: Deep Learning CNN (Requires TensorFlow)

```bash
# Train custom CNN
python cli.py train-model data/processed/ \
    --model-type cnn \
    --epochs 50 \
    --batch-size 32

# With experiment tracking
python cli.py train-model data/processed/ \
    --model-type cnn \
    --epochs 50 \
    --experiment-name my_cnn_experiment
```

#### Option C: Transfer Learning (Best accuracy, requires TensorFlow)

```bash
# Train with pre-trained VGG16
python cli.py train-model data/processed/ \
    --model-type transfer_learning \
    --epochs 30 \
    --batch-size 16
```

### 4. Make Predictions

```bash
# Predict single image
python cli.py predict models/random_forest_model.joblib path/to/test/image.jpg

# Predict entire directory with visualizations
python cli.py predict models/cnn_model.h5 data/processed/test/ --visualize

# Batch prediction with results saved to CSV
python cli.py predict models/model.joblib data/test/ --output results.csv
```

### 5. View Results

Check these directories:
- **Models**: `models/` directory
- **Reports**: `reports/` directory  
- **Visualizations**: `reports/figures/` directory
- **Logs**: `logs/` directory

## 🔧 Troubleshooting

### Virtual Environment Not Activating (PowerShell)

```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.\activate.ps1
```

### Missing Dependencies

```bash
# Activate venv first
activate.bat

# Reinstall all dependencies
pip install -r requirements.txt

# Install specific package
pip install package-name
```

### OpenCV/TensorFlow Not Working

```bash
# Activate venv
activate.bat

# Install optional packages
pip install opencv-python
pip install tensorflow

# Verify
python -c "import cv2; print('OpenCV OK')"
python -c "import tensorflow; print('TensorFlow OK')"
```

### Project Structure Issues

```bash
# Re-run setup
python setup_project.py

# Check structure
python check_project.py
```

### Cannot Find Modules

Make sure virtual environment is active:
```bash
# Check if venv is active (you should see (venv) in prompt)
activate.bat

# Or check Python path
python -c "import sys; print(sys.prefix)"
```

## 🧪 Testing the Installation

### Quick Test (No dataset needed)

```bash
# Activate venv
activate.bat

# Run basic tests
python test_basic.py

# Show project info
python cli.py info
```

### Full System Test

```bash
# Run comprehensive tests
python test_system.py

# Demo the complete system
python demo.py
```

## 📚 Documentation

### Command Reference

```bash
# General help
python cli.py --help

# Command-specific help
python cli.py <command> --help

# Examples:
python cli.py preprocess-data --help
python cli.py train-model --help
python cli.py predict --help
```

### Python API Usage

```python
# Train a model programmatically
from src.models.train_model import DocumentForgeryDetector

detector = DocumentForgeryDetector(model_type='traditional_ml')
X, y = detector.load_data('data/processed/features.csv')
results = detector.train_model(X, y, model_name='random_forest')

# Make predictions
from src.models.predict_model import DocumentForgeryPredictor

predictor = DocumentForgeryPredictor('models/model.joblib')
result = predictor.predict_single_image('test.jpg')
print(f"Prediction: {result['prediction']} ({result['confidence']:.2%})")
```

### Jupyter Notebooks

```bash
# Activate venv
activate.bat

# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook

# Open notebooks in notebooks/ directory
```

## 🎯 Example Workflow

Complete example from start to finish:

```bash
# 1. Setup (one-time)
SETUP_PROJECT.bat

# 2. Activate environment
activate.bat

# 3. Add your dataset
# - Copy images to data/raw/authentic/ and data/raw/forged/

# 4. Preprocess
python cli.py preprocess-data data/raw/ --augment

# 5. Train model
python cli.py extract-features data/processed/train/
python cli.py train-model data/processed/ --model-type traditional_ml --features-path data/processed/features.csv

# 6. Predict
python cli.py predict models/random_forest_model.joblib data/processed/test/ --visualize

# 7. View results in reports/figures/
```

## 💡 Tips

1. **Start Small**: Test with 100-200 images before scaling up
2. **Try Traditional ML First**: Faster training, easier to debug
3. **Use Transfer Learning**: If you have limited data
4. **Monitor Training**: Check logs/ and reports/ directories
5. **Experiment Tracking**: Use --experiment-name flag for better organization

## 📞 Getting Help

- Check project structure: `python check_project.py`
- Verify installation: `python verify_setup.py`
- Show CLI help: `python cli.py --help`
- Run tests: `python test_basic.py`

## 🔗 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Project README](README.md)

---

**Ready to detect document forgeries! 🚀**
