# Document Forgery Detection - Project Structure

## 📁 Essential Files and Directories

This document describes the minimal, production-ready structure of the project.

### Root Directory Files

```
Document-Forgery-Detection/
├── cli.py                  # Command-line interface (main entry point)
├── setup.py                # Package installation configuration
├── requirements.txt        # Python dependencies
├── config.yaml            # Default configuration
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── .gitignore             # Git ignore rules
├── SETUP_PROJECT.bat      # Windows setup script
├── QUICK_START.bat        # Quick start script
└── SETUP_GUIDE.md         # Setup instructions
```

### Source Code (`src/`)

```
src/
├── __init__.py            # Makes src a Python package
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── data/
│   ├── __init__.py
│   └── make_dataset.py    # Data loading and preprocessing
├── features/
│   ├── __init__.py
│   └── build_features.py  # Feature extraction
├── models/
│   ├── __init__.py
│   ├── train_model.py     # Model training
│   └── predict_model.py   # Prediction/inference
└── visualization/
    ├── __init__.py
    └── visualize.py       # Plotting and visualization
```

### Data Directories (`data/`)

```
data/
├── raw/                   # Original, immutable data
│   ├── authentic/         # Real documents
│   └── forged/           # Fake documents
├── processed/            # Preprocessed data ready for modeling
│   ├── train/
│   ├── val/
│   └── test/
├── interim/              # Intermediate transformations
└── external/             # External data sources
```

### Models Directory (`models/`)

```
models/
└── .gitkeep              # Stores trained model files (.joblib, .h5)
```

### Reports Directory (`reports/`)

```
reports/
├── figures/              # Generated graphics and figures
│   └── .gitkeep
└── .gitkeep
```

### Notebooks Directory (`notebooks/`)

```
notebooks/
├── 01-complete-workflow.ipynb    # End-to-end demo
└── 02-data-exploration.ipynb     # Data analysis
```

### Configuration (`config/`)

```
config/
└── .gitkeep              # Additional config files can go here
```

## 🗑️ Removed Files

The following files have been removed as they were duplicates or not needed for production:

- `README_old.md` - Old readme backup
- `check_project.py` - Development audit script
- `demo.py` - Development demo script
- `test_basic.py` - Development test
- `test_environment.py` - Development test
- `test_system.py` - Development test
- `setup_project.py` - Replaced by SETUP_PROJECT.bat
- `verify_setup.py` - Not needed in production
- `training_guide.py` - Information now in documentation
- `tox.ini` - Not needed for this project
- `Makefile` - Using CLI instead
- `__pycache__/` - Compiled Python files (auto-generated)
- `docs/` - Minimal Sphinx docs (not used)
- `references/` - Empty directory

## 📋 What Each Component Does

### Core Application
- **`cli.py`**: Main command-line interface with commands for setup, preprocessing, training, and prediction
- **`config.yaml`**: Default project settings (paths, model parameters, etc.)

### Source Code
- **`src/config.py`**: Configuration dataclasses and loaders
- **`src/utils.py`**: Logging, timing, experiment tracking
- **`src/data/make_dataset.py`**: Load images, split data, augment
- **`src/features/build_features.py`**: Extract texture, edge, statistical features
- **`src/models/train_model.py`**: Train Random Forest, SVM, CNN, Transfer Learning
- **`src/models/predict_model.py`**: Load models and make predictions
- **`src/visualization/visualize.py`**: Plot confusion matrices, ROC curves, etc.

### Setup & Documentation
- **`setup.py`**: Enables `pip install -e .` for development
- **`requirements.txt`**: All Python dependencies
- **`SETUP_GUIDE.md`**: Detailed setup instructions
- **`SETUP_PROJECT.bat`**: Automated Windows setup
- **`QUICK_START.bat`**: Quick environment activation

### Notebooks
- **`01-complete-workflow.ipynb`**: Complete pipeline demonstration
- **`02-data-exploration.ipynb`**: Data analysis and exploration

## 🚀 Quick Start

```bash
# 1. Run setup
SETUP_PROJECT.bat

# 2. Activate environment
activate.bat

# 3. Check installation
python cli.py info

# 4. Add your data to data/raw/authentic/ and data/raw/forged/

# 5. Run the pipeline
python cli.py preprocess-data data/raw/
python cli.py train-model data/processed/ --model-type traditional_ml
```

## 📦 Installing the Project

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install
pip install -e .

# Or use automated setup
SETUP_PROJECT.bat
```

## 🔧 Development vs Production

**This is now a production-ready structure with:**
- ✅ No duplicate files
- ✅ No test/development scripts
- ✅ Clean, organized directories
- ✅ Clear documentation
- ✅ Easy setup process

**For development, you can add:**
- `tests/` directory for unit tests
- Additional notebooks for experiments
- Development scripts as needed
