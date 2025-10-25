# 🔍 Document Forgery Detection System# Document Forgery Detection



[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)<div align="center">

[![Accuracy](https://img.shields.io/badge/Accuracy-94.94%25-brightgreen.svg)](/)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning-based system for detecting forged Aadhaar cards with **94.94% accuracy** using ensemble methods and advanced feature engineering.![Status](https://img.shields.io/badge/status-production-brightgreen.svg)



---**A comprehensive machine learning system for detecting forged documents using computer vision and advanced image analysis techniques.**



## 🎯 Project Overview[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Contributing](#contributing)



This project implements a robust document forgery detection system specifically designed for Aadhaar card verification. Using an ensemble of machine learning models and 32 engineered features, the system can accurately distinguish between authentic and forged documents.</div>



### 🌟 Key Achievements---

- ✅ **94.94% Testing Accuracy**

- ✅ **97.24% Precision** on forgery detection## 🎯 Overview

- ✅ **98.22% ROC AUC Score**

- ✅ **8,000 Images** training datasetThis project implements a state-of-the-art document forgery detection system that uses machine learning and computer vision techniques to identify manipulated documents. The system can detect various types of document forgeries including:

- ✅ **7 Forgery Techniques** for synthetic data generation

- **Copy-Move Forgery**: Detecting regions copied from one part of the document to another

---- **Splicing**: Identifying content from different documents merged together  

- **Digital Tampering**: Detecting digitally altered text, signatures, or stamps

## 📊 Performance Metrics- **Print-Scan Forgery**: Identifying documents that have been printed and re-scanned

- **Content Manipulation**: Detecting altered dates, amounts, or other textual content

| Metric | Value |

|--------|-------|## ✨ Features

| **Testing Accuracy** | 94.94% |

| **ROC AUC Score** | 98.22% |### 🔬 Advanced Analysis Techniques

| **Forgery Detection Precision** | 97.24% |- **Multi-Modal Feature Extraction**: Texture analysis (GLCM, LBP), edge detection, statistical features

| **Forgery Detection Recall** | 92.50% |- **Frequency Domain Analysis**: FFT-based artifact detection and compression analysis

| **Training Dataset Size** | 8,000 images |- **Deep Learning Models**: CNN and transfer learning approaches for complex pattern recognition

| **Test Dataset Size** | 1,600 images |- **Traditional ML Models**: Random Forest, SVM, and ensemble methods for robust classification



### Confusion Matrix (Test Set)### 🛠️ Production-Ready Components

```- **Complete ML Pipeline**: From data preprocessing to model deployment

                Predicted- **CLI Interface**: Easy-to-use command-line tools for all operations

                Auth  Forged- **Comprehensive Visualization**: Model performance analysis and result interpretation

Actual Auth    [ 779    21 ]- **Configurable Architecture**: YAML-based configuration for easy customization

       Forged  [  60   740 ]- **Experiment Tracking**: Built-in logging and experiment management

```

### 📊 Supported Formats

---- **Image Formats**: JPEG, PNG, TIFF, BMP

- **Document Types**: Scanned documents, digital documents, mixed content

## 🚀 Quick Start- **Batch Processing**: Efficient processing of large document collections



### Installation## 🚀 Quick Start

```bash

# Clone the repository### Installation

git clone https://github.com/JanhviAditi/Forgery_Detection.git

cd Forgery_Detection1. **Clone the repository:**

```bash

# Create virtual environmentgit clone https://github.com/Ztrimus/Document-Forgery-Detection.git

python -m venv venvcd Document-Forgery-Detection

venv\Scripts\activate  # Windows```

# source venv/bin/activate  # Linux/Mac

2. **Set up Python environment:**

# Install dependencies```bash

pip install -r requirements.txtpython -m venv venv

```source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### Test Your First Image

```bash3. **Install the package:**

python test_model.py --image path/to/your/image.jpg```bash

```pip install -e .

```

**Output:**

```4. **Initialize the project:**

✅ Prediction: Authentic```bash

🎯 Confidence: 95.65%python cli.py setup

``````



---### Basic Usage



## 🔬 Features1. **Prepare your data:**

```bash

### Advanced Feature Extraction (32 Features)# Place your document images in data/raw/ directory

- Statistical features (mean, variance, std)# Organize as: data/raw/authentic/ and data/raw/forged/

- Edge detection (Laplacian, Sobel)

- DCT frequency analysispython cli.py preprocess-data data/raw/

- Compression artifact detection```

- Noise pattern analysis

- Edge consistency metrics2. **Extract features:**

- Contrast variation analysis```bash

- Color histogram entropypython cli.py extract-features data/processed/train/ --output-file data/processed/features.csv

```

### Ensemble Learning Models

- Random Forest (300 trees)3. **Train a model:**

- Gradient Boosting (95.44% accuracy)```bash

- SVM (RBF kernel)python cli.py train-model data/processed/ --features-path data/processed/features.csv --model-type traditional_ml

- Logistic Regression```

- Voting Classifier (soft voting)

4. **Make predictions:**

### Synthetic Forgery Generation```bash

7 sophisticated techniques:python cli.py predict models/random_forest_model.joblib path/to/test/image.jpg

1. Copy-Move```

2. Splicing

3. Text Swap## 📖 Documentation

4. Photo Swap

5. Compression Artifacts### CLI Commands

6. Noise Injection

7. Blur Application| Command | Description | Example |

|---------|-------------|---------|

---| `setup` | Initialize project structure | `python cli.py setup` |

| `preprocess-data` | Process raw images for training | `python cli.py preprocess-data data/raw/` |

## 📖 Usage| `extract-features` | Extract ML features from images | `python cli.py extract-features data/processed/train/` |

| `train-model` | Train a forgery detection model | `python cli.py train-model data/processed/ --model-type cnn` |

### Test Single Image| `predict` | Make predictions on new images | `python cli.py predict model.joblib image.jpg` |

```bash| `info` | Show project and system information | `python cli.py info` |

python test_model.py --image "C:\path\to\image.jpg"

```### Model Types



### Test Folder#### Traditional Machine Learning

```bash- **Random Forest**: Ensemble method with excellent interpretability

python test_model.py --folder "C:\path\to\folder" --limit 10- **SVM**: Support Vector Machine with RBF kernel for non-linear classification

```- **Gradient Boosting**: Advanced ensemble method for complex patterns



### Interactive Mode#### Deep Learning

```bash- **CNN**: Custom convolutional neural network for image classification

python test_model.py -i- **Transfer Learning**: Pre-trained models (ResNet, VGG, etc.) fine-tuned for document analysis

```

### Configuration

### Visual Testing

```bashThe system uses YAML configuration files for easy customization:

python test_visual.py

``````yaml

# config.yaml

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for complete documentation.data:

  target_image_size: [224, 224]

---  train_split: 0.7

  validation_split: 0.2

## 🎓 Training  test_split: 0.1



### Generate Synthetic Forgeriesmodel:

```bash  random_state: 42

python generate_forgeries.py --authentic-dir data/raw/authentic --output-dir data/raw/forged --num-forgeries 3  use_grid_search: false

```

features:

### Train Model  extract_texture_features: true

```bash  extract_edge_features: true

python train_final_model.py  extract_frequency_features: true

``````



---### Python API Usage



## 📈 Model Performance Journey```python

from src.models.train_model import DocumentForgeryDetector

| Version | Dataset | Accuracy | Improvement |from src.models.predict_model import DocumentForgeryPredictor

|---------|---------|----------|-------------|

| V1 | 2.4k | 45.02% | Baseline |# Train a model

| V2 | 2.4k | 76.17% | +31.15% |detector = DocumentForgeryDetector(model_type='traditional_ml')

| V3 | 3.0k | 89.47% | +13.30% |X, y = detector.load_data('data/processed/features.csv')

| **V4** | **8.0k** | **94.94%** | **+5.47%** |results = detector.train_model(X, y)



**Total: +49.92% improvement!**# Make predictions

predictor = DocumentForgeryPredictor('models/model.joblib')

---result = predictor.predict_single_image('test_image.jpg')

print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")

## 📁 Project Structure```



```## 📊 Project Structure

Document-Forgery-Detection/

├── src/features/            # Feature extraction```

├── models/                  # Trained models (not tracked)Document-Forgery-Detection/

├── data/                    # Datasets (not tracked)├── cli.py                 # Command-line interface

├── consolidate_all_datasets.py├── config.yaml           # Default configuration

├── generate_forgeries.py├── requirements.txt      # Python dependencies

├── train_final_model.py├── setup.py             # Package installation

├── test_model.py│

├── test_visual.py├── data/                # Data directories

├── TESTING_GUIDE.md│   ├── raw/            # Original document images

├── PROJECT_SUMMARY.md│   ├── processed/      # Preprocessed data

└── requirements.txt│   ├── interim/        # Intermediate processing results

```│   └── external/       # External datasets

│

---├── src/                 # Source code

│   ├── config.py       # Configuration management

## 🛠️ Tech Stack│   ├── utils.py        # Utility functions

│   ├── data/           # Data processing modules

- **Python 3.8+**│   ├── features/       # Feature extraction

- **scikit-learn**: ML models│   ├── models/         # Model training and prediction

- **OpenCV**: Image processing│   └── visualization/  # Plotting and analysis

- **NumPy/Pandas**: Data processing│

- **Matplotlib**: Visualization├── notebooks/          # Jupyter notebooks

│   ├── 01-complete-workflow.ipynb

---│   └── 02-data-exploration.ipynb

│

## 📚 Documentation├── models/             # Trained model files

├── reports/            # Analysis reports and results

- [TESTING_GUIDE.md](TESTING_GUIDE.md) - How to test the model└── docs/              # Documentation

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Complete project details```

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Setup instructions

## 🧪 Model Performance

---

Our system achieves state-of-the-art performance on document forgery detection:

## ⚠️ Limitations

| Model Type | Accuracy | Precision | Recall | F1-Score |

- Trained on synthetic forgeries only|------------|----------|-----------|---------|----------|

- Specific to Aadhaar card format| Random Forest | 94.2% | 93.8% | 94.6% | 94.2% |

- 7.5% false negative rate| SVM | 92.7% | 91.9% | 93.5% | 92.7% |

- May miss sophisticated forgeries| CNN | 96.1% | 95.8% | 96.4% | 96.1% |

| Transfer Learning | 97.3% | 97.1% | 97.5% | 97.3% |

---

## 🔬 Technical Details

## 🤝 Contributing

### Feature Extraction Methods

Contributions welcome! Areas for improvement:

- Deep learning models (CNNs)1. **Texture Analysis**

- Web/mobile interface   - Gray-Level Co-occurrence Matrix (GLCM) features

- Real forgery datasets   - Local Binary Patterns (LBP) for texture characterization

- Explainability features   - Gabor filter responses for directional texture analysis



---2. **Edge and Contour Analysis**

   - Canny edge detection for boundary analysis

## 📄 License   - Sobel operators for gradient-based features

   - Contour-based shape descriptors

MIT License - see [LICENSE](LICENSE) file

3. **Frequency Domain Features**

---   - Fast Fourier Transform (FFT) analysis

   - Discrete Cosine Transform (DCT) coefficients

## 👥 Author   - JPEG compression artifact detection



**Janhvi Aditi** - [GitHub](https://github.com/JanhviAditi)4. **Statistical Features**

   - Histogram-based color analysis

---   - Moment-based shape descriptors

   - Entropy and energy measures

## 🙏 Acknowledgments

### Deep Learning Architecture

- Roboflow for datasets

- scikit-learn & OpenCV communitiesOur CNN model uses a custom architecture optimized for document analysis:



---- **Convolutional Layers**: Feature extraction with batch normalization

- **Pooling Layers**: Spatial dimension reduction

<div align="center">- **Dropout**: Regularization to prevent overfitting

- **Dense Layers**: Final classification with softmax activation

**⭐ Star this repo if you find it helpful!**

## 📋 Requirements

Made with ❤️ for Document Security

- Python 3.7+

</div>- TensorFlow 2.8+

- OpenCV 4.5+
- scikit-learn 1.0+
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for complete list

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- Inspired by research in digital forensics and computer vision
- Special thanks to the open-source community for the amazing tools and libraries

## 📞 Support

- 📧 Email: contact@document-forgery-detection.com
- 🐛 Issues: [GitHub Issues](https://github.com/Ztrimus/Document-Forgery-Detection/issues)
- 📖 Documentation: [Project Wiki](https://github.com/Ztrimus/Document-Forgery-Detection/wiki)

---

<div align="center">
<p><strong>⭐ If you find this project useful, please give it a star! ⭐</strong></p>
</div>