# 🔍 ForgeX - Document Forgery Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-94.94%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

**A comprehensive machine learning system for detecting forged documents using computer vision and advanced image analysis.**

[🌐 Web App](#-web-application) • [🔌 API](#-rest-api) • [📊 Features](#-features) • [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation)

</div>

---

## 🎯 Overview

ForgeX is a state-of-the-art document forgery detection system that combines ensemble machine learning with advanced computer vision to identify manipulated documents with **94.94% accuracy**.

### ✨ Key Highlights

- 🎯 **94.94% Testing Accuracy** with ensemble models
- 📈 **98.22% ROC AUC Score** for reliable predictions
- 🔥 **97.24% Precision** on forgery detection
- 🌐 **Interactive Web Interface** with real-time predictions
- 🔌 **REST API** for easy integration
- 📊 **32 Advanced Features** extracted from each document
- 🖼️ **8,000+ Images** training dataset
- 🎨 **Beautiful Visualizations** for result interpretation

### 🔍 Detection Capabilities

- **Copy-Move Forgery**: Regions copied within the same document
- **Splicing**: Content from different documents merged together  
- **Digital Tampering**: Altered text, signatures, or stamps
- **Print-Scan Forgery**: Documents printed and re-scanned
- **Content Manipulation**: Modified dates, amounts, or textual content
- **JPEG Artifacts**: Compression-based forgery detection
- **Noise Analysis**: Statistical inconsistencies

---

## 🚀 Quick Start

### Option 1: Web Application (Recommended) 🌐

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web interface
streamlit run app.py
```

Then open **http://localhost:8501** in your browser!

![Web App Demo](reports/figures/web_app_demo.png)

### Option 2: REST API 🔌

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python api.py
```

API available at **http://localhost:5000**

### Option 3: Command Line 💻

```bash
# Test single image
python test_model.py --image path/to/document.jpg

# Test folder
python test_model.py --folder path/to/images/
```

**Sample Output:**
```
✅ Prediction: Authentic
🎯 Confidence: 95.65%
📊 Probabilities: Authentic: 95.65% | Forged: 4.35%
```

---

## 🌐 Web Application

ForgeX includes a beautiful, interactive web interface built with Streamlit:

### Features:
- 📤 **Drag-and-drop** image upload
- ⚡ **Real-time predictions** with confidence scores
- 📊 **Interactive visualizations** (confidence charts, feature importance)
- 🔍 **Detailed analysis** of detection factors
- 📁 **Batch processing** for multiple images
- 💾 **Downloadable results** in CSV format

### Interface Highlights:

#### 1. **Upload & Analyze Page**
- Drag-and-drop interface for easy image upload
- Instant prediction with confidence percentage
- Color-coded results (🟢 Green = Authentic, 🔴 Red = Forged)
- Progress bars showing confidence levels

#### 2. **Visualizations**
- **Confidence Pie Chart**: Visual breakdown of prediction
- **Feature Importance**: Bar chart showing top contributing factors
- **Probability Distribution**: Detailed probability analysis

#### 3. **Batch Processing**
- Upload multiple images simultaneously
- Summary statistics (total, authentic, forged)
- Export results to CSV for further analysis

### Quick Demo:

```bash
streamlit run app.py
```

See [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) for detailed usage instructions.

---

## 🔌 REST API

ForgeX provides a production-ready Flask API for integration:

### Endpoints:

#### **POST /api/predict** - Single Document Analysis

```bash
curl -X POST -F "image=@document.jpg" http://localhost:5000/api/predict
```

**Response:**
```json
{
  "success": true,
  "result": {
    "prediction": "authentic",
    "confidence": 95.65,
    "probabilities": {
      "authentic": 95.65,
      "forged": 4.35
    },
    "timestamp": "2025-01-25T10:30:00"
  },
  "filename": "document.jpg"
}
```

#### **POST /api/batch** - Batch Processing

```bash
curl -X POST \
  -F "images=@doc1.jpg" \
  -F "images=@doc2.jpg" \
  -F "images=@doc3.jpg" \
  http://localhost:5000/api/batch
```

#### **GET /api/stats** - Model Statistics

```bash
curl http://localhost:5000/api/stats
```

#### **GET /api/health** - Health Check

```bash
curl http://localhost:5000/api/health
```

### Python Integration Example:

```python
import requests

# Single prediction
with open('document.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    
print(f"Prediction: {result['result']['prediction']}")
print(f"Confidence: {result['result']['confidence']}%")
```

See [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) for complete API documentation.

---

## 📊 Performance Metrics

### Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 94.94% |
| **ROC AUC Score** | 98.22% |
| **Forgery Precision** | 97.24% |
| **Forgery Recall** | 92.50% |
| **Training Dataset** | 8,000 images |
| **Test Dataset** | 1,600 images |

### Confusion Matrix (Test Set)

```
                Predicted
                Auth  Forged
Actual Auth    [ 779    21 ]  97.37% correct
       Forged  [  60   740 ]  92.50% correct
```

### Model Composition

Our ensemble model combines:
- **Random Forest** (100 estimators)
- **Gradient Boosting** (100 estimators)
- **Support Vector Machine** (RBF kernel)
- **Logistic Regression** (L2 regularization)

Voting Strategy: **Soft Voting** (probability-based)

---

## 🔬 Features

### Advanced Feature Extraction (32 Features)

#### 1. **Statistical Features** (8 features)
- Mean, variance, standard deviation
- Skewness, kurtosis
- Channel-wise statistics (RGB)

#### 2. **Edge Detection** (6 features)
- Laplacian edge detection
- Sobel operators (horizontal/vertical)
- Canny edge detection
- Edge density and consistency

#### 3. **Texture Analysis** (8 features)
- GLCM (Gray Level Co-occurrence Matrix)
- LBP (Local Binary Patterns)
- Contrast, homogeneity
- Energy, correlation

#### 4. **Frequency Domain** (5 features)
- FFT (Fast Fourier Transform)
- DCT (Discrete Cosine Transform)
- High-frequency content analysis
- Compression artifacts

#### 5. **JPEG Analysis** (5 features)
- JPEG quality estimation
- Quantization table analysis
- Block artifacts detection
- Compression consistency
- Double JPEG detection

### Feature Engineering Process

```python
from src.features.build_features import DocumentFeatureExtractor

extractor = DocumentFeatureExtractor()
features = extractor.extract_all_features('path/to/image.jpg')
# Returns dictionary with 32 features
```

---

## 🏗️ Project Structure

```
Document-Forgery-Detection/
│
├── app.py                      # Streamlit web application
├── api.py                      # Flask REST API
├── train_final_model.py        # Model training script
├── test_model.py               # CLI testing interface
│
├── src/                        # Source code
│   ├── data/
│   │   └── make_dataset.py     # Data loading and preprocessing
│   ├── features/
│   │   └── build_features.py   # Feature extraction
│   ├── models/
│   │   ├── train_model.py      # Model training
│   │   └── predict_model.py    # Model inference
│   └── visualization/
│       └── visualize.py        # Plotting and visualization
│
├── models/                     # Saved models
│   ├── final_ensemble_model.joblib
│   ├── final_scaler.joblib
│   └── final_feature_selector.joblib
│
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # Dataset directory
│   ├── raw/                    # Original images
│   ├── processed/              # Preprocessed images
│   └── interim/                # Intermediate data
│
├── reports/                    # Generated reports and figures
│   └── figures/                # Visualizations
│
├── requirements.txt            # Python dependencies
├── WEB_APP_GUIDE.md           # Web app documentation
├── TESTING_GUIDE.md           # Testing instructions
└── README.md                  # This file
```

---

## 📥 Installation

### Prerequisites

- Python 3.13+ (or 3.8+)
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the repository:**

```bash
git clone https://github.com/JanhviAditi/ForgeX.git
cd ForgeX
```

2. **Create virtual environment:**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify installation:**

```bash
python test_environment.py
```

---

## 🎓 Usage Examples

### 1. Web Application

```bash
# Launch the web app
streamlit run app.py

# The app will open in your browser at http://localhost:8501
# Drag and drop an image to get instant predictions
```

### 2. REST API

```bash
# Start the API server
python api.py

# Test with curl
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/predict

# Or use Python
import requests
response = requests.post(
    'http://localhost:5000/api/predict',
    files={'image': open('test_image.jpg', 'rb')}
)
print(response.json())
```

### 3. Command Line Interface

```bash
# Test single image
python test_model.py --image path/to/image.jpg

# Test entire folder
python test_model.py --folder path/to/images/

# Get detailed output
python test_model.py --image path/to/image.jpg --verbose
```

### 4. Python Script Integration

```python
import joblib
import numpy as np
from src.features.build_features import DocumentFeatureExtractor

# Load model
model = joblib.load('models/final_ensemble_model.joblib')
scaler = joblib.load('models/final_scaler.joblib')
selector = joblib.load('models/final_feature_selector.joblib')

# Extract features
extractor = DocumentFeatureExtractor()
features = extractor.extract_all_features('image.jpg')

# Prepare for prediction
features_array = np.array(list(features.values())).reshape(1, -1)
features_scaled = scaler.transform(features_array)
features_selected = selector.transform(features_scaled)

# Predict
prediction = model.predict(features_selected)[0]
probabilities = model.predict_proba(features_selected)[0]

print(f"Prediction: {'Authentic' if prediction == 0 else 'Forged'}")
print(f"Confidence: {max(probabilities) * 100:.2f}%")
```

---

## 🧪 Model Training

### Training from Scratch

```bash
# 1. Prepare your dataset
# Place images in:
#   - data/raw/authentic/
#   - data/raw/forged/

# 2. Train the model
python train_final_model.py

# 3. The trained model will be saved to models/
```

### Custom Training Configuration

Edit `train_final_model.py` to customize:
- Model hyperparameters
- Feature selection method
- Cross-validation strategy
- Ensemble composition

---

## 📊 Dataset

### Dataset Structure

```
data/
├── raw/
│   ├── authentic/          # Authentic documents
│   │   ├── doc_001.jpg
│   │   ├── doc_002.jpg
│   │   └── ...
│   └── forged/             # Forged documents
│       ├── forged_001.jpg
│       ├── forged_002.jpg
│       └── ...
```

### Statistics

- **Total Images**: 10,226
- **Authentic**: 4,826 images
- **Forged**: 5,400 images
- **Training Set**: 8,000 images (80%)
- **Test Set**: 1,600 images (20%)
- **Validation Set**: 10-fold cross-validation

### Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- Image resolution: 200x200 to 4000x4000 pixels

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
MODEL_PATH=models/final_ensemble_model.joblib
SCALER_PATH=models/final_scaler.joblib
SELECTOR_PATH=models/final_feature_selector.joblib
UPLOAD_FOLDER=uploads
MAX_FILE_SIZE=16777216  # 16MB
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
maxUploadSize = 200

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## 🚢 Deployment

### Streamlit Cloud (Web App)

1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository and deploy `app.py`

### Heroku (API)

```bash
# Create Procfile
echo "web: python api.py" > Procfile

# Deploy
heroku create forgex-api
git push heroku main
```

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t forgex .
docker run -p 8501:8501 forgex
```

---

## 📚 Documentation

- **[WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)** - Complete web app and API documentation
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing procedures and examples
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Project overview and achievements

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Janhvi Aditi**
- GitHub: [@JanhviAditi](https://github.com/JanhviAditi)
- Email: janhvi0912@gmail.com

---

## 🙏 Acknowledgments

- Dataset sources and contributors
- OpenCV and scikit-learn communities
- Streamlit team for the amazing web framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/JanhviAditi/ForgeX/issues)
- **Discussions**: [GitHub Discussions](https://github.com/JanhviAditi/ForgeX/discussions)
- **Email**: janhvi0912@gmail.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Janhvi Aditi

</div>
