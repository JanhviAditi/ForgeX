# ForgeX Web Application Guide

## 🌐 Overview

ForgeX provides **two web interfaces** for document forgery detection:

1. **Streamlit Web App** - Interactive UI for end users
2. **Flask REST API** - Backend API for integrations

---

## 🚀 Quick Start

### Running the Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Running the Flask API

```bash
python api.py
```

API will be available at: **http://localhost:5000**

---

## 📱 Streamlit Web App Features

### 1. **Upload & Analyze Page**
- **Drag-and-drop** image upload
- **Real-time prediction** display
- **Confidence visualization** (progress bars, pie chart)
- **Feature importance** analysis
- **Probability distribution** chart
- **Detailed analysis** of contributing factors

### 2. **Batch Processing Page**
- Upload **multiple images** at once
- Process **entire folders** of documents
- **Summary statistics** (total, authentic, forged)
- **Downloadable CSV** results
- **Visual comparison** of results

### 3. **About Page**
- Model performance metrics
- Training data statistics
- Feature extraction details
- Project information

---

## 🔌 Flask REST API Endpoints

### **GET /** - API Information
```bash
curl http://localhost:5000/
```

Response:
```json
{
  "name": "ForgeX API",
  "version": "1.0.0",
  "description": "Document Forgery Detection API",
  "accuracy": "94.94%",
  "endpoints": {
    "/api/predict": "POST - Single document analysis",
    "/api/batch": "POST - Batch document analysis",
    "/api/health": "GET - Health check",
    "/api/stats": "GET - Model statistics"
  }
}
```

### **GET /api/health** - Health Check
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-25T10:30:00"
}
```

### **GET /api/stats** - Model Statistics
```bash
curl http://localhost:5000/api/stats
```

Response:
```json
{
  "model": {
    "type": "Ensemble (Random Forest + Gradient Boosting + SVM + LR)",
    "accuracy": 94.94,
    "precision": 97.24,
    "recall": 92.50,
    "roc_auc": 98.22,
    "features": 32
  },
  "training": {
    "total_images": 8000,
    "authentic": 4000,
    "forged": 4000,
    "test_size": 1600
  }
}
```

### **POST /api/predict** - Single Document Prediction

**Using cURL:**
```bash
curl -X POST -F "image=@path/to/document.jpg" http://localhost:5000/api/predict
```

**Using Python:**
```python
import requests

with open('document.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    result = response.json()
    print(result)
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

### **POST /api/batch** - Batch Document Prediction

**Using cURL:**
```bash
curl -X POST \
  -F "images=@doc1.jpg" \
  -F "images=@doc2.jpg" \
  -F "images=@doc3.jpg" \
  http://localhost:5000/api/batch
```

**Using Python:**
```python
import requests

files = [
    ('images', open('doc1.jpg', 'rb')),
    ('images', open('doc2.jpg', 'rb')),
    ('images', open('doc3.jpg', 'rb'))
]

response = requests.post('http://localhost:5000/api/batch', files=files)
results = response.json()
print(results)
```

**Response:**
```json
{
  "success": true,
  "total_processed": 3,
  "successful": 3,
  "failed": 0,
  "summary": {
    "authentic": 2,
    "forged": 1
  },
  "results": [
    {
      "filename": "doc1.jpg",
      "result": {
        "prediction": "authentic",
        "confidence": 95.65,
        "probabilities": {"authentic": 95.65, "forged": 4.35},
        "timestamp": "2025-01-25T10:30:00"
      }
    },
    {
      "filename": "doc2.jpg",
      "result": {
        "prediction": "forged",
        "confidence": 87.32,
        "probabilities": {"authentic": 12.68, "forged": 87.32},
        "timestamp": "2025-01-25T10:30:01"
      }
    },
    {
      "filename": "doc3.jpg",
      "result": {
        "prediction": "authentic",
        "confidence": 92.15,
        "probabilities": {"authentic": 92.15, "forged": 7.85},
        "timestamp": "2025-01-25T10:30:02"
      }
    }
  ]
}
```

---

## 🎨 Streamlit App Usage

### **Step 1: Launch App**
```bash
streamlit run app.py
```

### **Step 2: Upload Document**
1. Click "Browse files" or drag-and-drop image
2. Supported formats: JPG, PNG, JPEG
3. Max file size: 200MB

### **Step 3: View Results**
- **Prediction**: AUTHENTIC ✅ or FORGED ❌
- **Confidence**: Percentage score with progress bar
- **Visualizations**: 
  - Confidence pie chart
  - Feature importance bar chart
  - Probability distribution

### **Step 4: Analyze Details**
- Click "📊 Show Detailed Analysis"
- View top contributing features
- Understand why document was classified

### **Batch Processing:**
1. Go to "Batch Processing" page
2. Upload multiple images
3. Click "Process All Images"
4. Download results as CSV

---

## 🔧 Configuration

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

### API Configuration
Edit `api.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
```

---

## 📊 Understanding Results

### **Confidence Score**
- **90-100%**: Very high confidence
- **75-90%**: High confidence
- **60-75%**: Moderate confidence
- **Below 60%**: Low confidence (review manually)

### **Feature Analysis**
Top features analyzed:
- **JPEG Quality**: Compression artifacts
- **Noise Pattern**: Statistical noise analysis
- **Edge Consistency**: Edge detection patterns
- **Color Distribution**: Histogram entropy
- **Texture Features**: LBP patterns
- **Frequency Analysis**: DCT coefficients

---

## 🐛 Troubleshooting

### Streamlit App Not Loading
```bash
# Check if Streamlit is installed
pip install streamlit

# Try running with Python module
python -m streamlit run app.py

# Check port availability
netstat -ano | findstr :8501
```

### API Not Responding
```bash
# Check Flask installation
pip install flask flask-cors

# Check port availability
netstat -ano | findstr :5000

# Run in debug mode
python api.py
```

### Model Not Loading
```bash
# Check model files exist
dir models\final_ensemble_model.joblib
dir models\final_scaler.joblib
dir models\final_feature_selector.joblib

# Retrain if necessary
python train_final_model.py
```

---

## 🚢 Deployment

### **Streamlit Cloud** (Recommended for Web App)

1. Push to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub repository
4. Deploy `app.py`

### **Heroku** (For Flask API)

```bash
# Create Procfile
echo "web: python api.py" > Procfile

# Create runtime.txt
echo "python-3.13.3" > runtime.txt

# Deploy
heroku create forgex-api
git push heroku main
```

### **Docker** (For Both)

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# For Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

# For API
# EXPOSE 5000
# CMD ["python", "api.py"]
```

---

## 📱 Mobile Access

Both apps are **mobile-responsive**:
- Access from any device on your network
- Streamlit: `http://<your-ip>:8501`
- API: `http://<your-ip>:5000`

---

## 🔐 Security Considerations

### For Production:
1. **Add authentication** (JWT tokens, API keys)
2. **Rate limiting** (prevent abuse)
3. **HTTPS** (SSL certificates)
4. **Input validation** (file type, size checks)
5. **Error handling** (don't expose stack traces)

### Example API Key Protection:
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/predict', methods=['POST'])
@require_api_key
def predict():
    # Your code here
```

---

## 💡 Tips & Best Practices

1. **Image Quality**: Higher resolution = better accuracy
2. **File Format**: JPG/PNG recommended
3. **Lighting**: Ensure good lighting in scanned documents
4. **Batch Size**: Process 10-50 images at a time
5. **Confidence**: Manual review for scores below 75%

---

## 📞 Support

- **GitHub Issues**: Report bugs or feature requests
- **Documentation**: See `README.md` for more info
- **Contact**: janhvi0912@gmail.com

---

## 🎯 Next Steps

- [ ] Add user authentication
- [ ] Implement file upload history
- [ ] Create admin dashboard
- [ ] Add API rate limiting
- [ ] Deploy to cloud
- [ ] Mobile app development

---

**Happy Detecting! 🔍**
