# Model Testing Guide

## 📋 Overview

This guide explains how to test your Document Forgery Detection model (94.94% accuracy).

## 🎯 Available Testing Scripts

### 1. `test_model.py` - Command Line Testing
Professional command-line interface for testing single images, folders, or interactive mode.

### 2. `test_visual.py` - Visual Testing  
Creates visual outputs with prediction overlays (requires matplotlib).

---

## 🚀 Quick Start

### Test a Single Image
```bash
python test_model.py --image path/to/your/image.jpg
```

**Example:**
```bash
python test_model.py --image "data\consolidated\authentic\sample.jpg"
```

**Output:**
```
======================================================================
📄 File: sample.jpg
======================================================================
✅ Prediction: Authentic
🎯 Confidence: 89.3%

📊 Probabilities:
   ✅ Authentic: 89.3%
   ⚠️  Forged:    10.7%
======================================================================
```

---

### Test Multiple Images (Folder)
```bash
python test_model.py --folder path/to/folder
```

**Example:**
```bash
python test_model.py --folder "data\consolidated\authentic" --limit 10
```

**Output:**
```
🔍 Testing 10 images from: data\consolidated\authentic
======================================================================
✅ image1.jpg -> Authentic (89.3%)
✅ image2.jpg -> Authentic (92.1%)
⚠️ image3.jpg -> Forged (71.2%)
...

📊 SUMMARY
✅ Authentic: 8
⚠️  Forged:    2
📁 Total:     10
```

---

### Interactive Mode
```bash
python test_model.py --interactive
```

**Usage:**
```
🎮 INTERACTIVE TESTING MODE
Enter image path to test (or 'quit' to exit)

📄 Image path: data\test\image1.jpg
✅ Prediction: Authentic (89.3%)

📄 Image path: data\test\image2.jpg
⚠️ Prediction: Forged (95.2%)

📄 Image path: quit
👋 Goodbye!
```

---

## 📊 All Command Options

### Basic Usage
```bash
# Single image
python test_model.py --image <path>

# Folder
python test_model.py --folder <path>

# Folder with limit
python test_model.py --folder <path> --limit 20

# Interactive mode
python test_model.py --interactive
python test_model.py -i

# Use different model
python test_model.py --image test.jpg --model models/final_gradient_boosting.joblib
```

---

## 🎨 Visual Testing

### Generate Visual Report
```bash
python test_visual.py
```

This will:
1. Test 6 sample images (3 authentic + 3 forged)
2. Create a visual grid with predictions
3. Save to `reports/figures/test_results.png`
4. Display the results

**Requirements:**
```bash
pip install matplotlib
```

---

## 📁 Testing Your Own Images

### Option 1: Test Single Image
1. Have your Aadhaar card image ready (JPG/PNG)
2. Run:
   ```bash
   python test_model.py --image "C:\path\to\your\image.jpg"
   ```

### Option 2: Test a Folder
1. Put all your test images in a folder
2. Run:
   ```bash
   python test_model.py --folder "C:\path\to\your\folder"
   ```

### Option 3: Interactive Testing
1. Start interactive mode:
   ```bash
   python test_model.py -i
   ```
2. Paste image paths when prompted
3. Type 'quit' to exit

---

## 🔍 Understanding the Results

### Prediction
- **✅ Authentic** - The document appears genuine
- **⚠️ Forged** - The document shows signs of manipulation

### Confidence Score
- **90-100%** - Very confident (high reliability)
- **70-89%** - Confident (good reliability)
- **50-69%** - Uncertain (review manually)
- **<50%** - Low confidence (likely borderline case)

### Probabilities
Shows the model's probability distribution:
- Higher authentic % → More likely authentic
- Higher forged % → More likely forged

---

## 🎯 Testing Strategies

### 1. Validation Testing
Test on images you know the ground truth:
```bash
# Test known authentic images
python test_model.py --folder "data\consolidated\authentic" --limit 50

# Test known forged images  
python test_model.py --folder "data\consolidated\forged" --limit 50
```

### 2. Real-World Testing
Test on new, unseen Aadhaar cards:
```bash
python test_model.py --folder "data\new_test_images"
```

### 3. Batch Processing
Test large batches and save results:
```bash
python test_model.py --folder "data\production" > test_results.txt
```

---

## 📊 Model Comparison

Test different models to compare performance:

```bash
# Test with ensemble model (BEST - 94.94%)
python test_model.py --image test.jpg --model models/final_ensemble_model.joblib

# Test with Gradient Boosting (95.44%)
python test_model.py --image test.jpg --model models/final_gradient_boosting.joblib

# Test with Random Forest (94.81%)
python test_model.py --image test.jpg --model models/final_random_forest.joblib

# Test with older model (89.47%)
python test_model.py --image test.jpg --model models/ensemble_voting_model.joblib
```

---

## 🛠️ Troubleshooting

### Error: "Model not found"
**Solution:** Make sure you've trained the model first:
```bash
python train_final_model.py
```

### Error: "Cannot read image"
**Solution:** 
- Check the file path is correct
- Ensure image is JPG, PNG, or BMP format
- Verify the image isn't corrupted

### Low Confidence Scores
**Causes:**
- Image quality too poor
- Unusual Aadhaar card format
- Borderline forgery case

**Solution:** Manually review these cases

### Incorrect Predictions
**Remember:** Model is 94.94% accurate, meaning:
- ~5% of predictions may be wrong
- Some authentic images may be flagged
- Some forgeries may slip through

---

## 📈 Performance Expectations

Based on testing set (1,600 images):
- **Overall Accuracy:** 94.94%
- **Authentic Detection:** 97.4% recall (catches 97.4% of authentic)
- **Forgery Detection:** 92.5% recall (catches 92.5% of forgeries)
- **Precision (Forgery):** 97.24% (when it says forged, 97% accurate)

---

## 💡 Best Practices

### ✅ DO:
- Test on high-quality images (clear, well-lit)
- Use consistent image formats (JPG recommended)
- Review low-confidence predictions manually
- Test regularly with known samples

### ❌ DON'T:
- Test on extremely low-resolution images (<500px)
- Test on partially cropped documents
- Rely solely on model for critical decisions
- Ignore confidence scores

---

## 🔬 Advanced Testing

### Python API Usage
```python
from test_model import ForgeryDetector

# Initialize detector
detector = ForgeryDetector()

# Test an image
result = detector.predict('path/to/image.jpg')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Processing Script
```python
from test_model import ForgeryDetector
from pathlib import Path
import csv

detector = ForgeryDetector()

# Test all images
images = Path('data/test').glob('*.jpg')
results = []

for img in images:
    result = detector.predict(img)
    results.append({
        'filename': img.name,
        'prediction': result['prediction'],
        'confidence': result['confidence']
    })

# Save to CSV
with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'prediction', 'confidence'])
    writer.writeheader()
    writer.writerows(results)
```

---

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Verify model files exist in `models/` folder
3. Ensure all dependencies are installed
4. Check image format and quality

---

## 🎓 Summary

**Simplest way to test:**
```bash
python test_model.py -i
```
Then paste image paths when prompted!

**Most comprehensive:**
```bash
python test_model.py --folder "your_test_folder"
```

**Visual results:**
```bash
python test_visual.py
```

Happy testing! 🚀
