"""
Reorganize dataset with synthetic forgeries and retrain the model.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tqdm import tqdm
import sys
sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

print("="*70)
print("🔄 REORGANIZING DATASET WITH SYNTHETIC FORGERIES")
print("="*70)

# Create new organized dataset directory
new_raw_dir = Path('data/raw_synthetic')
new_authentic_dir = new_raw_dir / 'authentic'
new_forged_dir = new_raw_dir / 'forged'

new_authentic_dir.mkdir(parents=True, exist_ok=True)
new_forged_dir.mkdir(parents=True, exist_ok=True)

# Copy authentic images (sample to balance dataset)
print("\n📋 Copying authentic images...")
authentic_src = Path('data/raw/authentic')
authentic_images = list(authentic_src.glob('*.jpg'))[:1800]  # Match forged count

for img_path in tqdm(authentic_images, desc="Copying authentic"):
    shutil.copy(img_path, new_authentic_dir / img_path.name)

# Copy synthetic forged images
print("\n📋 Copying synthetic forged images...")
forged_src = Path('data/raw/forged_synthetic')
forged_images = list(forged_src.glob('forged_*.jpg'))

for img_path in tqdm(forged_images, desc="Copying forged"):
    shutil.copy(img_path, new_forged_dir / img_path.name)

print(f"\n✅ Dataset organized!")
print(f"  - Authentic: {len(list(new_authentic_dir.glob('*.jpg')))} images")
print(f"  - Forged: {len(list(new_forged_dir.glob('*.jpg')))} images")

# Now extract features and train
print("\n" + "="*70)
print("🔬 EXTRACTING FEATURES FROM NEW DATASET")
print("="*70)

extractor = DocumentFeatureExtractor()

X = []
y = []
filenames = []

# Process authentic images
print("\n📊 Processing authentic images...")
authentic_imgs = list(new_authentic_dir.glob('*.jpg'))[:1200]  # Limit for speed
for img_path in tqdm(authentic_imgs, desc="Authentic features"):
    features = extractor.extract_all_features(str(img_path))
    if features:
        # Convert dict to list (exclude metadata)
        feature_values = [v for k, v in features.items() 
                         if k not in ['filename', 'filepath']]
        X.append(feature_values)
        y.append(0)  # 0 = authentic
        filenames.append(img_path.name)

# Process forged images
print("\n📊 Processing forged images...")
forged_imgs = list(new_forged_dir.glob('*.jpg'))[:1200]  # Match authentic count
for img_path in tqdm(forged_imgs, desc="Forged features"):
    features = extractor.extract_all_features(str(img_path))
    if features:
        feature_values = [v for k, v in features.items() 
                         if k not in ['filename', 'filepath']]
        X.append(feature_values)
        y.append(1)  # 1 = forged
        filenames.append(img_path.name)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Features extracted!")
print(f"  - Total samples: {len(X)}")
print(f"  - Features per sample: {X.shape[1]}")
print(f"  - Authentic: {np.sum(y == 0)}")
print(f"  - Forged: {np.sum(y == 1)}")

# Train model
print("\n" + "="*70)
print("🚀 TRAINING MODEL WITH SYNTHETIC FORGERIES")
print("="*70)

# Split data
print("\n📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest with better parameters
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=30,      # Deeper trees
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf.fit(X_train_scaled, y_train)

# Evaluate
print("\n📈 Evaluating model...")
y_pred_train = rf.predict(X_train_scaled)
y_pred_test = rf.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("\n" + "="*70)
print("📊 TRAINING RESULTS WITH SYNTHETIC FORGERIES")
print("="*70)
print(f"\n🎯 Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"🎯 Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Authentic', 'Forged'],
                          digits=4))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"\nTrue Negatives (Authentic correctly identified): {cm[0][0]}")
print(f"False Positives (Authentic wrongly flagged as forged): {cm[0][1]}")
print(f"False Negatives (Forged wrongly identified as authentic): {cm[1][0]}")
print(f"True Positives (Forged correctly identified): {cm[1][1]}")

# Feature importance
print("\n🔍 Top 10 Most Important Features:")
feature_names = list(extractor.extract_all_features(str(authentic_imgs[0])).keys())
feature_names = [f for f in feature_names if f not in ['filename', 'filepath']]

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(10, len(feature_names))):
    idx = indices[i]
    if idx < len(feature_names):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Save improved model
print("\n💾 Saving improved model...")
joblib.dump(rf, 'models/random_forest_synthetic_v2.joblib')
joblib.dump(scaler, 'models/scaler_synthetic_v2.joblib')

print("\n✅ Model saved successfully!")
print("  - models/random_forest_synthetic_v2.joblib")
print("  - models/scaler_synthetic_v2.joblib")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

# Compare with previous model
print("\n📊 COMPARISON:")
print(f"  Previous model (with augmented forgeries): 45.02% accuracy")
print(f"  New model (with synthetic forgeries):      {test_acc*100:.2f}% accuracy")
print(f"  Improvement: {(test_acc - 0.4502)*100:.2f} percentage points")
