"""
FINAL MODEL TRAINING - Complete Dataset
========================================
Training with consolidated dataset:
- 4,826 authentic Aadhaar cards (from 3 sources)
- 5,401 synthetic forged documents
- Total: 10,227 images

Using best practices from previous iterations.
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
from tqdm import tqdm
import sys
sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

print("="*70)
print("🚀 FINAL MODEL TRAINING - COMPLETE DATASET")
print("="*70)

# Load dataset
print("\n📊 Loading consolidated dataset...")
authentic_dir = Path('data/consolidated/authentic')
forged_dir = Path('data/consolidated/forged')

extractor = DocumentFeatureExtractor()

def extract_enhanced_features(img_path):
    """Extract enhanced features with additional discriminative features."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Get base features
    features = extractor.extract_all_features(str(img_path))
    if not features:
        return None
    
    # Add additional advanced features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. DCT features (frequency domain)
    dct = cv2.dct(np.float32(gray))
    features['dct_mean'] = np.mean(dct)
    features['dct_std'] = np.std(dct)
    features['dct_max'] = np.max(np.abs(dct))
    
    # 2. Compression artifact analysis
    _, compressed = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img, decompressed)
    features['compression_diff_mean'] = np.mean(diff)
    features['compression_diff_std'] = np.std(diff)
    
    # 3. High-frequency noise analysis
    kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    high_freq = cv2.filter2D(gray, -1, kernel)
    features['noise_level'] = np.mean(np.abs(high_freq))
    features['noise_variance'] = np.var(high_freq)
    
    # 4. Edge consistency
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    features['edge_consistency'] = np.sum(dilated) / (gray.shape[0] * gray.shape[1])
    
    # 5. Local contrast variation (block-based)
    block_size = 32
    h, w = gray.shape
    contrasts = []
    for i in range(0, h-block_size, block_size):
        for j in range(0, w-block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            contrasts.append(np.std(block))
    features['contrast_variation'] = np.std(contrasts) if contrasts else 0
    features['contrast_mean'] = np.mean(contrasts) if contrasts else 0
    
    # 6. Color histogram entropy
    hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
    
    def entropy(hist):
        hist = hist / (hist.sum() + 1e-7)
        return -np.sum(hist * np.log2(hist + 1e-7))
    
    features['color_hist_entropy'] = (entropy(hist_b) + entropy(hist_g) + entropy(hist_r)) / 3
    
    return features

# Extract features
print("\n🔬 Extracting enhanced features from all images...")
print("   This will take a while with 10,000+ images...")

X = []
y = []

# Process authentic images (use subset for faster training if needed)
print("\n📊 Processing authentic images...")
authentic_images = list(authentic_dir.glob('*.jpg')) + list(authentic_dir.glob('*.jpeg')) + \
                   list(authentic_dir.glob('*.png'))

print(f"   Found {len(authentic_images)} authentic images")

# Limit to 4000 for balanced dataset
authentic_sample = authentic_images[:4000] if len(authentic_images) > 4000 else authentic_images

for img_path in tqdm(authentic_sample, desc="Authentic features"):
    features = extract_enhanced_features(img_path)
    if features:
        X.append(list(features.values()))
        y.append(0)  # 0 = authentic

print(f"Loaded {len([label for label in y if label == 0])} authentic samples")

# Process forged images
print("\n📊 Processing forged images...")
forged_images = list(forged_dir.glob('*.jpg')) + list(forged_dir.glob('*.jpeg')) + \
                list(forged_dir.glob('*.png'))

print(f"   Found {len(forged_images)} forged images")

# Limit to 4000 for balanced dataset
forged_sample = forged_images[:4000] if len(forged_images) > 4000 else forged_images

for img_path in tqdm(forged_sample, desc="Forged features"):
    features = extract_enhanced_features(img_path)
    if features:
        X.append(list(features.values()))
        y.append(1)  # 1 = forged

print(f"Loaded {len([label for label in y if label == 1])} forged samples")

X = np.array(X)
y = np.array(y)

print(f"\n✅ Total samples: {len(X)}")
print(f"   Features: {X.shape[1]}")
print(f"   Authentic: {np.sum(y == 0)}")
print(f"   Forged: {np.sum(y == 1)}")

# Split dataset
print("\n📊 Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Scale features
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
print("\n🔍 Selecting best features...")
selector = SelectKBest(f_classif, k=min(25, X.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
print(f"   Selected {X_train_selected.shape[1]} best features")

print("\n" + "="*70)
print("🎯 TRAINING ENSEMBLE OF MODELS")
print("="*70)

# Train individual models
print("\n1️⃣  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_selected, y_train)
rf_acc = rf.score(X_test_selected, y_test)
print(f"   Random Forest accuracy: {rf_acc:.4f}")

print("\n2️⃣  Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_selected, y_train)
gb_acc = gb.score(X_test_selected, y_test)
print(f"   Gradient Boosting accuracy: {gb_acc:.4f}")

print("\n3️⃣  Training SVM...")
svm = SVC(
    C=10,
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42
)
svm.fit(X_train_selected, y_train)
svm_acc = svm.score(X_test_selected, y_test)
print(f"   SVM accuracy: {svm_acc:.4f}")

print("\n4️⃣  Training Logistic Regression...")
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_selected, y_train)
lr_acc = lr.score(X_test_selected, y_test)
print(f"   Logistic Regression accuracy: {lr_acc:.4f}")

# Create voting ensemble
print("\n5️⃣  Creating Voting Ensemble...")
voting = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svm', svm),
        ('lr', lr)
    ],
    voting='soft',
    weights=[2, 2, 1, 1]  # Give more weight to RF and GB
)
voting.fit(X_train_selected, y_train)
voting_acc = voting.score(X_test_selected, y_test)
print(f"   Voting Ensemble accuracy: {voting_acc:.4f}")

print("\n" + "="*70)
print("📊 DETAILED EVALUATION OF VOTING ENSEMBLE")
print("="*70)

# Predictions
y_pred = voting.predict(X_test_selected)
y_pred_proba = voting.predict_proba(X_test_selected)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 Testing Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"📈 ROC AUC Score: {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Authentic', 'Forged']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\n✅ True Negatives (Authentic correctly identified): {tn}")
print(f"⚠️  False Positives (Authentic wrongly flagged): {fp}")
print(f"⚠️  False Negatives (Forged missed): {fn}")
print(f"✅ True Positives (Forged correctly detected): {tp}")

# Forged detection metrics
precision_forged = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_forged = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_forged = 2 * (precision_forged * recall_forged) / (precision_forged + recall_forged) if (precision_forged + recall_forged) > 0 else 0

print(f"\n🎯 Forged Document Detection:")
print(f"   Precision: {precision_forged:.4f} ({precision_forged*100:.2f}%)")
print(f"   Recall: {recall_forged:.4f} ({recall_forged*100:.2f}%)")
print(f"   F1-Score: {f1_forged:.4f}")

# Cross-validation (on subset for speed)
print("\n🔄 Cross-validation (5-fold on training set)...")
cv_scores = cross_val_score(voting, X_train_selected[:2000], y_train[:2000], cv=5, n_jobs=1)
print(f"   CV Scores: {cv_scores}")
print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save the models
print("\n💾 Saving models...")
joblib.dump(voting, 'models/final_ensemble_model.joblib')
joblib.dump(rf, 'models/final_random_forest.joblib')
joblib.dump(gb, 'models/final_gradient_boosting.joblib')
joblib.dump(svm, 'models/final_svm.joblib')
joblib.dump(scaler, 'models/final_scaler.joblib')
joblib.dump(selector, 'models/final_feature_selector.joblib')

print("\n✅ All models saved!")
print("   - models/final_ensemble_model.joblib (BEST)")
print("   - models/final_random_forest.joblib")
print("   - models/final_gradient_boosting.joblib")
print("   - models/final_svm.joblib")
print("   - models/final_scaler.joblib")
print("   - models/final_feature_selector.joblib")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)

print("\n📊 FINAL ACCURACY PROGRESSION:")
print("   1st model (augmented, 2.4k images):    45.02%")
print("   2nd model (synthetic, 2.4k images):    76.17%")
print("   3rd model (ensemble, 3k images):       89.47%")
print(f"   4th model (FINAL, {len(X)/1000:.1f}k images):        {accuracy*100:.2f}% 🚀")

print("\n💡 Model is ready for deployment!")
print(f"   Dataset size: {len(X):,} images")
print(f"   Test accuracy: {accuracy*100:.2f}%")
print(f"   ROC AUC: {roc_auc:.4f}")
