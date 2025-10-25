"""Quick training script for document forgery detection."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Loading features...")
# Load the features CSV
features_df = pd.read_csv('data/processed/features.csv')

print(f"Dataset loaded: {features_df.shape}")
print(f"Columns: {list(features_df.columns)}")

# Check for label column
if 'class' in features_df.columns:
    label_col = 'class'
elif 'label' in features_df.columns:
    label_col = 'label'
else:
    print("\nERROR: No 'class' or 'label' column found!")
    print("We need to extract features WITH labels.")
    print("The features CSV only contains feature values, not the class labels.")
    print("\nInstead, let's load images directly from the train folder...")
    
    # Alternative approach: Load images and extract features on the fly
    from pathlib import Path
    import cv2
    from src.features.build_features import DocumentFeatureExtractor
    
    train_dir = Path('data/processed/train')
    extractor = DocumentFeatureExtractor()
    
    X = []
    y = []
    
    print("\nExtracting features from training images...")
    # Load authentic images
    authentic_dir = train_dir / 'authentic'
    authentic_images = list(authentic_dir.glob('*.jpg'))[:1000]  # Use first 1000
    print(f"Processing {len(authentic_images)} authentic images...")
    
    for i, img_path in enumerate(authentic_images):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(authentic_images)} authentic images...")
        img = cv2.imread(str(img_path))
        if img is not None:
            features_dict = extractor.extract_features(img)
            # Extract just the numerical values
            features_list = [v for k, v in sorted(features_dict.items()) if isinstance(v, (int, float, np.number))]
            X.append(features_list)
            y.append(0)  # 0 = authentic
    
    print(f"Loaded {len([y_val for y_val in y if y_val == 0])} authentic images")
    
    # Load forged images
    forged_dir = train_dir / 'forged'
    forged_images = list(forged_dir.glob('*.jpg'))[:500]  # Balance dataset
    print(f"Processing {len(forged_images)} forged images...")
    
    for i, img_path in enumerate(forged_images):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(forged_images)} forged images...")
        img = cv2.imread(str(img_path))
        if img is not None:
            features_dict = extractor.extract_features(img)
            # Extract just the numerical values
            features_list = [v for k, v in sorted(features_dict.items()) if isinstance(v, (int, float, np.number))]
            X.append(features_list)
            y.append(1)  # 1 = forged
    
    print(f"Loaded {len([y_val for y_val in y if y_val == 1])} forged images")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nTotal features shape: {X.shape}")
    print(f"Total labels shape: {y.shape}")
    
    # Split the data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\nEvaluating model...")
    y_pred = rf.predict(X_test_scaled)
    y_pred_train = rf.predict(X_train_scaled)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    print("="*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Authentic', 'Forged']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    joblib.dump(rf, 'models/random_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("✅ Model saved successfully!")
    print("  - models/random_forest_model.joblib")
    print("  - models/scaler.joblib")
