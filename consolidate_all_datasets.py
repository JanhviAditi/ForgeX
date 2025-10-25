"""
Consolidate All Aadhaar Card Datasets
======================================
This script combines data from:
1. Original dataset (data/raw/authentic/)
2. Archive dataset (data/raw/archive/)
3. Detection Dataset (data/raw/Detection Dataset/)

Output: Unified dataset in data/consolidated/
"""

import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

print("="*70)
print("🔄 CONSOLIDATING ALL DATASETS")
print("="*70)

# Define source directories
sources = {
    'original_authentic': Path('data/raw/authentic'),
    'archive_train': Path('data/raw/archive/train/images'),
    'archive_valid': Path('data/raw/archive/valid/images'),
    'archive_test': Path('data/raw/archive/test/images'),
    'detection_train': Path('data/raw/Detection Dataset/dataset/dataset/images/train'),
    'detection_val': Path('data/raw/Detection Dataset/dataset/dataset/images/val'),
}

# Define output directory
output_authentic = Path('data/consolidated/authentic')
output_forged = Path('data/consolidated/forged')

# Create output directories
output_authentic.mkdir(parents=True, exist_ok=True)
output_forged.mkdir(parents=True, exist_ok=True)

print("\n📊 Scanning datasets...")

# Count images in each source
total_count = 0
source_counts = {}

for name, path in sources.items():
    if path.exists():
        # Get all image files
        images = list(path.glob('*.jpg')) + list(path.glob('*.jpeg')) + \
                 list(path.glob('*.png')) + list(path.glob('*.bmp'))
        count = len(images)
        source_counts[name] = count
        total_count += count
        print(f"   {name}: {count} images")
    else:
        print(f"   ⚠️  {name}: Not found")
        source_counts[name] = 0

print(f"\n✅ Total authentic images found: {total_count}")

# Copy all authentic images
print("\n📂 Consolidating authentic images...")
copied = 0
skipped = 0
errors = 0

for source_name, source_path in tqdm(sources.items(), desc="Processing sources"):
    if not source_path.exists():
        continue
    
    # Get all image files
    images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.jpeg')) + \
             list(source_path.glob('*.png')) + list(source_path.glob('*.bmp'))
    
    for img_path in tqdm(images, desc=f"  {source_name}", leave=False):
        try:
            # Verify it's a valid image
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            
            # Create unique filename: source_originalname
            new_name = f"{source_name}_{img_path.name}"
            dest_path = output_authentic / new_name
            
            # Copy if not already exists
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                copied += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path}: {e}")
            errors += 1

print(f"\n✅ Consolidation complete!")
print(f"   Copied: {copied} images")
print(f"   Skipped: {skipped} images")
print(f"   Errors: {errors} images")

# Now copy existing forged synthetic images
print("\n📂 Copying synthetic forged images...")
forged_source = Path('data/raw/forged_synthetic')

if forged_source.exists():
    forged_images = list(forged_source.glob('*.jpg')) + list(forged_source.glob('*.jpeg')) + \
                    list(forged_source.glob('*.png')) + list(forged_source.glob('*.bmp'))
    
    forged_copied = 0
    for img_path in tqdm(forged_images, desc="Copying forged"):
        try:
            dest_path = output_forged / img_path.name
            if not dest_path.exists():
                shutil.copy2(img_path, dest_path)
                forged_copied += 1
        except Exception as e:
            print(f"⚠️  Error: {e}")
    
    print(f"✅ Copied {forged_copied} forged images")
else:
    print("⚠️  No existing forged images found")

print("\n" + "="*70)
print("📊 FINAL DATASET SUMMARY")
print("="*70)

# Count final images
final_authentic = len(list(output_authentic.glob('*.jpg'))) + \
                  len(list(output_authentic.glob('*.jpeg'))) + \
                  len(list(output_authentic.glob('*.png'))) + \
                  len(list(output_authentic.glob('*.bmp')))

final_forged = len(list(output_forged.glob('*.jpg'))) + \
               len(list(output_forged.glob('*.jpeg'))) + \
               len(list(output_forged.glob('*.png'))) + \
               len(list(output_forged.glob('*.bmp')))

print(f"\n✅ Authentic images: {final_authentic}")
print(f"✅ Forged images: {final_forged}")
print(f"✅ Total images: {final_authentic + final_forged}")

print("\n📁 Output location: data/consolidated/")
print("   - data/consolidated/authentic/")
print("   - data/consolidated/forged/")

# Calculate how many more forgeries we need
if final_forged < final_authentic:
    needed = final_authentic - final_forged
    print(f"\n💡 Recommendation: Generate {needed} more forged images to balance the dataset")
    print(f"   Run: python generate_forgeries.py --authentic-dir data/consolidated/authentic --output-dir data/consolidated/forged --num-forgeries 3 --max-images {needed//3 + 100}")
else:
    print("\n✅ Dataset is balanced!")

print("\n" + "="*70)
