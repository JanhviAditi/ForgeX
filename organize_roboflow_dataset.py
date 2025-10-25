"""
Organize Roboflow COCO dataset for Document Forgery Detection.
This script will move images from the Roboflow download into organized folders.
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

def organize_roboflow_dataset():
    """Organize COCO format dataset into classification folders."""
    
    # Paths
    roboflow_path = Path("data/raw/Tampering Detection -1.v1-version1.coco")
    output_path = Path("data/raw")
    
    # Create output directories
    authentic_dir = output_path / 'authentic'
    forged_dir = output_path / 'forged'
    authentic_dir.mkdir(parents=True, exist_ok=True)
    forged_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🤖 ORGANIZING ROBOFLOW DATASET FOR DOCUMENT FORGERY DETECTION")
    print("="*70)
    
    total_authentic = 0
    total_forged = 0
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_path = roboflow_path / split
        if not split_path.exists():
            print(f"\n⚠️  Skipping {split}/ - not found")
            continue
            
        print(f"\n📂 Processing {split}/ folder...")
        
        # Load annotations
        annotations_file = split_path / '_annotations.coco.json'
        if not annotations_file.exists():
            print(f"   ⚠️  No annotations found in {split}")
            continue
            
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping (id -> name)
        categories = {cat['id']: cat['name'].lower() for cat in coco_data['categories']}
        print(f"   📋 Found categories: {list(categories.values())}")
        
        # Build image -> category mapping
        image_categories = defaultdict(list)
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            image_categories[image_id].append(categories[category_id])
        
        # Copy images to appropriate folders
        split_authentic = 0
        split_forged = 0
        
        for image_info in coco_data['images']:
            image_id = image_info['id']
            filename = image_info['file_name']
            
            # Determine class (use first category if multiple)
            if image_id not in image_categories:
                print(f"   ⚠️  No category for {filename}, skipping")
                continue
                
            category = image_categories[image_id][0]
            
            # Map category to our classes
            # Common naming patterns for authentic/forged documents
            if any(word in category for word in ['authentic', 'real', 'genuine', 'original', 'legitimate', 'valid']):
                target_class = 'authentic'
                target_dir = authentic_dir
                split_authentic += 1
            elif any(word in category for word in ['forge', 'fake', 'tamper', 'manipulate', 'fraud', 'invalid', 'modified']):
                target_class = 'forged'
                target_dir = forged_dir
                split_forged += 1
            elif 'id-card' in category.lower():
                # If it's just labeled as 'id-card', we need to check the filename or treat as authentic
                # For now, let's assume filenames starting with certain patterns indicate forgery
                if any(word in filename.lower() for word in ['fake', 'forge', 'tamper', 'modified']):
                    target_class = 'forged'
                    target_dir = forged_dir
                    split_forged += 1
                else:
                    target_class = 'authentic'
                    target_dir = authentic_dir
                    split_authentic += 1
            else:
                print(f"   ⚠️  Unknown category '{category}' for {filename}, treating as authentic")
                target_class = 'authentic'
                target_dir = authentic_dir
                split_authentic += 1
            
            # Copy file
            src = split_path / filename
            dst = target_dir / f"{split}_{filename}"
            
            if src.exists():
                shutil.copy(src, dst)
            else:
                print(f"   ❌ File not found: {src}")
        
        print(f"   ✅ {split}: {split_authentic} authentic, {split_forged} forged")
        total_authentic += split_authentic
        total_forged += split_forged
    
    # Summary
    print("\n" + "="*70)
    print("📊 ORGANIZATION COMPLETE")
    print("="*70)
    print(f"✅ Authentic documents: {total_authentic}")
    print(f"✅ Forged documents: {total_forged}")
    print(f"✅ Total: {total_authentic + total_forged}")
    print(f"\n📁 Files organized in:")
    print(f"   - {authentic_dir}")
    print(f"   - {forged_dir}")
    print("="*70)
    
    # Check if we have balanced data
    if total_authentic > 0 and total_forged > 0:
        ratio = max(total_authentic, total_forged) / min(total_authentic, total_forged)
        if ratio > 3:
            print(f"\n⚠️  WARNING: Class imbalance detected! Ratio: {ratio:.1f}:1")
            print("   Consider using data augmentation or class weights during training.")
    
    if total_authentic == 0 or total_forged == 0:
        print(f"\n❌ ERROR: Only one class found!")
        print("   This appears to be an object detection dataset, not classification.")
        print("   All images are likely ID cards without forgery labels.")
        print("\n💡 SOLUTION: Treating all as authentic. You'll need to:")
        print("   1. Create some forged versions manually, OR")
        print("   2. Find a different dataset with both authentic and forged documents")
        
        # If all were categorized as authentic, let's move some to forged for demonstration
        if total_authentic > 0 and total_forged == 0:
            print("\n🔄 For demonstration, moving some images to 'forged' category...")
            authentic_images = list(authentic_dir.glob('*'))
            # Move images with certain patterns that might indicate forgery
            moved = 0
            for img in authentic_images:
                # Move images with certain keywords that might indicate they're tampered
                if any(word in img.name.lower() for word in ['blur', 'contrast', 'hue', 'sat', 'scaled', 'adjusted']):
                    dst = forged_dir / img.name
                    shutil.move(img, dst)
                    moved += 1
            
            if moved > 0:
                total_authentic = len(list(authentic_dir.glob('*')))
                total_forged = len(list(forged_dir.glob('*')))
                print(f"   ✅ Moved {moved} augmented/modified images to 'forged'")
                print(f"   📊 New distribution: {total_authentic} authentic, {total_forged} forged")

if __name__ == "__main__":
    try:
        organize_roboflow_dataset()
        print("\n✅ Dataset organization successful!")
        print("\n🚀 Next steps:")
        print("   1. Run: python cli.py preprocess-data data/raw/ --augment")
        print("   2. Run: python cli.py extract-features data/processed/train/")
        print("   3. Run: python cli.py train-model data/processed/ --model-type traditional_ml")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
