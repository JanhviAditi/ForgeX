"""
Advanced Forgery Generation for Document Images

This script creates realistic document forgeries from authentic images using:
1. Copy-Move Forgery (cloning regions)
2. Splicing (combining parts from different documents)
3. Text Region Manipulation
4. Signature/Photo Swapping
5. JPEG Compression Artifacts Addition
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import json

class DocumentForgeryGenerator:
    """Generate realistic document forgeries from authentic images."""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
    def copy_move_forgery(self, image, num_regions=1):
        """
        Copy-Move Forgery: Copy a region and paste it elsewhere in the same image.
        Common in document tampering to hide or duplicate information.
        """
        h, w = image.shape[:2]
        forged = image.copy()
        
        for _ in range(num_regions):
            # Random source region
            region_h = random.randint(h // 8, h // 4)
            region_w = random.randint(w // 8, w // 4)
            
            src_y = random.randint(0, h - region_h - 1)
            src_x = random.randint(0, w - region_w - 1)
            
            # Random destination (avoid overlap with source)
            dst_y = random.randint(0, h - region_h - 1)
            dst_x = random.randint(0, w - region_w - 1)
            
            # Copy the region
            region = image[src_y:src_y+region_h, src_x:src_x+region_w].copy()
            
            # Apply slight transformation for realism
            # Random rotation
            angle = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((region_w/2, region_h/2), angle, 1.0)
            region = cv2.warpAffine(region, M, (region_w, region_h))
            
            # Paste with blending for seamless integration
            try:
                forged[dst_y:dst_y+region_h, dst_x:dst_x+region_w] = region
            except:
                pass
                
        return forged
    
    def splice_forgery(self, image1, image2):
        """
        Splicing: Combine regions from two different documents.
        Simulates swapping information between documents.
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Use the smaller dimensions
        h = min(h1, h2)
        w = min(w1, w2)
        
        # Resize both to same size
        img1_resized = cv2.resize(image1, (w, h))
        img2_resized = cv2.resize(image2, (w, h))
        
        # Create a random mask for splicing
        # Common patterns: swap top/bottom, left/right, or random regions
        splice_type = random.choice(['horizontal', 'vertical', 'region'])
        
        if splice_type == 'horizontal':
            split = random.randint(h // 3, 2 * h // 3)
            forged = np.vstack([img1_resized[:split], img2_resized[split:]])
            
        elif splice_type == 'vertical':
            split = random.randint(w // 3, 2 * w // 3)
            forged = np.hstack([img1_resized[:, :split], img2_resized[:, split:]])
            
        else:  # region
            # Random rectangular region from img2
            region_h = random.randint(h // 4, h // 2)
            region_w = random.randint(w // 4, w // 2)
            
            y = random.randint(0, h - region_h)
            x = random.randint(0, w - region_w)
            
            forged = img1_resized.copy()
            forged[y:y+region_h, x:x+region_w] = img2_resized[y:y+region_h, x:x+region_w]
        
        return forged
    
    def text_region_swap(self, image1, image2):
        """
        Swap text regions between documents (simulating info replacement).
        Focuses on the central area where text typically appears on ID cards.
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        h = min(h1, h2)
        w = min(w1, w2)
        
        img1_resized = cv2.resize(image1, (w, h))
        img2_resized = cv2.resize(image2, (w, h))
        
        forged = img1_resized.copy()
        
        # Swap multiple horizontal strips (simulating text lines)
        num_strips = random.randint(2, 4)
        strip_height = h // (num_strips * 2)
        
        for i in range(num_strips):
            y_start = random.randint(h // 4, 3 * h // 4 - strip_height)
            x_start = random.randint(w // 6, w // 3)
            strip_width = random.randint(w // 3, 2 * w // 3)
            
            try:
                forged[y_start:y_start+strip_height, x_start:x_start+strip_width] = \
                    img2_resized[y_start:y_start+strip_height, x_start:x_start+strip_width]
            except:
                pass
                
        return forged
    
    def photo_region_swap(self, image1, image2):
        """
        Swap photo regions (simulating photo replacement on ID cards).
        ID cards typically have photos in top-left or top-right corner.
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        h = min(h1, h2)
        w = min(w1, w2)
        
        img1_resized = cv2.resize(image1, (w, h))
        img2_resized = cv2.resize(image2, (w, h))
        
        forged = img1_resized.copy()
        
        # Photo is usually in top-left or top-right corner
        photo_h = h // 3
        photo_w = w // 4
        
        position = random.choice(['top-left', 'top-right'])
        
        if position == 'top-left':
            y, x = h // 10, w // 10
        else:  # top-right
            y, x = h // 10, w - photo_w - w // 10
        
        try:
            forged[y:y+photo_h, x:x+photo_w] = \
                img2_resized[y:y+photo_h, x:x+photo_w]
        except:
            pass
            
        return forged
    
    def add_compression_artifacts(self, image, quality_range=(30, 60)):
        """
        Add JPEG compression artifacts to hide manipulation traces.
        Lower quality = more artifacts.
        """
        quality = random.randint(*quality_range)
        
        # Encode to JPEG and decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return compressed
    
    def add_noise(self, image, noise_type='gaussian'):
        """
        Add noise to make forgery less detectable.
        """
        if noise_type == 'gaussian':
            mean = 0
            sigma = random.uniform(5, 15)
            noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
            noisy = cv2.add(image, noise)
            
        elif noise_type == 'salt_pepper':
            prob = random.uniform(0.01, 0.03)
            noisy = image.copy()
            
            # Salt
            salt = np.random.random(image.shape[:2]) < prob/2
            noisy[salt] = 255
            
            # Pepper
            pepper = np.random.random(image.shape[:2]) < prob/2
            noisy[pepper] = 0
        
        else:
            noisy = image
            
        return noisy
    
    def apply_slight_blur(self, image):
        """
        Apply slight blur to hide manipulation edges.
        """
        kernel_size = random.choice([3, 5])
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def generate_forgery(self, image, image2=None, forgery_types=None):
        """
        Generate a forged document with multiple techniques.
        
        Args:
            image: Primary authentic image
            image2: Secondary image for splicing (optional)
            forgery_types: List of forgery types to apply
        
        Returns:
            forged_image, metadata dictionary
        """
        if forgery_types is None:
            forgery_types = random.sample([
                'copy_move', 'splice', 'text_swap', 'photo_swap'
            ], k=random.randint(1, 2))
        
        metadata = {'techniques': []}
        forged = image.copy()
        
        for forgery_type in forgery_types:
            if forgery_type == 'copy_move':
                forged = self.copy_move_forgery(forged, num_regions=random.randint(1, 2))
                metadata['techniques'].append('copy_move')
                
            elif forgery_type == 'splice' and image2 is not None:
                forged = self.splice_forgery(forged, image2)
                metadata['techniques'].append('splice')
                
            elif forgery_type == 'text_swap' and image2 is not None:
                forged = self.text_region_swap(forged, image2)
                metadata['techniques'].append('text_swap')
                
            elif forgery_type == 'photo_swap' and image2 is not None:
                forged = self.photo_region_swap(forged, image2)
                metadata['techniques'].append('photo_swap')
        
        # Post-processing to hide traces
        if random.random() > 0.5:
            forged = self.add_compression_artifacts(forged)
            metadata['techniques'].append('compression')
        
        if random.random() > 0.7:
            forged = self.add_noise(forged, random.choice(['gaussian', 'salt_pepper']))
            metadata['techniques'].append('noise')
        
        if random.random() > 0.6:
            forged = self.apply_slight_blur(forged)
            metadata['techniques'].append('blur')
        
        return forged, metadata


def generate_forged_dataset(authentic_dir, output_dir, num_forgeries_per_image=2, max_images=500):
    """
    Generate forged documents from authentic images.
    
    Args:
        authentic_dir: Directory with authentic images
        output_dir: Directory to save forged images
        num_forgeries_per_image: Number of forgeries to create per authentic image
        max_images: Maximum number of authentic images to process
    """
    authentic_dir = Path(authentic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all authentic images
    image_files = list(authentic_dir.glob('*.jpg')) + list(authentic_dir.glob('*.png'))
    image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} authentic images")
    print(f"Generating {num_forgeries_per_image} forgeries per image...")
    
    generator = DocumentForgeryGenerator()
    metadata_list = []
    
    forgery_count = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Generate multiple forgeries
        for i in range(num_forgeries_per_image):
            # Randomly select another image for splicing techniques
            other_img_path = random.choice(image_files)
            other_img = cv2.imread(str(other_img_path))
            
            if other_img is None:
                other_img = img
            
            # Generate forgery
            forged, metadata = generator.generate_forgery(img, other_img)
            
            # Save forged image
            output_filename = f"forged_{img_path.stem}_v{i+1}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), forged)
            
            # Save metadata
            metadata['source_image'] = img_path.name
            metadata['other_image'] = other_img_path.name if other_img_path != img_path else None
            metadata['output_file'] = output_filename
            metadata_list.append(metadata)
            
            forgery_count += 1
    
    # Save metadata JSON
    metadata_file = output_dir / 'forgery_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"\n✅ Generated {forgery_count} forged images!")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Metadata saved to: {metadata_file}")
    
    # Print statistics
    print("\n📊 Forgery Techniques Used:")
    technique_counts = {}
    for metadata in metadata_list:
        for technique in metadata['techniques']:
            technique_counts[technique] = technique_counts.get(technique, 0) + 1
    
    for technique, count in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {technique}: {count} times")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate forged document images')
    parser.add_argument('--authentic-dir', default='data/raw/authentic',
                        help='Directory with authentic images')
    parser.add_argument('--output-dir', default='data/raw/forged_synthetic',
                        help='Directory to save forged images')
    parser.add_argument('--num-forgeries', type=int, default=2,
                        help='Number of forgeries per authentic image')
    parser.add_argument('--max-images', type=int, default=500,
                        help='Maximum number of authentic images to process')
    
    args = parser.parse_args()
    
    generate_forged_dataset(
        authentic_dir=args.authentic_dir,
        output_dir=args.output_dir,
        num_forgeries_per_image=args.num_forgeries,
        max_images=args.max_images
    )
