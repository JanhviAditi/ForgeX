# -*- coding: utf-8 -*-
import click
import logging
import os
import shutil
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Optional imports
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for document forgery detection.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (width, height)
    
    Returns:
        np.ndarray: Preprocessed image array
    """
    try:
        # Load image using PIL
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image) / 255.0
        
        return image_array
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None


def extract_image_metadata(image_path):
    """
    Extract metadata from image that might be useful for forgery detection.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        dict: Dictionary containing image metadata
    """
    metadata = {
        'filename': os.path.basename(image_path),
        'file_size': os.path.getsize(image_path),
        'width': 0,
        'height': 0,
        'channels': 0,
        'format': '',
        'mode': ''
    }
    
    try:
        with Image.open(image_path) as img:
            metadata['width'] = img.width
            metadata['height'] = img.height
            metadata['format'] = img.format
            metadata['mode'] = img.mode
            
        # Load with OpenCV to get channels
        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            metadata['channels'] = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
            
    except Exception as e:
        logging.error(f"Error extracting metadata from {image_path}: {str(e)}")
    
    return metadata


def create_dataset_structure(input_dir, output_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Create a structured dataset from raw images.
    
    Args:
        input_dir (str): Directory containing raw images
        output_dir (str): Directory to save processed dataset
        train_split (float): Proportion of data for training
        val_split (float): Proportion of data for validation
        test_split (float): Proportion of data for testing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['authentic', 'forged']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'**/*{ext}'))
        image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
    
    if not image_files:
        logging.warning(f"No image files found in {input_dir}")
        return
    
    # Split files
    np.random.shuffle(image_files)
    n_files = len(image_files)
    
    train_end = int(train_split * n_files)
    val_end = train_end + int(val_split * n_files)
    
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }
    
    metadata_list = []
    
    for split_name, files in splits.items():
        logging.info(f"Processing {split_name} split with {len(files)} images")
        
        for file_path in tqdm(files, desc=f"Processing {split_name}"):
            # Determine class based on filename or directory structure
            # This is a placeholder - you'll need to implement your own logic
            # based on how your data is organized
            class_name = determine_class(file_path)
            
            # Load and preprocess image
            image_array = load_image(str(file_path))
            if image_array is None:
                continue
            
            # Extract metadata
            metadata = extract_image_metadata(str(file_path))
            metadata['split'] = split_name
            metadata['class'] = class_name
            metadata['original_path'] = str(file_path)
            
            # Save processed image
            output_file = output_path / split_name / class_name / file_path.name
            
            # Convert back to PIL Image and save
            image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
            image_pil.save(output_file)
            
            metadata['processed_path'] = str(output_file)
            metadata_list.append(metadata)
    
    # Save metadata CSV
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_path / 'dataset_metadata.csv', index=False)
    logging.info(f"Dataset created successfully in {output_dir}")
    logging.info(f"Total images processed: {len(metadata_list)}")


def determine_class(file_path):
    """
    Determine the class (authentic/forged) of an image based on its path or filename.
    
    This is a placeholder function that you should customize based on your data organization.
    
    Args:
        file_path (Path): Path to the image file
    
    Returns:
        str: Class name ('authentic' or 'forged')
    """
    # Example logic - customize based on your data structure
    file_str = str(file_path).lower()
    
    if any(keyword in file_str for keyword in ['fake', 'forged', 'manipulated', 'tampered']):
        return 'forged'
    elif any(keyword in file_str for keyword in ['authentic', 'original', 'real', 'genuine']):
        return 'authentic'
    else:
        # Default classification - you may want to handle this differently
        return 'authentic'


def augment_images(input_dir, output_dir, augmentation_factor=2):
    """
    Apply data augmentation to increase dataset size.
    
    Args:
        input_dir (str): Directory containing images to augment
        output_dir (str): Directory to save augmented images
        augmentation_factor (int): Number of augmented versions per original image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'**/*{ext}'))
        image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
    
    for file_path in tqdm(image_files, desc="Augmenting images"):
        # Load original image
        image = cv2.imread(str(file_path))
        if image is None:
            continue
        
        # Create augmented versions
        for i in range(augmentation_factor):
            augmented = apply_augmentation(image)
            
            # Save augmented image
            base_name = file_path.stem
            extension = file_path.suffix
            output_file = output_path / f"{base_name}_aug_{i}{extension}"
            
            cv2.imwrite(str(output_file), augmented)


def apply_augmentation(image):
    """
    Apply random augmentation to an image.
    
    Args:
        image (np.ndarray): Input image
    
    Returns:
        np.ndarray: Augmented image
    """
    # Random rotation
    angle = np.random.uniform(-15, 15)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Random noise
    if np.random.random() < 0.3:
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    
    # Random blur
    if np.random.random() < 0.2:
        kernel_size = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return image


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.option('--augment', '-a', is_flag=True, help='Apply data augmentation')
@click.option('--train-split', default=0.7, help='Training split ratio')
@click.option('--val-split', default=0.2, help='Validation split ratio')
@click.option('--test-split', default=0.1, help='Test split ratio')
def main(input_filepath, output_filepath, augment, train_split, val_split, test_split):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test splits must sum to 1.0")
    
    # Create structured dataset
    create_dataset_structure(
        input_filepath, 
        output_filepath,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split
    )
    
    # Apply augmentation if requested
    if augment:
        logger.info('Applying data augmentation')
        for split in ['train']:  # Usually only augment training data
            split_dir = Path(output_filepath) / split
            augment_dir = Path(output_filepath) / f"{split}_augmented"
            augment_images(str(split_dir), str(augment_dir))
    
    logger.info('Data processing completed successfully')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
