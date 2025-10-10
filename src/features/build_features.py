# -*- coding: utf-8 -*-
"""
Feature extraction module for document forgery detection.

This module contains functions to extract various features from document images
that can be used to detect forgery, including texture features, edge features,
statistical features, and frequency domain features.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

# Optional imports with fallbacks
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    ndimage = None

try:
    from skimage import feature, measure, filters
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    feature = measure = filters = graycomatrix = graycoprops = None


class DocumentFeatureExtractor:
    """
    A comprehensive feature extractor for document forgery detection.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the feature extractor.
        
        Args:
            image_size: Target size for image preprocessing
        """
        self.image_size = image_size
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, image: np.ndarray) -> Dict:
        """
        Extract available features from a numpy array image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Always extract statistical features (no special libraries needed)
        features.update(self.extract_statistical_features(image))
        
        # Extract advanced features if libraries are available
        if HAS_OPENCV and len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3:
            # Fallback: simple RGB to grayscale conversion
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray = image
        
        if HAS_SKIMAGE:
            features.update(self.extract_texture_features(gray))
        
        if HAS_OPENCV:
            features.update(self.extract_edge_features(gray))
        
        return features

    def extract_all_features(self, image_path: str) -> Dict:
        """
        Extract all available features from an image file.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing all extracted features
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for loading images from files. "
                            "Install with: pip install opencv-python")
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return {}
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            return self.extract_features(image)
            features.update(self.extract_statistical_features(gray))
            features.update(self.extract_frequency_features(gray))
            features.update(self.extract_local_binary_pattern_features(gray))
            features.update(self.extract_glcm_features(gray))
            features.update(self.extract_noise_features(gray))
            features.update(self.extract_compression_artifacts(image))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return {}
    
    def extract_texture_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract texture-based features that can indicate image manipulation.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing texture features
        """
        features = {}
        
        # Gabor filter responses
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            for frequency in [0.1, 0.3, 0.5]:  # Different frequencies
                real, _ = filters.gabor(gray_image, frequency=frequency, 
                                      theta=np.deg2rad(theta))
                gabor_responses.append(real)
        
        # Statistics of Gabor responses
        for i, response in enumerate(gabor_responses):
            features[f'gabor_mean_{i}'] = np.mean(response)
            features[f'gabor_std_{i}'] = np.std(response)
            features[f'gabor_energy_{i}'] = np.sum(response**2)
        
        # Entropy
        features['entropy'] = measure.shannon_entropy(gray_image)
        
        return features
    
    def extract_edge_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract edge-based features for forgery detection.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing edge features
        """
        features = {}
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['sobel_mean'] = np.mean(sobel_magnitude)
        features['sobel_std'] = np.std(sobel_magnitude)
        features['sobel_max'] = np.max(sobel_magnitude)
        
        # Canny edge detection
        canny_edges = cv2.Canny(gray_image, 50, 150)
        features['canny_edge_density'] = np.sum(canny_edges > 0) / canny_edges.size
        
        # Laplacian
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        features['laplacian_mean'] = np.mean(np.abs(laplacian))
        
        return features
    
    def extract_statistical_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract statistical features from the image.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing statistical features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(gray_image)
        features['std'] = np.std(gray_image)
        features['var'] = np.var(gray_image)
        features['skewness'] = self._calculate_skewness(gray_image)
        features['kurtosis'] = self._calculate_kurtosis(gray_image)
        features['min'] = np.min(gray_image)
        features['max'] = np.max(gray_image)
        features['range'] = np.max(gray_image) - np.min(gray_image)
        
        # Histogram features
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        features['hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
        features['hist_peak'] = np.max(hist)
        features['hist_uniformity'] = np.sum(hist**2)
        
        # Percentiles
        for p in [25, 50, 75, 95]:
            features[f'percentile_{p}'] = np.percentile(gray_image, p)
        
        return features
    
    def extract_frequency_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract frequency domain features using FFT.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing frequency features
        """
        features = {}
        
        # 2D FFT
        fft = np.fft.fft2(gray_image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        
        # Statistics of magnitude spectrum
        features['fft_mean'] = np.mean(magnitude_spectrum)
        features['fft_std'] = np.std(magnitude_spectrum)
        features['fft_energy'] = np.sum(magnitude_spectrum**2)
        
        # DCT features
        dct = cv2.dct(gray_image.astype(np.float32))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        
        # Low and high frequency energy
        h, w = dct.shape
        low_freq = dct[:h//4, :w//4]
        high_freq = dct[3*h//4:, 3*w//4:]
        
        features['low_freq_energy'] = np.sum(low_freq**2)
        features['high_freq_energy'] = np.sum(high_freq**2)
        features['freq_energy_ratio'] = features['high_freq_energy'] / (features['low_freq_energy'] + 1e-10)
        
        return features
    
    def extract_local_binary_pattern_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract Local Binary Pattern (LBP) features.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing LBP features
        """
        features = {}
        
        # LBP with different parameters
        radius = 3
        n_points = 8 * radius
        
        lbp = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-10)  # Normalize
        
        # LBP statistics
        features['lbp_uniformity'] = np.sum(lbp_hist**2)
        features['lbp_entropy'] = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        features['lbp_contrast'] = np.var(lbp)
        
        # Top LBP patterns
        top_patterns = np.argsort(lbp_hist)[-5:]
        for i, pattern in enumerate(top_patterns):
            features[f'lbp_top_pattern_{i}'] = lbp_hist[pattern]
        
        return features
    
    def extract_glcm_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract Gray Level Co-occurrence Matrix (GLCM) features.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing GLCM features
        """
        features = {}
        
        # Reduce gray levels for GLCM computation
        gray_reduced = (gray_image // 32).astype(np.uint8)  # 8 gray levels
        
        # Compute GLCM for different angles and distances
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray_reduced, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # Extract GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(prop_values)
            features[f'glcm_{prop}_std'] = np.std(prop_values)
        
        return features
    
    def extract_noise_features(self, gray_image: np.ndarray) -> Dict:
        """
        Extract features related to image noise that might indicate manipulation.
        
        Args:
            gray_image: Grayscale image array
            
        Returns:
            Dictionary containing noise-related features
        """
        features = {}
        
        # Noise estimation using Laplacian
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        features['noise_laplacian'] = np.var(laplacian)
        
        # Wavelet-based noise estimation
        try:
            from pywt import dwt2, wavedec2
            
            # Single-level DWT
            cA, (cH, cV, cD) = dwt2(gray_image, 'db4')
            
            # Noise estimation from high-frequency coefficients
            sigma = np.median(np.abs(cD)) / 0.6745
            features['noise_wavelet'] = sigma
            
            # Multi-level wavelet decomposition
            coeffs = wavedec2(gray_image, 'db4', level=3)
            detail_coeffs = [coeffs[i] for i in range(1, len(coeffs))]
            
            for level, coeff_tuple in enumerate(detail_coeffs):
                if isinstance(coeff_tuple, tuple):
                    for i, coeff in enumerate(coeff_tuple):
                        features[f'wavelet_level_{level}_detail_{i}_std'] = np.std(coeff)
                        
        except ImportError:
            self.logger.warning("PyWavelets not available, skipping wavelet noise features")
        
        # Gradient-based noise features
        grad_x = np.gradient(gray_image, axis=1)
        grad_y = np.gradient(gray_image, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_noise'] = np.std(grad_magnitude)
        
        return features
    
    def extract_compression_artifacts(self, color_image: np.ndarray) -> Dict:
        """
        Extract features related to compression artifacts that might indicate forgery.
        
        Args:
            color_image: Color image array (BGR)
            
        Returns:
            Dictionary containing compression artifact features
        """
        features = {}
        
        # Convert to YUV color space
        yuv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2YUV)
        
        # Blocking artifacts detection in Y channel
        y_channel = yuv_image[:, :, 0].astype(np.float32)
        
        # 8x8 block analysis (typical JPEG block size)
        h, w = y_channel.shape
        block_size = 8
        
        horizontal_diffs = []
        vertical_diffs = []
        
        # Analyze block boundaries
        for i in range(block_size, h, block_size):
            if i < h:
                diff = np.mean(np.abs(y_channel[i, :] - y_channel[i-1, :]))
                horizontal_diffs.append(diff)
        
        for j in range(block_size, w, block_size):
            if j < w:
                diff = np.mean(np.abs(y_channel[:, j] - y_channel[:, j-1]))
                vertical_diffs.append(diff)
        
        features['block_horizontal_diff'] = np.mean(horizontal_diffs) if horizontal_diffs else 0
        features['block_vertical_diff'] = np.mean(vertical_diffs) if vertical_diffs else 0
        features['block_diff_ratio'] = (features['block_horizontal_diff'] / 
                                       (features['block_vertical_diff'] + 1e-10))
        
        # Color channel inconsistencies
        b, g, r = cv2.split(color_image)
        features['channel_correlation_bg'] = np.corrcoef(b.flatten(), g.flatten())[0, 1]
        features['channel_correlation_br'] = np.corrcoef(b.flatten(), r.flatten())[0, 1]
        features['channel_correlation_gr'] = np.corrcoef(g.flatten(), r.flatten())[0, 1]
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0


def extract_features_from_directory(input_dir: str, output_file: str, 
                                   image_size: Tuple[int, int] = (224, 224)) -> None:
    """
    Extract features from all images in a directory and save to CSV.
    
    Args:
        input_dir: Directory containing images
        output_file: Path to output CSV file
        image_size: Target image size for processing
    """
    extractor = DocumentFeatureExtractor(image_size)
    
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'**/*{ext}'))
        image_files.extend(input_path.glob(f'**/*{ext.upper()}'))
    
    if not image_files:
        logging.warning(f"No image files found in {input_dir}")
        return
    
    # Extract features from all images
    all_features = []
    
    for image_file in image_files:
        logging.info(f"Processing {image_file}")
        
        features = extractor.extract_all_features(str(image_file))
        if features:
            features['filename'] = image_file.name
            features['filepath'] = str(image_file)
            all_features.append(features)
    
    # Save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        logging.info(f"Features saved to {output_file}")
        logging.info(f"Extracted {len(df.columns)-2} features from {len(df)} images")
    else:
        logging.error("No features were extracted")


def main():
    """Main function for testing feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from document images')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('output_file', help='Output CSV file for features')
    parser.add_argument('--image-size', nargs=2, type=int, default=[224, 224],
                       help='Target image size (width height)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    extract_features_from_directory(
        args.input_dir, 
        args.output_file, 
        tuple(args.image_size)
    )


if __name__ == '__main__':
    main()
