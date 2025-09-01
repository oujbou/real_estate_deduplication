"""
Image Feature Extractor for Real Estate Duplicate Detection

This module provides production-ready image feature extraction for comparing
real estate property photos using ResNet50-based visual similarity.

Features extracted:
1. max_image_similarity - Highest similarity between any image pair (0-1 scale)
2. avg_image_similarity - Average similarity across all image pairs (0-1 scale)

Uses pre-trained ResNet50 for robust visual feature extraction with MPS acceleration support.

Author: Real Estate ML Team
Date: 2025
"""

import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ImageFeatureExtractor:
    """
    Production-ready extractor for image-based real estate features.
    
    Extracts 2 features by comparing property images using ResNet50:
    - Maximum similarity between any image pair
    - Average similarity across all image pairs
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 scaler_path: Optional[str] = None,
                 use_mps: bool = True):
        """
        Initialize the image feature extractor.
        
        Args:
            model_path: Path to saved ResNet50 model. If None, uses pre-trained.
            scaler_path: Path to the fitted StandardScaler. If None, uses default location.
            use_mps: Whether to use MPS acceleration on Apple Silicon (M1/M2/M4).
        """
        self.feature_names = ['max_image_similarity', 'avg_image_similarity']
        
        # Set device for optimal performance
        if use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using MPS acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA acceleration")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU (no GPU acceleration available)")
        
        # Initialize ResNet50 feature extractor
        self._initialize_image_model(model_path)
        
        # Load the fitted scaler
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent / "models" / "image_scaler.pkl"
        
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded image scaler from {scaler_path}")
        except FileNotFoundError:
            logger.warning(f"Image scaler not found at {scaler_path}. Features will not be scaled.")
            self.scaler = None
        except Exception as e:
            logger.error(f"Error loading image scaler: {e}")
            self.scaler = None
        
        # Image preprocessing pipeline (ResNet50 standard)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _initialize_image_model(self, model_path: Optional[str]):
        """Initialize ResNet50 feature extractor."""
        try:
            # Load pre-trained ResNet50
            base_model = models.resnet50(pretrained=True)
            
            # Remove final classification layer to get features
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            
            # Load custom weights if provided
            if model_path is not None:
                state_dict = torch.load(model_path, map_location=self.device)
                self.feature_extractor.load_state_dict(state_dict)
                logger.info(f"Loaded custom model weights from {model_path}")
            else:
                logger.info("Using pre-trained ResNet50 weights")
            
            # Set to evaluation mode and move to device
            self.feature_extractor.eval()
            self.feature_extractor = self.feature_extractor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error initializing image model: {e}")
            raise
    
    def extract_features(self, images_A: List[Union[str, Path]], 
                        images_B: List[Union[str, Path]]) -> Dict[str, Union[np.ndarray, float]]:
        """
        Extract image features comparing two sets of property images.
        
        Args:
            images_A: List of image paths for first property
            images_B: List of image paths for second property
            
        Returns:
            Dictionary with 'scaled_features' array and individual feature values
        """
        try:
            raw_features = self._extract_raw_features(images_A, images_B)
            
            if self.scaler is not None:
                # Scale features using fitted scaler
                features_array = np.array(raw_features['features']).reshape(1, -1)
                scaled_features = self.scaler.transform(features_array)
                
                return {
                    'scaled_features': np.array(raw_features['features']),
                    **raw_features
                }
                
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            # Return zero vector as fallback
            zero_features = np.zeros(len(self.feature_names))
            return {
                'scaled_features': zero_features,
                'max_similarity': 0.0,
                'avg_similarity': 0.0
            }
    
    def _extract_raw_features(self, images_A: List[Union[str, Path]], 
                             images_B: List[Union[str, Path]]) -> Dict:
        """
        Extract raw (unscaled) image features.
        
        Args:
            images_A: First property image paths
            images_B: Second property image paths
            
        Returns:
            Dictionary with feature values and array
        """
        # Extract embeddings for both image sets
        embeddings_A = self._extract_embeddings(images_A)
        embeddings_B = self._extract_embeddings(images_B)
        
        if len(embeddings_A) == 0 or len(embeddings_B) == 0:
            # No images available for comparison
            max_sim = 0.0
            avg_sim = 0.0
        else:
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(embeddings_A, embeddings_B)
            max_sim = float(np.max(similarity_matrix))
            avg_sim = float(np.mean(similarity_matrix))
        
        features = [max_sim, avg_sim]
        
        return {
            'features': features,
            'max_similarity': max_sim,
            'avg_similarity': avg_sim
        }
    
    def _extract_embeddings(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract ResNet50 embeddings from a list of image paths.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Array of image embeddings (n_images, 2048)
        """
        if not image_paths:
            return np.array([])
        
        embeddings = []
        
        for path in image_paths:
            try:
                embedding = self._extract_single_embedding(path)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process image {path}: {e}")
                continue
        
        return np.array(embeddings) if embeddings else np.array([])
    
    def _extract_single_embedding(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Extract embedding from a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding array or None if failed
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                features = features.view(features.size(0), -1)  # Flatten
                
            return features.cpu().numpy()[0]
            
        except FileNotFoundError:
            logger.warning(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {e}")
            return None
    
    def batch_extract_embeddings(self, image_paths: List[Union[str, Path]], 
                                batch_size: int = 16) -> np.ndarray:
        """
        Extract embeddings from multiple images in batches for efficiency.
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once
            
        Returns:
            Array of image embeddings
        """
        if not image_paths:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Load batch of images
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.image_transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    continue
            
            if batch_tensors:
                # Stack into batch tensor and process
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                with torch.no_grad():
                    features = self.feature_extractor(batch_tensor)
                    features = features.view(features.size(0), -1)
                    all_embeddings.extend(features.cpu().numpy())
        
        return np.array(all_embeddings) if all_embeddings else np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names.copy()
    
    def get_feature_explanation(self, result: Dict) -> Dict[str, str]:
        """
        Get human-readable explanation of extracted features.
        
        Args:
            result: Result dictionary from extract_features()
            
        Returns:
            Dictionary with feature explanations
        """
        explanations = {
            'device_used': str(self.device),
            'max_similarity': f"Best matching image pair similarity: {result['max_similarity']:.3f}",
            'avg_similarity': f"Average image similarity: {result['avg_similarity']:.3f}",
            'interpretation': self._interpret_similarity(result['max_similarity'])
        }
        
        return explanations
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Provide business interpretation of similarity score."""
        if similarity >= 0.9:
            return "Very high similarity - likely same property"
        elif similarity >= 0.8:
            return "High similarity - strong duplicate candidate"
        elif similarity >= 0.6:
            return "Moderate similarity - possible duplicate"
        elif similarity >= 0.3:
            return "Low similarity - different properties"
        else:
            return "Very low similarity - clearly different properties"


# Convenience function for quick feature extraction
def extract_image_features(images_A: List[Union[str, Path]], 
                          images_B: List[Union[str, Path]],
                          model_path: Optional[str] = None,
                          scaler_path: Optional[str] = None,
                          use_mps: bool = True) -> Dict:
    """
    Convenience function to extract image features from two property image sets.
    
    Args:
        images_A: First property image paths
        images_B: Second property image paths
        model_path: Path to saved model (optional)
        scaler_path: Path to fitted scaler (optional)
        use_mps: Whether to use MPS acceleration
        
    Returns:
        Dictionary with scaled features and individual values
    """
    extractor = ImageFeatureExtractor(model_path, scaler_path, use_mps)
    return extractor.extract_features(images_A, images_B)


def get_listing_image_paths(listing_id: Union[str, int], 
                           pictures_dir: Union[str, Path],
                           max_images: int = 5) -> List[Path]:
    """
    Get all image paths for a specific listing.
    
    Args:
        listing_id: ID of the listing
        pictures_dir: Directory containing images
        max_images: Maximum number of images to return
        
    Returns:
        List of image paths for the listing
    """
    pictures_path = Path(pictures_dir)
    image_paths = []
    
    # Try numbered images first (listing_id_1.jpg, listing_id_2.jpg, etc.)
    for i in range(1, max_images + 1):
        img_path = pictures_path / f"{listing_id}_{i}.jpg"
        if img_path.exists():
            image_paths.append(img_path)
    
    # Try single image without suffix if no numbered images found
    if not image_paths:
        img_path = pictures_path / f"{listing_id}.jpg"
        if img_path.exists():
            image_paths.append(img_path)
    
    return image_paths


if __name__ == "__main__":
    pass
