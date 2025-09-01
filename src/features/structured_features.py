"""
Structured Feature Extractor for Real Estate Duplicate Detection

This module provides production-ready structured feature extraction for comparing
two real estate listings based on numerical and categorical properties.

Features extracted:
1. price_similarity - Relative price difference (0-1 scale)
2. surface_similarity - Relative surface area difference (0-1 scale)  
3. price_per_m2_consistency - Price per square meter consistency
4. room_match - Exact room count match (0 or 1)
5. floor_similarity - Floor level similarity (0-1 scale)
6. floor_count_match - Building floor count match (0 or 1)
7. structured_consistency - Overall consistency score

"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class StructuredFeatureExtractor:
    """
    Extracts 7 numerical features by comparing structured properties of two listings.
    """
    
    def __init__(self, scaler_path: Optional[str] = None):
        """
        Initialize the structured feature extractor.
        
        Args:
            scaler_path: Path to the fitted StandardScaler. If None, uses default location.
        """
        self.feature_names = [
            'price_similarity',
            'surface_similarity', 
            'price_per_m2_consistency',
            'room_match',
            'floor_similarity',
            'floor_count_match',
            'structured_consistency'
        ]
        
        # Load the fitted scaler
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent / "models" / "structured_scaler.pkl"
        
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        except FileNotFoundError:
            logger.warning(f"Scaler not found at {scaler_path}. Features will not be scaled.")
            self.scaler = None
        except Exception as e:
            logger.error(f"Error loading scaler: {e}")
            self.scaler = None
    
    def extract_features(self, listing_A: Dict, listing_B: Dict) -> np.ndarray:
        """
        Extract structured features comparing two listings.
        
        Args:
            listing_A: Dictionary with listing properties (price, surface_m2, etc.)
            listing_B: Dictionary with listing properties (price, surface_m2, etc.)
            
        Returns:
            numpy array of 7 scaled features
        """
        try:
            raw_features = self._extract_raw_features(listing_A, listing_B)
            
            if self.scaler is not None:
                # Scale features using fitted scaler
                features_array = np.array(raw_features).reshape(1, -1)
                scaled_features = self.scaler.transform(features_array)
                return scaled_features[0]
            else:
                logger.warning("No scaler available, returning raw features")
                return np.array(raw_features)
                
        except Exception as e:
            logger.error(f"Error extracting structured features: {e}")
            # Return zero vector as fallback
            return np.zeros(len(self.feature_names))
    
    def _extract_raw_features(self, listing_A: Dict, listing_B: Dict) -> List[float]:
        """
        Extract raw (unscaled) structured features.
        
        Args:
            listing_A: Dictionary with listing properties
            listing_B: Dictionary with listing properties
            
        Returns:
            List of 7 raw feature values
        """
        features = []
        
        # Feature 1: Price similarity
        price_A = self._get_numeric_value(listing_A, 'current_price', 0)
        price_B = self._get_numeric_value(listing_B, 'current_price', 0)
        
        if price_A > 0 and price_B > 0:
            price_sim = 1 - abs(price_A - price_B) / max(price_A, price_B)
        else:
            price_sim = 0.0
        features.append(price_sim)
        
        # Feature 2: Surface similarity
        surface_A = self._get_numeric_value(listing_A, 'surface_m2', 0)
        surface_B = self._get_numeric_value(listing_B, 'surface_m2', 0)
        
        if surface_A > 0 and surface_B > 0:
            surface_sim = 1 - abs(surface_A - surface_B) / max(surface_A, surface_B)
        else:
            surface_sim = 0.0
        features.append(surface_sim)
        
        # Feature 3: Price per m² consistency
        if all(x > 0 for x in [price_A, price_B, surface_A, surface_B]):
            ppm2_A = price_A / surface_A
            ppm2_B = price_B / surface_B
            ppm2_sim = 1 - abs(ppm2_A - ppm2_B) / max(ppm2_A, ppm2_B)
        else:
            ppm2_sim = 0.0
        features.append(ppm2_sim)
        
        # Feature 4: Room count match
        room_A = self._get_numeric_value(listing_A, 'room_count', None)
        room_B = self._get_numeric_value(listing_B, 'room_count', None)
        
        if room_A is not None and room_B is not None:
            room_match = 1.0 if room_A == room_B else 0.0
        else:
            room_match = 0.5  # Neutral value for missing data
        features.append(room_match)
        
        # Feature 5: Floor similarity
        floor_A = self._get_numeric_value(listing_A, 'floor', None)
        floor_B = self._get_numeric_value(listing_B, 'floor', None)
        
        if floor_A is not None and floor_B is not None:
            if floor_A == floor_B:
                floor_sim = 1.0
            else:
                # Similar floors get partial credit
                floor_diff = abs(float(floor_A) - float(floor_B))
                floor_sim = max(0.0, 1.0 - floor_diff / 5.0)  # Normalize by max 5 floor difference
        else:
            floor_sim = 0.5  # Neutral value for missing data
        features.append(floor_sim)
        
        # Feature 6: Floor count match
        floor_count_A = self._get_numeric_value(listing_A, 'floor_count', None)
        floor_count_B = self._get_numeric_value(listing_B, 'floor_count', None)
        
        if floor_count_A is not None and floor_count_B is not None:
            floor_count_match = 1.0 if floor_count_A == floor_count_B else 0.0
        else:
            floor_count_match = 0.5  # Neutral value for missing data
        features.append(floor_count_match)
        
        # Feature 7: Overall structured consistency
        # Average of valid features (excluding neutral values)
        valid_features = [f for f in features[:6] if f != 0.5 and f > 0]
        consistency_score = np.mean(valid_features) if valid_features else 0.0
        features.append(consistency_score)
        
        return features
    
    def _get_numeric_value(self, listing: Dict, key: str, default: Union[float, None]) -> Union[float, None]:
        """
        Safely extract numeric value from listing dictionary.
        
        Args:
            listing: Listing dictionary
            key: Key to extract
            default: Default value if key missing or invalid
            
        Returns:
            Numeric value or default
        """
        try:
            value = listing.get(key, default)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid numeric value for {key}: {listing.get(key)}")
            return default
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        return self.feature_names.copy()
    
    def get_feature_explanation(self, features: np.ndarray) -> Dict[str, float]:
        """
        Get human-readable explanation of extracted features.
        
        Args:
            features: Array of extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        if len(features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")
        
        return dict(zip(self.feature_names, features))


# Convenience function for quick feature extraction
def extract_structured_features(listing_A: Dict, listing_B: Dict, 
                               scaler_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to extract structured features from two listings.
    
    Args:
        listing_A: First listing dictionary
        listing_B: Second listing dictionary
        scaler_path: Path to fitted scaler (optional)
        
    Returns:
        Array of 7 scaled structured features
    """
    extractor = StructuredFeatureExtractor(scaler_path)
    return extractor.extract_features(listing_A, listing_B)


if __name__ == "__main__":
    pass
