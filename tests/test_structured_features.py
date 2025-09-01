"""
Unit tests for StructuredFeatureExtractor

Tests the structured feature extraction functionality including:
- Price similarity calculation
- Surface similarity calculation  
- Room matching
- Floor comparison
- Error handling and edge cases
- Feature scaling

Run with: python -m pytest tests/test_structured_features.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.structured_features import StructuredFeatureExtractor, extract_structured_features


class TestStructuredFeatureExtractor:
    """Test suite for StructuredFeatureExtractor class."""
    
    @pytest.fixture
    def sample_listings(self):
        """Sample listing data for testing."""
        listing_a = {
            'current_price': 180000,
            'surface_m2': 45,
            'room_count': 2,
            'floor': 3,
            'floor_count': 5
        }
        
        listing_b = {
            'current_price': 185000,
            'surface_m2': 47,
            'room_count': 2,
            'floor': 3,
            'floor_count': 5
        }
        
        return listing_a, listing_b
    
    @pytest.fixture
    def extractor_no_scaler(self):
        """Create extractor without scaler for testing raw features."""
        with patch('joblib.load', side_effect=FileNotFoundError):
            return StructuredFeatureExtractor()
    
    @pytest.fixture
    def mock_scaler(self):
        """Create mock scaler for testing."""
        scaler = Mock()
        scaler.transform.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
        return scaler
    
    def test_initialization_with_scaler(self, mock_scaler):
        """Test extractor initialization with valid scaler."""
        with patch('joblib.load', return_value=mock_scaler):
            extractor = StructuredFeatureExtractor()
            assert extractor.scaler is not None
            assert len(extractor.feature_names) == 7
    
    def test_initialization_without_scaler(self):
        """Test extractor initialization with missing scaler."""
        with patch('joblib.load', side_effect=FileNotFoundError):
            extractor = StructuredFeatureExtractor()
            assert extractor.scaler is None
            assert len(extractor.feature_names) == 7
    
    def test_identical_listings(self, sample_listings, extractor_no_scaler):
        """Test feature extraction for identical listings."""
        listing_a, _ = sample_listings
        features = extractor_no_scaler.extract_features(listing_a, listing_a)
        
        # All similarity features should be 1.0 for identical listings
        assert features[0] == 1.0  # price_similarity
        assert features[1] == 1.0  # surface_similarity
        assert features[2] == 1.0  # price_per_m2_consistency
        assert features[3] == 1.0  # room_match
        assert features[4] == 1.0  # floor_similarity
        assert features[5] == 1.0  # floor_count_match
        assert features[6] == 1.0  # structured_consistency
    
    def test_similar_listings(self, sample_listings, extractor_no_scaler):
        """Test feature extraction for similar listings."""
        listing_a, listing_b = sample_listings
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        # Check that features are reasonable for similar listings
        assert 0.9 <= features[0] <= 1.0  # price_similarity (5k difference on 180k)
        assert 0.9 <= features[1] <= 1.0  # surface_similarity (2m2 difference on 45m2)
        assert 0.9 <= features[2] <= 1.0  # price_per_m2_consistency
        assert features[3] == 1.0  # room_match (both 2 rooms)
        assert features[4] == 1.0  # floor_similarity (same floor)
        assert features[5] == 1.0  # floor_count_match (same building height)
        assert 0.9 <= features[6] <= 1.0  # overall consistency
    
    def test_very_different_listings(self, extractor_no_scaler):
        """Test feature extraction for very different listings."""
        listing_a = {
            'current_price': 100000,
            'surface_m2': 30,
            'room_count': 1,
            'floor': 1,
            'floor_count': 3
        }
        
        listing_b = {
            'current_price': 500000,
            'surface_m2': 100,
            'room_count': 5,
            'floor': 8,
            'floor_count': 10
        }
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        # Features should show low similarity
        assert features[0] < 0.5  # price_similarity
        assert features[1] < 0.5  # surface_similarity
        assert features[3] == 0.0  # room_match (different room counts)
        assert features[5] == 0.0  # floor_count_match
        assert features[6] < 0.5  # overall consistency
    
    def test_missing_price_data(self, extractor_no_scaler):
        """Test handling of missing price data."""
        listing_a = {'current_price': 180000, 'surface_m2': 45}
        listing_b = {'surface_m2': 47}  # Missing price
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        assert features[0] == 0.0  # price_similarity should be 0
        assert features[2] == 0.0  # price_per_m2_consistency should be 0
        assert 0 < features[1] < 1  # surface_similarity should still work
    
    def test_missing_surface_data(self, extractor_no_scaler):
        """Test handling of missing surface data."""
        listing_a = {'current_price': 180000, 'surface_m2': 45}
        listing_b = {'current_price': 185000}  # Missing surface
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        assert features[1] == 0.0  # surface_similarity should be 0
        assert features[2] == 0.0  # price_per_m2_consistency should be 0
        assert 0 < features[0] < 1  # price_similarity should still work
    
    def test_missing_room_data(self, extractor_no_scaler):
        """Test handling of missing room count data."""
        listing_a = {'current_price': 180000, 'room_count': 2}
        listing_b = {'current_price': 185000}  # Missing room_count
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        assert features[3] == 0.5  # room_match should be neutral (0.5)
    
    def test_floor_similarity_calculation(self, extractor_no_scaler):
        """Test floor similarity calculation with different floor values."""
        # Same floor
        listing_a = {'floor': 3}
        listing_b = {'floor': 3}
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        assert features[4] == 1.0
        
        # Adjacent floors
        listing_a = {'floor': 3}
        listing_b = {'floor': 4}
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        assert 0.7 < features[4] < 1.0  # Should get partial credit
        
        # Distant floors
        listing_a = {'floor': 1}
        listing_b = {'floor': 10}
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        assert features[4] == 0.0  # Too far apart
    
    def test_zero_price_handling(self, extractor_no_scaler):
        """Test handling of zero prices (should be treated as missing)."""
        listing_a = {'current_price': 0, 'surface_m2': 45}
        listing_b = {'current_price': 180000, 'surface_m2': 47}
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        assert features[0] == 0.0  # price_similarity should be 0 for zero price
        assert features[2] == 0.0  # price_per_m2_consistency should be 0
    
    def test_negative_values(self, extractor_no_scaler):
        """Test handling of invalid negative values."""
        listing_a = {'current_price': -100, 'surface_m2': -20}
        listing_b = {'current_price': 180000, 'surface_m2': 45}
        
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        
        # Negative values should be treated as missing
        assert features[0] == 0.0  # price_similarity
        assert features[1] == 0.0  # surface_similarity
    
    def test_feature_scaling(self, sample_listings, mock_scaler):
        """Test that features are properly scaled when scaler is available."""
        listing_a, listing_b = sample_listings
        
        with patch('joblib.load', return_value=mock_scaler):
            extractor = StructuredFeatureExtractor()
            features = extractor.extract_features(listing_a, listing_b)
        
        # Should return scaled features from mock
        expected_features = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        np.testing.assert_array_equal(features, expected_features)
        
        # Verify scaler was called
        mock_scaler.transform.assert_called_once()
    
    def test_get_feature_names(self, extractor_no_scaler):
        """Test getting feature names."""
        names = extractor_no_scaler.get_feature_names()
        
        expected_names = [
            'price_similarity', 'surface_similarity', 'price_per_m2_consistency',
            'room_match', 'floor_similarity', 'floor_count_match', 'structured_consistency'
        ]
        
        assert names == expected_names
        assert len(names) == 7
    
    def test_feature_explanation(self, sample_listings, extractor_no_scaler):
        """Test feature explanation functionality."""
        listing_a, listing_b = sample_listings
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        explanation = extractor_no_scaler.get_feature_explanation(features)
        
        assert isinstance(explanation, dict)
        assert len(explanation) == 7
        assert 'price_similarity' in explanation
        assert all(isinstance(v, (int, float, np.number)) for v in explanation.values())
    
    def test_error_handling(self, extractor_no_scaler):
        """Test error handling with invalid input."""
        # Empty dictionaries
        features = extractor_no_scaler.extract_features({}, {})
        assert len(features) == 7
        assert all(isinstance(f, (int, float, np.number)) for f in features)
        
        # None values
        features = extractor_no_scaler.extract_features(None, {})
        assert len(features) == 7  # Should return zero vector
    
    def test_convenience_function(self, sample_listings):
        """Test the convenience function extract_structured_features."""
        listing_a, listing_b = sample_listings
        
        with patch('joblib.load', side_effect=FileNotFoundError):
            features = extract_structured_features(listing_a, listing_b)
        
        assert len(features) == 7
        assert all(isinstance(f, (int, float, np.number)) for f in features)
    
    def test_invalid_data_types(self, extractor_no_scaler):
        """Test handling of invalid data types."""
        listing_a = {'current_price': 'invalid', 'surface_m2': 'also_invalid'}
        listing_b = {'current_price': 180000, 'surface_m2': 45}
        
        # Should not crash, should handle gracefully
        features = extractor_no_scaler.extract_features(listing_a, listing_b)
        assert len(features) == 7
        assert features[0] == 0.0  # price_similarity should be 0 for invalid price
        assert features[1] == 0.0  # surface_similarity should be 0 for invalid surface


class TestPricePerM2Calculation:
    """Specific tests for price per m² consistency calculation."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor for testing."""
        with patch('joblib.load', side_effect=FileNotFoundError):
            return StructuredFeatureExtractor()
    
    def test_consistent_price_per_m2(self, extractor):
        """Test listings with consistent price per m²."""
        # Both have 4000€/m²
        listing_a = {'current_price': 200000, 'surface_m2': 50}
        listing_b = {'current_price': 160000, 'surface_m2': 40}
        
        features = extractor.extract_features(listing_a, listing_b)
        
        # Price per m² is identical, so consistency should be 1.0
        assert features[2] == 1.0
    
    def test_inconsistent_price_per_m2(self, extractor):
        """Test listings with very different price per m²."""
        listing_a = {'current_price': 200000, 'surface_m2': 50}  # 4000€/m²
        listing_b = {'current_price': 200000, 'surface_m2': 100}  # 2000€/m²
        
        features = extractor.extract_features(listing_a, listing_b)
        
        # 50% difference in price per m², so consistency should be 0.5
        assert abs(features[2] - 0.5) < 0.01


if __name__ == "__main__":
    # Run tests if script is called directly
    pytest.main([__file__, "-v"])