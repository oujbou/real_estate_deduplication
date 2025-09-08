"""
Test script for MultiModalFeatureExtractor
"""

import sys
import os
from pathlib import Path
sys.path.append('.')
# Try multiple import strategies
def import_extractor():
    """Try different ways to import the extractor"""
    
    # Strategy 1: Direct import (same directory)
    try:
        from multimodal_feature_extractor import MultiModalFeatureExtractor
        return MultiModalFeatureExtractor
    except ImportError:
        pass
    
    # Strategy 2: Add current directory to path
    try:
        sys.path.insert(0, '.')
        from multimodal_feature_extractor import MultiModalFeatureExtractor
        return MultiModalFeatureExtractor
    except ImportError:
        pass
    
    # Strategy 3: Add src directory to path
    try:
        sys.path.insert(0, './src')
        from multimodal_feature_extractor import MultiModalFeatureExtractor
        return MultiModalFeatureExtractor
    except ImportError:
        pass
    
    # Strategy 4: Add parent directory
    try:
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        from multimodal_feature_extractor import MultiModalFeatureExtractor
        return MultiModalFeatureExtractor
    except ImportError:
        pass
    
    # If all fail, give helpful error
    raise ImportError(
        "Could not import MultiModalFeatureExtractor. "
        "Please ensure multimodal_feature_extractor.py is in:\n"
        "1. Same directory as this test file, OR\n"
        "2. ./src/ directory, OR\n" 
        "3. Add the correct path to sys.path"
    )

# Import the extractor
try:
    MultiModalFeatureExtractor = import_extractor()
    print("Successfully imported MultiModalFeatureExtractor")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def test_extractor_initialization():
    """Test that the extractor initializes correctly"""
    print("Testing extractor initialization...")
    try:
        extractor = MultiModalFeatureExtractor()
        print("✓ Extractor initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False

def test_structured_features():
    """Test structured feature extraction"""
    print("\nTesting structured features...")
    
    extractor = MultiModalFeatureExtractor()
    
    listing_1 = {
        'current_price': 500000,
        'surface_m2': 75,
        'room_count': 3,
        'floor': 2,
        'floor_count': 5
    }
    
    listing_2 = {
        'current_price': 520000,
        'surface_m2': 73,
        'room_count': 3,
        'floor': 2,
        'floor_count': 5
    }
    
    try:
        features = extractor.extract_structured_features(listing_1, listing_2)
        print(f"✓ Structured features extracted: {features.shape}")
        print(f"  Features: {features}")
        
        # Verify we get 7 features
        assert len(features) == 7, f"Expected 7 features, got {len(features)}"
        print("✓ Correct number of features")
        
        return True
    except Exception as e:
        print(f"✗ Structured features failed: {e}")
        return False

def test_text_similarity():
    """Test text similarity extraction"""
    print("\nTesting text similarity...")
    
    extractor = MultiModalFeatureExtractor()
    
    desc_1 = "Bel appartement 3 pièces lumineux avec balcon"
    desc_2 = "Appartement lumineux 3 pièces avec balcon"
    
    try:
        similarity = extractor.extract_text_similarity(desc_1, desc_2)
        print(f"✓ Text similarity extracted: {similarity:.3f}")
        
        # Verify similarity is between 0 and 1
        assert 0 <= similarity <= 1, f"Similarity should be [0,1], got {similarity}"
        print("✓ Similarity in valid range")
        
        return True
    except Exception as e:
        print(f"✗ Text similarity failed: {e}")
        return False

def test_full_feature_extraction():
    """Test complete feature extraction pipeline"""
    print("\nTesting full feature extraction...")
    
    extractor = MultiModalFeatureExtractor()
    
    listing_1 = {
        'listing_id': 12345,
        'current_price': 500000,
        'surface_m2': 75,
        'room_count': 3,
        'floor': 2,
        'floor_count': 5,
        'description': 'Bel appartement 3 pièces lumineux avec balcon'
    }
    
    listing_2 = {
        'listing_id': 67890,
        'current_price': 520000,
        'surface_m2': 73,
        'room_count': 3,
        'floor': 2,
        'floor_count': 5,
        'description': 'Appartement lumineux 3 pièces avec balcon'
    }
    
    try:
        features = extractor.extract_features(listing_1, listing_2)
        print(f"✓ Full features extracted: {features}")
        
        # Verify we get 3 features: [structured_prediction, text_similarity, image_similarity]
        assert len(features) == 3, f"Expected 3 features, got {len(features)}"
        print("✓ Correct output format")
        
        # Verify all features are in valid range
        for i, feature in enumerate(features):
            assert 0 <= feature <= 1, f"Feature {i} out of range [0,1]: {feature}"
        print("✓ All features in valid range")
        
        return True
    except Exception as e:
        print(f"✗ Full feature extraction failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and missing data"""
    print("\nTesting edge cases...")
    
    extractor = MultiModalFeatureExtractor()
    
    # Test with missing data
    listing_empty = {
        'listing_id': 99999,
        'description': ''
    }
    
    listing_partial = {
        'listing_id': 88888,
        'current_price': 100000,
        'description': 'Simple description'
    }
    
    try:
        features = extractor.extract_features(listing_empty, listing_partial)
        print(f"✓ Edge case handled: {features}")
        
        # Should still return 3 features
        assert len(features) == 3, f"Expected 3 features, got {len(features)}"
        print("✓ Edge case produces correct format")
        
        return True
    except Exception as e:
        print(f"✗ Edge case failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MULTIMODAL FEATURE EXTRACTOR TESTS")
    print("=" * 40)
    
    tests = [
        test_extractor_initialization,
        test_structured_features,
        test_text_similarity,
        test_full_feature_extraction,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\nRESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Feature extractor is ready.")
        return True
    else:
        print("Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    main()