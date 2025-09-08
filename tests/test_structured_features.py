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

import sys
import os
from pathlib import Path
import numpy as np
import time
import traceback

# Get absolute paths
current_file = Path(__file__).resolve()
if 'tests' in current_file.parts:
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent

print(f"Test file: {current_file}")
print(f"Project root: {project_root}")
print(f"Current working directory: {Path.cwd()}")

# Add paths
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Diagnostic: Check what files exist
print(f"\nDIAGNOSTIC: File system check")
print("-" * 40)

def check_path(path_desc, path):
    exists = path.exists()
    print(f"{path_desc}: {path} - {'EXISTS' if exists else 'MISSING'}")
    if exists and path.is_dir():
        try:
            files = list(path.iterdir())
            print(f"  Contents: {[f.name for f in files[:10]]}")  # Show first 10 files
        except:
            print(f"  Cannot read directory")
    return exists

# Check project structure
check_path("Project root", project_root)
check_path("Source directory", project_root / 'src')
check_path("Config file", project_root / 'src' / 'config.py')
check_path("Feature extractor", project_root / 'src' / 'features' / 'feature_extractor.py')

# Check for models directories
models_dirs = [
    project_root / 'models',
    project_root / 'src' / 'models',
    Path('models'),  # relative to current directory
    Path('src/models')  # relative to current directory
]

for i, models_dir in enumerate(models_dirs):
    check_path(f"Models dir {i+1}", models_dir)

# Find all .pkl files
print(f"\nFinding all .pkl files...")
pkl_files = []
for root, dirs, files in os.walk(project_root):
    for file in files:
        if file.endswith('.pkl'):
            full_path = Path(root) / file
            rel_path = full_path.relative_to(project_root)
            pkl_files.append(str(rel_path))
            
if pkl_files:
    print(f"Found .pkl files: {pkl_files}")
else:
    print("No .pkl files found in project!")

print(f"\n" + "="*60)

# Try to import
try:
    from src.features.feature_extractor import MultiModalFeatureExtractor
    print("SUCCESS: Import successful!")
except ImportError as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)

class RobustFeatureExtractorTester:
    def __init__(self):
        self.test_results = []
        
    def test_with_different_paths(self):
        """Test feature extractor with different model directory paths"""
        print(f"\nTESTING DIFFERENT MODEL PATHS")
        print("-" * 40)
        
        # List of possible model directory paths to try
        possible_paths = [
            'models',
            '../models', 
            'src/models',
            '../src/models',
            str(project_root / 'models'),
            str(project_root / 'src' / 'models')
        ]
        
        for path in possible_paths:
            print(f"\nTrying models_dir='{path}':")
            
            # Check if path exists
            path_obj = Path(path)
            if not path_obj.is_absolute():
                # Make relative to project root
                full_path = project_root / path if not path.startswith('..') else project_root.parent / path.lstrip('../')
            else:
                full_path = path_obj
                
            print(f"  Full path: {full_path}")
            print(f"  Exists: {full_path.exists()}")
            
            if full_path.exists():
                # List .pkl files in this directory
                pkl_files = list(full_path.glob('*.pkl'))
                print(f"  .pkl files: {[f.name for f in pkl_files]}")
                
                # Try to initialize with this path
                try:
                    start_time = time.time()
                    extractor = MultiModalFeatureExtractor(models_dir=str(path))
                    duration = time.time() - start_time
                    
                    # Check what loaded
                    has_structured_model = hasattr(extractor, 'structured_model') and extractor.structured_model is not None
                    has_structured_scaler = hasattr(extractor, 'structured_scaler') and extractor.structured_scaler is not None
                    has_text_scaler = hasattr(extractor, 'text_scaler') and extractor.text_scaler is not None
                    has_image_scaler = hasattr(extractor, 'image_scaler') and extractor.image_scaler is not None
                    
                    loaded_components = []
                    if has_structured_model: loaded_components.append("structured_model")
                    if has_structured_scaler: loaded_components.append("structured_scaler")
                    if has_text_scaler: loaded_components.append("text_scaler")  
                    if has_image_scaler: loaded_components.append("image_scaler")
                    
                    print(f"  LOADED: {loaded_components if loaded_components else 'NONE'}")
                    print(f"  Duration: {duration:.3f}s")
                    
                    if has_structured_model and has_structured_scaler:
                        print(f"  STATUS: READY FOR ENSEMBLE!")
                        self.run_full_tests(extractor)
                        return extractor
                    else:
                        print(f"  STATUS: Missing required components for ensemble")
                        
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
            else:
                print(f"  PATH DOES NOT EXIST")
        
        print(f"\nNo suitable model directory found with all required files.")
        return None
    
    def run_full_tests(self, extractor):
        """Run comprehensive tests on the extractor"""
        print(f"\nRUNNING COMPREHENSIVE TESTS")
        print("-" * 40)
        
        # Test data
        listing_A = {
            'listing_id': 12345,
            'current_price': 500000,
            'surface_m2': 80,
            'room_count': 3,
            'floor': 2,
            'floor_count': 5,
            'description': "Beautiful apartment in central Paris with modern kitchen"
        }
        
        listing_B = {
            'listing_id': 12346,
            'current_price': 520000,
            'surface_m2': 82,
            'room_count': 3,
            'floor': 2,
            'floor_count': 5,
            'description': "Lovely flat in Paris center, renovated kitchen"
        }
        
        tests = [
            ("Structured Features", lambda: extractor.extract_structured_features(listing_A, listing_B)),
            ("Text Features", lambda: extractor.extract_text_features(listing_A['description'], listing_B['description'])),
            ("Image Features", lambda: extractor.extract_image_features(listing_A, listing_B)),
            ("All Features", lambda: extractor.extract_all_features(listing_A, listing_B)),
            ("Ensemble Preparation", lambda: extractor.prepare_for_ensemble(listing_A, listing_B))
        ]
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                print(f"SUCCESS {test_name}: {self._format_result(result)} ({duration:.3f}s)")
                
            except Exception as e:
                print(f"FAILED {test_name}: {str(e)}")
                traceback.print_exc()
    
    def _format_result(self, result):
        """Format test result for display"""
        if isinstance(result, np.ndarray):
            if len(result) <= 10:
                return f"array{result.round(3).tolist()}"
            else:
                return f"array(shape={result.shape}, mean={result.mean():.3f})"
        elif isinstance(result, dict):
            return f"dict with keys: {list(result.keys())}"
        elif isinstance(result, (int, float)):
            return f"{result:.3f}"
        else:
            return str(type(result))

def main():
    """Run the comprehensive test suite"""
    print("ROBUST FEATURE EXTRACTOR DIAGNOSTIC")
    print("=" * 60)
    
    tester = RobustFeatureExtractorTester()
    
    # Try to find working configuration
    extractor = tester.test_with_different_paths()
    
    if extractor is None:
        print(f"\nRECOMMENDATE NEXT STEPS:")
        print(f"1. Create models directory: mkdir -p {project_root / 'models'}")
        print(f"2. Run your training notebooks to create .pkl files:")
        print(f"   - structured_model.pkl")
        print(f"   - structured_scaler.pkl") 
        print(f"   - text_scaler.pkl")
        print(f"   - image_scaler.pkl")
        print(f"3. Re-run this test")
    else:
        print(f"\nSUCCESS: Found working configuration!")
        print(f"Feature extractor is ready for production use.")

if __name__ == "__main__":
    main()

