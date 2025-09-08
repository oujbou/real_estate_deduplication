# check_pkl_files.py
import joblib
from pathlib import Path

models_dir = Path("/Users/tarikoujlakh/Projects/real_estate_deduplication/models")

files_to_check = [
    'structured_model.pkl',
    'structured_scaler.pkl',
    'text_scaler.pkl',
    'image_scaler.pkl'
]

print("🔍 Checking .pkl files for corruption...")
print("=" * 50)

for file_name in files_to_check:
    file_path = models_dir / file_name
    print(f"\nChecking: {file_name}")
    
    if not file_path.exists():
        print(f"❌ File does not exist")
        continue
        
    file_size = file_path.stat().st_size
    print(f"   Size: {file_size} bytes")
    
    try:
        obj = joblib.load(file_path)
        print(f"✅ Loaded successfully")
        print(f"   Type: {type(obj).__name__}")
        
        # Additional info based on object type
        if hasattr(obj, 'scale_'):
            print(f"   Scaler - n_features: {len(obj.scale_)}")
        elif hasattr(obj, 'predict_proba'):
            print(f"   Model - has predict_proba: ✓")
        elif hasattr(obj, 'feature_importances_'):
            print(f"   Model - n_features: {len(obj.feature_importances_)}")
            
    except Exception as e:
        print(f"❌ Failed to load: {e}")