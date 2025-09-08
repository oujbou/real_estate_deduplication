"""
Simple test for the Real Estate Duplicate Detection API
"""

import requests
import json
import time

API_URL = "http://localhost:8000"

def test_api():
    """Test the API endpoints"""
    print("Testing Real Estate Duplicate Detection API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Service status: {data.get('status')}")
            print("✓ Health check passed")
        else:
            print("✗ Health check failed")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Models directory: {data.get('models_directory')}")
            print(f"Device: {data.get('device')}")
            print("✓ Model info passed")
        else:
            print("✗ Model info failed")
    except Exception as e:
        print(f"✗ Model info failed: {e}")
    
    # Test 3: Prediction - Similar listings (should be duplicates)
    print("\n3. Testing prediction with similar listings...")
    
    similar_data = {
        "listing_1": {
            "listing_id": 12345,
            "current_price": 500000,
            "surface_m2": 75,
            "room_count": 3,
            "floor": 2,
            "floor_count": 5,
            "description": "Bel appartement 3 pièces lumineux avec balcon, proche métro."
        },
        "listing_2": {
            "listing_id": 67890,
            "current_price": 520000,
            "surface_m2": 73,
            "room_count": 3,
            "floor": 2,
            "floor_count": 5,
            "description": "Appartement 3 pièces avec balcon, très lumineux, proche transport."
        },
        "threshold": 0.5
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict", 
            json=similar_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {response_time:.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            
            print(f"Is duplicate: {prediction['is_duplicate']}")
            print(f"Confidence: {prediction['confidence']:.3f}")
            print(f"Structured: {prediction['features']['structured_prediction']:.3f}")
            print(f"Text similarity: {prediction['features']['text_similarity']:.3f}")
            print(f"Image similarity: {prediction['features']['image_similarity']:.3f}")
            
            # Check response time requirement
            if response_time <= 2.0:
                print("✓ Response time within 2s requirement")
            else:
                print("⚠ Response time exceeds 2s requirement")
            
            print("✓ Similar listings prediction passed")
        else:
            print(f"✗ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False
    
    # Test 4: Prediction - Different listings (should not be duplicates)
    print("\n4. Testing prediction with different listings...")
    
    different_data = {
        "listing_1": {
            "listing_id": 11111,
            "current_price": 300000,
            "surface_m2": 45,
            "room_count": 2,
            "floor": 0,
            "floor_count": 3,
            "description": "Studio moderne dans résidence calme avec parking."
        },
        "listing_2": {
            "listing_id": 22222,
            "current_price": 800000,
            "surface_m2": 120,
            "room_count": 5,
            "floor": 10,
            "floor_count": 15,
            "description": "Magnifique penthouse avec vue panoramique et terrasse."
        },
        "threshold": 0.5
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/predict", 
            json=different_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {response_time:.3f}s")
        
        if response.status_code == 200:
            data = response.json()
            prediction = data['prediction']
            
            print(f"Is duplicate: {prediction['is_duplicate']}")
            print(f"Confidence: {prediction['confidence']:.3f}")
            
            print("✓ Different listings prediction passed")
        else:
            print(f"✗ Prediction failed: {response.text}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
    
    print("\n" + "=" * 50)
    print("API testing completed!")
    print("\nTo test manually, try:")
    print("curl http://localhost:5000/health")
    
    return True

if __name__ == "__main__":
    print("Make sure the API is running first:")
    print("python app.py")
    print("\nThen run this test...")
    input("Press Enter to continue...")
    
    test_api()