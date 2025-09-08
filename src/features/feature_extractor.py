"""
Multimodal Feature Extractor for Real Estate Duplicate Detection
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class MultiModalFeatureExtractor:
    """
    Multimodal feature extractor for real estate duplicate detection.
    """
    
    def __init__(self):
        self.device = torch.device("mps")
        self.models_dir = Path("./models")
        
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        # Load structured model and scaler
        self.structured_model = joblib.load(self.models_dir / "structured_model.pkl")
        self.structured_scaler = joblib.load(self.models_dir / "structured_scaler.pkl")
        
        # Load text model and scaler
        self.text_model = SentenceTransformer('distiluse-base-multilingual-cased')
        self.text_scaler = joblib.load(self.models_dir / "text_scaler.pkl")
        
        # Load image model and scaler
        self.image_model = models.resnet50(pretrained=True)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])
        self.image_model.eval()
        self.image_model = self.image_model.to(self.device)
        self.image_scaler = joblib.load(self.models_dir / "image_scaler.pkl")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_structured_features(self, listing_A, listing_B):
        """Extract 7 structured features"""
        features = []
        
        # Price similarity
        price_A = listing_A.get('current_price', 0)
        price_B = listing_B.get('current_price', 0)
        if price_A > 0 and price_B > 0:
            price_sim = 1 - abs(price_A - price_B) / max(price_A, price_B)
        else:
            price_sim = 0.0
        features.append(price_sim)
        
        # Surface similarity
        surface_A = listing_A.get('surface_m2', 0)
        surface_B = listing_B.get('surface_m2', 0)
        if surface_A > 0 and surface_B > 0:
            surface_sim = 1 - abs(surface_A - surface_B) / max(surface_A, surface_B)
        else:
            surface_sim = 0.0
        features.append(surface_sim)
        
        # Price per m² consistency
        if all(x > 0 for x in [price_A, price_B, surface_A, surface_B]):
            ppm2_A = price_A / surface_A
            ppm2_B = price_B / surface_B
            ppm2_sim = 1 - abs(ppm2_A - ppm2_B) / max(ppm2_A, ppm2_B)
        else:
            ppm2_sim = 0.0
        features.append(ppm2_sim)
        
        # Room count match
        room_A = listing_A.get('room_count')
        room_B = listing_B.get('room_count')
        room_match = 1.0 if room_A == room_B else 0.0 if room_A is not None and room_B is not None else 0.5
        features.append(room_match)
        
        # Floor similarity
        floor_A = listing_A.get('floor')
        floor_B = listing_B.get('floor')
        if floor_A is not None and floor_B is not None:
            floor_sim = 1.0 if floor_A == floor_B else max(0.0, 1.0 - abs(float(floor_A) - float(floor_B)) / 5.0)
        else:
            floor_sim = 0.5
        features.append(floor_sim)
        
        # Floor count match
        floor_count_A = listing_A.get('floor_count')
        floor_count_B = listing_B.get('floor_count')
        floor_count_match = 1.0 if floor_count_A == floor_count_B else 0.0 if floor_count_A is not None and floor_count_B is not None else 0.5
        features.append(floor_count_match)
        
        # Overall consistency
        valid_features = [f for f in features[:6] if f != 0.5 and f > 0]
        consistency_score = np.mean(valid_features) if valid_features else 0.0
        features.append(consistency_score)
        
        return np.array(features)
    
    def extract_text_similarity(self, desc_A, desc_B):
        """Extract semantic similarity"""
        desc_A = str(desc_A) if pd.notna(desc_A) else ""
        desc_B = str(desc_B) if pd.notna(desc_B) else ""
        
        if len(desc_A.strip()) == 0 or len(desc_B.strip()) == 0:
            return 0.0
        
        embedding_A = self.text_model.encode([desc_A])[0]
        embedding_B = self.text_model.encode([desc_B])[0]
        
        similarity = cosine_similarity([embedding_A], [embedding_B])[0][0]
        return float(similarity)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image).unsqueeze(0)
        except:
            return None
    
    def extract_image_similarity(self, listing_id_A, listing_id_B, pictures_dir="./data/pictures"):
        """Extract image similarity"""
        pictures_path = Path(pictures_dir)
        if not pictures_path.exists():
            return 0.0
        
        # Find images
        images_A = list(pictures_path.glob(f"{listing_id_A}__*.jpg"))
        images_B = list(pictures_path.glob(f"{listing_id_B}__*.jpg"))
        
        if not images_A or not images_B:
            return 0.0
        
        # Process first image of each listing
        tensor_A = self.load_and_preprocess_image(images_A[0])
        tensor_B = self.load_and_preprocess_image(images_B[0])
        
        if tensor_A is None or tensor_B is None:
            return 0.0
        
        # Extract features
        with torch.no_grad():
            features_A = self.image_model(tensor_A.to(self.device))
            features_B = self.image_model(tensor_B.to(self.device))
            
            features_A = features_A.view(features_A.size(0), -1).cpu().numpy()
            features_B = features_B.view(features_B.size(0), -1).cpu().numpy()
            
            similarity = cosine_similarity(features_A, features_B)[0][0]
            return float(similarity)
    
    def extract_features(self, listing_A, listing_B, pictures_dir="./data/pictures"):
        """
        Extract all features and return ensemble-compatible format.
        
        Returns:
            np.array: [structured_prediction, text_similarity, image_similarity]
        """
        # Structured features and prediction
        structured_features = self.extract_structured_features(listing_A, listing_B)
        structured_features_scaled = self.structured_scaler.transform([structured_features])
        structured_prediction = self.structured_model.predict_proba(structured_features_scaled)[0][1]
        
        # Text similarity
        desc_A = listing_A.get('description', '')
        desc_B = listing_B.get('description', '')
        text_similarity = self.extract_text_similarity(desc_A, desc_B)
        
        # Image similarity
        listing_id_A = listing_A.get('listing_id')
        listing_id_B = listing_B.get('listing_id')
        image_similarity = self.extract_image_similarity(listing_id_A, listing_id_B, pictures_dir)
        
        return np.array([structured_prediction, text_similarity, image_similarity])