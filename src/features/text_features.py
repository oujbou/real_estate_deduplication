"""
Text Feature Extractor for Real Estate Duplicate Detection

This module provides production-ready text feature extraction for comparing
Real estate descriptions using semantic similarity and keyword analysis.

Features extracted:
1. semantic_similarity - Cosine similarity of text embeddings (0-1 scale)
2. description_length_ratio - Length consistency between descriptions  
3. keyword_overlap - Overlap of real estate keywords

Supports both Sentence-BERT and TF-IDF approaches.

"""

import numpy as np
import pandas as pd
import joblib
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, fallback to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using TF-IDF fallback")


class TextFeatureExtractor:
    """
    Production-ready extractor for text-based real estate features.
    
    Extracts 3 features by comparing real estate descriptions:
    - Semantic similarity (via Sentence-BERT or TF-IDF)
    - Description length ratio
    - Real estate keyword overlap
    """
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 scaler_path: Optional[str] = None):
        """
        Initialize the text feature extractor.
        
        Args:
            model_name: Name of the sentence transformer model (if available)
            scaler_path: Path to the fitted StandardScaler. If None, uses default location.
        """
        self.feature_names = [
            'semantic_similarity',
            'description_length_ratio', 
            'keyword_overlap'
        ]
        
        # Initialize text processing model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer(model_name)
                self.method = "sentence_bert"
                logger.info(f"Loaded Sentence-BERT model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load Sentence-BERT: {e}")
                self._initialize_tfidf()
        else:
            self._initialize_tfidf()
        
        # Load the fitted scaler
        if scaler_path is None:
            scaler_path = Path(__file__).parent.parent.parent / "models" / "text_scaler.pkl"
        
        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded text scaler from {scaler_path}")
        except FileNotFoundError:
            logger.warning(f"Text scaler not found at {scaler_path}. Features will not be scaled.")
            self.scaler = None
        except Exception as e:
            logger.error(f"Error loading text scaler: {e}")
            self.scaler = None
        
        # Real estate keywords for overlap calculation
        self.domaine_keywords = {
        'location': ['proche', 'gare', 'commerce', 'école', 'transport', 'métro', 'rer', 'bus', 'train', 'centre', 'quartier', 'commodité'],
        'features': ['balcon', 'terrasse', 'cave', 'parking', 'ascenseur', 'gardien', 'jardin', 'cheminée', 'vis-a-vis'],
        'condition': ['rénové', 'refait', 'neuf', 'ancien', 'lumineux', 'calme', 'isolé'],
        'rooms': ['séjour', 'salon', 'cuisine', 'chambre', 'salle', 'bain', 'wc', 'dressing', 'hall', 'buanderie', 'grenier'],
        'building': ['étage', 'immeuble', 'résidence', 'copropriété', 'charges', 'fermé', 'sécurisée']
        }
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF model as fallback."""
        self.text_model = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True,
            stop_words=None  # Keep all words for real estate
        )
        self.method = "tfidf"
        self._tfidf_fitted = False
        logger.info("Initialized TF-IDF fallback model")
    
    def extract_features(self, description_A: Union[str, None], 
                        description_B: Union[str, None]) -> Dict[str, Union[np.ndarray, float]]:
        """
        Extract text features comparing two descriptions.
        
        Args:
            description_A: First listing description (can be None)
            description_B: Second listing description (can be None)
            
        Returns:
            Dictionary with 'scaled_features' array and individual feature values
        """
        try:
            raw_features = self._extract_raw_features(description_A, description_B)
            
            if self.scaler is not None:
                # Scale features using fitted scaler
                features_array = np.array(raw_features['features']).reshape(1, -1)
                scaled_features = self.scaler.transform(features_array)
                
                return {
                    'scaled_features': scaled_features[0],
                    'semantic_similarity': raw_features['semantic_similarity'],
                    'description_length_ratio': raw_features['description_length_ratio'],
                    'keyword_overlap': raw_features['keyword_overlap']
                }
            else:
                logger.warning("No text scaler available, returning raw features")
                return {
                    'scaled_features': np.array(raw_features['features']),
                    **raw_features
                }
                
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            # Return zero vector as fallback
            zero_features = np.zeros(len(self.feature_names))
            return {
                'scaled_features': zero_features,
                'semantic_similarity': 0.0,
                'description_length_ratio': 0.0,
                'keyword_overlap': 0.0
            }
    
    def _extract_raw_features(self, description_A: Union[str, None], 
                             description_B: Union[str, None]) -> Dict:
        """
        Extract raw (unscaled) text features.
        
        Args:
            description_A: First description
            description_B: Second description
            
        Returns:
            Dictionary with feature values and array
        """
        # Handle missing descriptions
        desc_A = self._clean_description(description_A)
        desc_B = self._clean_description(description_B)
        
        # Feature 1: Semantic similarity
        semantic_sim = self._calculate_semantic_similarity(desc_A, desc_B)
        
        # Feature 2: Description length ratio
        length_ratio = self._calculate_length_ratio(desc_A, desc_B)
        
        # Feature 3: Keyword overlap
        keyword_overlap = self._calculate_keyword_overlap(desc_A, desc_B)
        
        features = [semantic_sim, length_ratio, keyword_overlap]
        
        return {
            'features': features,
            'semantic_similarity': semantic_sim,
            'description_length_ratio': length_ratio,
            'keyword_overlap': keyword_overlap
        }
    
    def _clean_description(self, description: Union[str, None]) -> str:
        """Clean and normalize description text."""
        if description is None or pd.isna(description):
            return ""
        
        desc = str(description).strip().lower()
        # Basic cleanup - remove extra whitespace
        desc = re.sub(r'\s+', ' ', desc)
        return desc
    
    def _calculate_semantic_similarity(self, desc_A: str, desc_B: str) -> float:
        """Calculate semantic similarity between two descriptions."""
        if not desc_A or not desc_B:
            return 0.0
        
        try:
            if self.method == "sentence_bert":
                return self._bert_similarity(desc_A, desc_B)
            else:
                return self._tfidf_similarity(desc_A, desc_B)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _bert_similarity(self, desc_A: str, desc_B: str) -> float:
        """Calculate similarity using Sentence-BERT."""
        embedding_A = self.text_model.encode([desc_A])
        embedding_B = self.text_model.encode([desc_B])
        similarity = cosine_similarity(embedding_A, embedding_B)[0][0]
        return float(similarity)
    
    def _tfidf_similarity(self, desc_A: str, desc_B: str) -> float:
        """Calculate similarity using TF-IDF."""
        if not self._tfidf_fitted:
            # Fit on the current pair (not ideal but works for single predictions)
            self.text_model.fit([desc_A, desc_B])
            self._tfidf_fitted = True
        
        try:
            vectors = self.text_model.transform([desc_A, desc_B])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except:
            # If transform fails, fit on current texts
            vectors = self.text_model.fit_transform([desc_A, desc_B])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
    
    def _calculate_length_ratio(self, desc_A: str, desc_B: str) -> float:
        """Calculate description length ratio (shorter/longer)."""
        len_A = len(desc_A)
        len_B = len(desc_B)
        
        if len_A == 0 and len_B == 0:
            return 1.0  # Both empty - perfect match
        elif len_A == 0 or len_B == 0:
            return 0.0  # One empty - no similarity
        else:
            return min(len_A, len_B) / max(len_A, len_B)
    
    def _calculate_keyword_overlap(self, desc_A: str, desc_B: str) -> float:
        """Calculate real estate keyword overlap."""
        keywords_A = self._extract_keywords(desc_A)
        keywords_B = self._extract_keywords(desc_B)
        
        if len(keywords_A) == 0 and len(keywords_B) == 0:
            return 0.0  # No keywords found in either
        
        # Calculate Jaccard similarity
        intersection = len(keywords_A.intersection(keywords_B))
        union = len(keywords_A.union(keywords_B))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract real estate keywords from text."""
        found_keywords = set()
        
        for category, words in self.domaine_keywords.items():
            for word in words:
                if word in text:
                    found_keywords.add(word)
        
        return found_keywords
    
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
            'method_used': self.method,
            'semantic_similarity': f"Text semantic similarity: {result['semantic_similarity']:.3f}",
            'length_ratio': f"Description length consistency: {result['description_length_ratio']:.3f}",
            'keyword_overlap': f"Real estate keyword overlap: {result['keyword_overlap']:.3f}"
        }
        
        return explanations


# Convenience function for quick feature extraction
def extract_text_features(description_A: Union[str, None], 
                         description_B: Union[str, None],
                         model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                         scaler_path: Optional[str] = None) -> Dict:
    """
    Convenience function to extract text features from two descriptions.
    
    Args:
        description_A: First description
        description_B: Second description
        model_name: Sentence transformer model name
        scaler_path: Path to fitted scaler (optional)
        
    Returns:
        Dictionary with scaled features and individual values
    """
    extractor = TextFeatureExtractor(model_name, scaler_path)
    return extractor.extract_features(description_A, description_B)


if __name__ == "__main__":
    # Example usage and testing
    
    # Sample descriptions
    desc_1 = "Appartement 3 pièces avec balcon, cave, parking proche gare RER"
    desc_2 = "Appartement 3 pièces, balcon, cave et parking, près de la gare"
    
    # Test feature extraction
    extractor = TextFeatureExtractor()
    result = extractor.extract_features(desc_1, desc_2)
    
    print("Text Feature Extraction Test:")
    print(f"Method used: {extractor.method}")
    print(f"Scaled features: {result['scaled_features']}")
    print("\nFeature Explanation:")
    explanation = extractor.get_feature_explanation(result)
    for key, value in explanation.items():
        print(f"  {key}: {value}")