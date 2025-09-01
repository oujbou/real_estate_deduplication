"""
Configuration management 
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ProjectConfig:
    """Main project configuration"""
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    processed_dir: Path = project_root / "processed"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    # Data files
    listings_file: str = "listing.csv"
    duplicates_file: str = "duplicate_listing.csv"
    pictures_dir: str = "pictures"
    
    # Model parameters
    text_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    image_model_name: str = "resnet50"
    classifier_name: str = "lightgbm"
    
    # Feature engineering settings
    max_images_per_listing: int = 5
    image_batch_size: int = 16
    negative_sampling_ratio: int = 3
    train_test_split: float = 0.2
    random_state: int = 42
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    api_debug: bool = False
    
    # Processing settings
    n_jobs: int = -1  # Use all available cores
    verbose: bool = True
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for directory in [self.data_dir, self.processed_dir, 
                         self.models_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    @property
    def listings_path(self) -> Path:
        """Full path to listings CSV file"""
        return self.data_dir / self.listings_file
    
    @property
    def duplicates_path(self) -> Path:
        """Full path to duplicates CSV file"""
        return self.data_dir / self.duplicates_file
    
    @property
    def pictures_path(self) -> Path:
        """Full path to pictures directory"""
        return self.data_dir / self.pictures_dir
    
    @classmethod
    def from_env(cls) -> "ProjectConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if they exist
        config.api_host = os.getenv("API_HOST", config.api_host)
        config.api_port = int(os.getenv("API_PORT", config.api_port))
        config.api_debug = os.getenv("API_DEBUG", "False").lower() == "true"
        config.verbose = os.getenv("VERBOSE", "True").lower() == "true"
        
        return config

# Global configuration instance
config = ProjectConfig()

# Hardware-specific settings for M4 Pro
@dataclass
class HardwareConfig:
    
    # Memory management
    max_memory_gb: int = 16  # Leave 8GB for system
    image_processing_memory_limit: int = 2  # GB per batch

    # Image batch size and feature extraction batch size
    image_batch_size: int = 16
    feature_extraction_batch_size: int = 100
    
    # MPS (Metal Performance Shaders) settings for M4 Pro
    use_mps: bool = True
    mps_fallback_cpu: bool = True
    
    # Processing optimization
    multiprocessing_workers: int = 8  # M4 Pro has 8+4 cores, use performance cores
    image_resize_dims: tuple = (224, 224)  # Standard ResNet input size
    
    @property
    def optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory"""
        # Rough calculation: ~100MB per image in memory during processing
        available_memory_mb = self.image_processing_memory_limit * 1024
        return max(1, available_memory_mb // 100)

# Hardware configuration instance
hardware_config = HardwareConfig()