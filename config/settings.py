# config/settings.py
from typing import List, Tuple
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PPFL Demo"
    ENVIRONMENT: str = "development"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://yourdomain.com"  # Replace with your deployed Next.js domain
    ]
    
    # Model Settings
    NUM_CLIENTS: int = 5
    ROUNDS: int = 10
    BATCH_SIZE: int = 32
    EPOCHS_PER_ROUND: int = 1
    
    # Privacy Settings
    NOISE_MULTIPLIER: float = 1.0
    L2_NORM_CLIP: float = 1.0
    
    # MNIST Dataset Settings
    NUM_CLASSES: int = 10
    INPUT_SHAPE: Tuple[int, int, int] = (28, 28, 1)
    
    class Config:
        case_sensitive = True

settings = Settings()