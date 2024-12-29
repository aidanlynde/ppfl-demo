# api/routers/fl_routes.py

import logging
from fastapi import APIRouter, HTTPException, Header, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Dict, Any, Optional, List
from models.federated.private_fl_manager import PrivateFederatedLearningManager
from ..utils.session_manager import session_manager, Session
from ..utils.retry import with_retry
from api.utils.logger_config import logger


router = APIRouter(tags=["Federated Learning"])

class TrainingConfig(BaseModel):
    """Training configuration parameters with validation."""
    num_clients: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1.0

    @validator('num_clients')
    def validate_num_clients(cls, v):
        if not 2 <= v <= 100:
            raise ValueError('Number of clients must be between 2 and 100')
        return v

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1:
            raise ValueError('Batch size must be positive')
        return v

class PrivacyConfig(BaseModel):
    """Privacy parameter updates with validation."""
    noise_multiplier: Optional[float] = None
    l2_norm_clip: Optional[float] = None

    @validator('noise_multiplier')
    def validate_noise(cls, v):
        if v is not None and v < 0:
            raise ValueError('Noise multiplier must be non-negative')
        return v

    @validator('l2_norm_clip')
    def validate_clip(cls, v):
        if v is not None and v <= 0:
            raise ValueError('L2 norm clip must be positive')
        return v

class ClientInfo(BaseModel):
    """Client information for tooltips."""
    client_id: int
    data_size: int
    training_progress: List[Dict[str, float]]
    privacy_impact: Dict[str, float]

def validate_session(session_id: str):
    """Validate session and return session object or raise HTTPException."""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No session ID provided"
        )
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )
    return session

@router.post("/initialize", status_code=status.HTTP_201_CREATED)
async def initialize_training(
    config: TrainingConfig,
    x_session_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Initialize the federated learning process."""
    try:
        session = validate_session(x_session_id)
        
        session.fl_manager = PrivateFederatedLearningManager(
            num_clients=config.num_clients,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            noise_multiplier=config.noise_multiplier,
            l2_norm_clip=config.l2_norm_clip
        )

        if not session.fl_manager.is_ready_for_training():
            raise ValueError("FL manager failed initialization check")
        
        # Explicitly persist session after initialization
        session_manager._persist_session(session)
        
        logger.info("Initialized FL training for session %s with config: %s", 
                   x_session_id, config.dict())
        
        return {
            "status": "success",
            "message": "Federated learning initialized",
            "config": config.dict()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error initializing training: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/test-train")
async def test_train() -> Dict[str, Any]:
    """Test endpoint with minimal processing."""
    try:
        logger.info("Running test training...")
        return {
            "status": "success",
            "message": "Test training completed"
        }
    except Exception as e:
        logger.error("Test training error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/train_round")
@with_retry(max_retries=3)
async def train_round(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    try:
        # First verify session exists and load it
        logger.info(f"Starting train_round for session {x_session_id}")
        session = validate_session(x_session_id)
        
        # Verify FL manager exists
        if not session.fl_manager:
            logger.error(f"FL manager not found for session {x_session_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized - please initialize first"
            )
        
        logger.info(f"Current round before training: {session.fl_manager.current_round}")
        
        # Verify FL manager is ready
        if not session.fl_manager.is_ready_for_training():
            logger.error(f"FL manager not ready for training in session {x_session_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not properly initialized"
            )
            
        # Run training round
        metrics = session.fl_manager.train_round()
        
        # Immediately persist updated session
        session_manager._persist_session(session)
        logger.info(f"Completed training round for session {x_session_id}")
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_round: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e)
            }
        )

@router.post("/update_privacy")
async def update_privacy(
    config: PrivacyConfig,
    x_session_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Update privacy parameters with validation."""
    try:
        session = validate_session(x_session_id)
        
        if not session.fl_manager:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized"
            )
        
        session.fl_manager.update_privacy_parameters(
            noise_multiplier=config.noise_multiplier,
            l2_norm_clip=config.l2_norm_clip
        )

        session_manager._persist_session(session)
        
        current_metrics = session.fl_manager.get_privacy_metrics()
        logger.info("Updated privacy parameters for session %s: %s", 
                   x_session_id, current_metrics)
        
        return {
            "status": "success",
            "message": "Privacy parameters updated",
            "current_metrics": current_metrics
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error updating privacy parameters: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/metrics")
@with_retry(max_retries=3)
async def get_metrics(
    response: Response,
    x_session_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Get current training and privacy metrics with caching."""
    try:
        session = validate_session(x_session_id)
        
        if not session.fl_manager:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized"
            )
        
        metrics = {
            "status": "success",
            "training_history": session.fl_manager.history,
            "privacy_metrics": session.fl_manager.get_privacy_metrics()
        }
        
        # Add caching headers
        response.headers["Cache-Control"] = "max-age=5"
        
        return metrics
    except Exception as e:
        logger.error("Error getting metrics: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/current_state")
async def get_current_state(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current training state and configuration."""
    try:
        session = validate_session(x_session_id)
        
        if not session.fl_manager:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized"
            )
        
        state = {
            "status": "success",
            "current_round": session.fl_manager.current_round,
            "total_rounds": session.fl_manager.rounds,
            "privacy_settings": {
                "noise_multiplier": session.fl_manager.privacy_mechanism.noise_multiplier,
                "l2_norm_clip": session.fl_manager.privacy_mechanism.l2_norm_clip
            },
            "training_active": bool(session.fl_manager),
            "latest_accuracy": (
                session.fl_manager.history.get('training_metrics', [])[-1].global_metrics.get('test_accuracy')
                if session.fl_manager.history.get('training_metrics', []) else None
            )
        }
        
        logger.debug("Current state for session %s: %s", x_session_id, state)
        return state
        
    except Exception as e:
        logger.error("Error getting current state: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/client_info/{client_id}")
async def get_client_info(
    client_id: int,
    x_session_id: Optional[str] = Header(None)
) -> ClientInfo:
    """Get detailed information about a specific client."""
    try:
        session = validate_session(x_session_id)
        
        if not session.fl_manager:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized"
            )
        
        try:
            client_data = session.fl_manager.data_handler.get_client_data(client_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client {client_id} not found"
            )
        
        # Get all training progress including history
        training_progress = []
        for metrics in session.fl_manager.history['training_metrics']:
            if client_id in metrics.client_metrics:
                progress = {
                    'round': metrics.round_number,
                    'accuracy': metrics.client_metrics[client_id]['accuracy'],
                    'loss': metrics.client_metrics[client_id]['loss']
                }
                training_progress.append(progress)

        # Get latest privacy metrics with proper null handling
        try:
            privacy_metrics = session.fl_manager.history['privacy_metrics'][-1] if session.fl_manager.history['privacy_metrics'] else None
            privacy_impact = {
                'updates_clipped': float(privacy_metrics.clipped_updates) if privacy_metrics else 0.0,
                'original_norm': float(privacy_metrics.original_update_norms[client_id]) 
                    if privacy_metrics and client_id < len(privacy_metrics.original_update_norms) else 0.0,
                'clipped_norm': float(privacy_metrics.clipped_update_norms[client_id])
                    if privacy_metrics and client_id < len(privacy_metrics.clipped_update_norms) else 0.0
            }
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error getting privacy metrics for client {client_id}: {str(e)}")
            privacy_impact = {
                'updates_clipped': 0.0,
                'original_norm': 0.0,
                'clipped_norm': 0.0
            }
        
        # Get client's data size
        data_size = len(client_data['x_train']) if client_data and 'x_train' in client_data else 0
        
        return ClientInfo(
            client_id=client_id,
            data_size=data_size,
            training_progress=training_progress,
            privacy_impact=privacy_impact
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting client info: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting client info: {str(e)}"
        )

@router.post("/reset")
async def reset_training(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Reset the training process while maintaining configuration."""
    try:
        session = validate_session(x_session_id)
        
        if not session.fl_manager:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training not initialized"
            )
        
        # Store current configuration
        config = {
            'num_clients': session.fl_manager.num_clients,
            'local_epochs': session.fl_manager.local_epochs,
            'batch_size': session.fl_manager.batch_size,
            'rounds': session.fl_manager.rounds,  # Add rounds to config
            'noise_multiplier': session.fl_manager.privacy_mechanism.noise_multiplier,
            'l2_norm_clip': session.fl_manager.privacy_mechanism.l2_norm_clip,
            'test_mode': session.fl_manager.test_mode  # Add test_mode to config
        }
        
        # Reinitialize with same configuration
        session.fl_manager = PrivateFederatedLearningManager(**config)
        session_manager._persist_session(session)
        
        logger.info("Reset training for session %s with config: %s", 
                   x_session_id, config)
        
        return {
            "status": "success",
            "message": "Training reset successfully",
            "config": config
        }
    except Exception as e:
        logger.error("Error resetting training: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/explanations/{concept}")
async def get_explanation(concept: str) -> Dict[str, Any]:
    """Get explanatory content for various FL and privacy concepts."""
    explanations = {
        "federated_learning": {
            "title": "What is Federated Learning?",
            "content": "Federated learning enables training AI models without sharing raw data. Instead, each client trains locally and shares only model updates.",
            "key_points": [
                "Privacy preservation - raw data stays with clients",
                "Collaborative learning - multiple parties contribute to model improvement",
                "Decentralized approach - no need for centralized data collection"
            ]
        },
        "differential_privacy": {
            "title": "Understanding Differential Privacy",
            "content": "Differential privacy protects individual privacy by adding controlled noise to model updates, making it difficult to extract information about any single participant.",
            "key_points": [
                "Noise addition - random noise masks individual contributions",
                "Privacy budget - tracks and limits information disclosure",
                "Controlled data exposure - mathematically proven privacy guarantees"
            ]
        },
        "model_averaging": {
            "title": "Federated Averaging (FedAvg)",
            "content": "FedAvg is the core algorithm that combines model updates from all clients into a single global model.",
            "key_points": [
                "Weight averaging - combines client models thoughtfully",
                "Communication efficiency - only model updates are shared",
                "Scalability - works with many clients"
            ]
        }
    }
    
    if concept not in explanations:
        raise HTTPException(status_code=404, detail="Concept not found")
    
    return explanations[concept]