# api/routers/fl_routes.py

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from models.federated.private_fl_manager import PrivateFederatedLearningManager
from ..utils.session_manager import session_manager

router = APIRouter()


class TrainingConfig(BaseModel):
    """Training configuration parameters."""
    num_clients: int = 5
    local_epochs: int = 1
    batch_size: int = 32
    noise_multiplier: float = 1.0
    l2_norm_clip: float = 1.0

class PrivacyConfig(BaseModel):
    """Privacy parameter updates."""
    noise_multiplier: Optional[float] = None
    l2_norm_clip: Optional[float] = None

class ClientInfo(BaseModel):
    """Client information for tooltips."""
    client_id: int
    data_size: int
    training_progress: List[Dict[str, float]]
    privacy_impact: Dict[str, float]

@router.post("/initialize")
async def initialize_training(
    config: TrainingConfig,
    x_session_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Initialize the federated learning process."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    try:
        session.fl_manager = PrivateFederatedLearningManager(
            num_clients=config.num_clients,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            noise_multiplier=config.noise_multiplier,
            l2_norm_clip=config.l2_norm_clip
        )
        
        return {
            "status": "success",
            "message": "Federated learning initialized",
            "config": config.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-train")
async def test_train():
    """Test endpoint with minimal processing."""
    try:
        print("Testing minimal training...")
        return {
            "status": "success",
            "message": "Test training completed"
        }
    except Exception as e:
        print(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train_round")
async def train_round(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Execute one round of federated learning."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        metrics = session.fl_manager.train_round()
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during training round: {str(e)}")
        print(f"Error traceback: {error_details}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "details": error_details
            }
        )

@router.post("/update_privacy")
async def update_privacy(
    config: PrivacyConfig,
    x_session_id: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Update privacy parameters."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        session.fl_manager.update_privacy_parameters(
            noise_multiplier=config.noise_multiplier,
            l2_norm_clip=config.l2_norm_clip
        )
        
        return {
            "status": "success",
            "message": "Privacy parameters updated",
            "current_metrics": session.fl_manager.get_privacy_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current training and privacy metrics."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        return {
            "status": "success",
            "training_history": session.fl_manager.history,
            "privacy_metrics": session.fl_manager.get_privacy_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/current_state")
async def get_current_state(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current training state and configuration."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        return {
            "status": "success",
            "current_round": len(session.fl_manager.history['rounds']),
            "total_rounds": session.fl_manager.rounds,
            "privacy_settings": {
                "noise_multiplier": session.fl_manager.privacy_mechanism.noise_multiplier,
                "l2_norm_clip": session.fl_manager.privacy_mechanism.l2_norm_clip
            },
            "training_active": True if session.fl_manager else False,
            "latest_accuracy": session.fl_manager.history['training_metrics'][-1].global_metrics['test_accuracy']
            if session.fl_manager.history['training_metrics'] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/client_info/{client_id}")
async def get_client_info(
    client_id: int,
    x_session_id: Optional[str] = Header(None)
) -> ClientInfo:
    """Get detailed information about a specific client."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        client_data = session.fl_manager.data_handler.get_client_data(client_id)
        training_progress = [
            {
                'round': i,
                'accuracy': metrics.client_metrics[client_id]['accuracy'],
                'loss': metrics.client_metrics[client_id]['loss']
            }
            for i, metrics in enumerate(session.fl_manager.history['training_metrics'])
            if client_id in metrics.client_metrics
        ]
        
        privacy_metrics = session.fl_manager.history['privacy_metrics'][-1] if session.fl_manager.history['privacy_metrics'] else None
        privacy_impact = {
            'updates_clipped': privacy_metrics.clipped_updates if privacy_metrics else 0,
            'original_norm': privacy_metrics.original_update_norms[client_id] if privacy_metrics else 0.0,
            'clipped_norm': privacy_metrics.clipped_update_norms[client_id] if privacy_metrics else 0.0
        }
        
        return ClientInfo(
            client_id=client_id,
            data_size=len(client_data['x_train']),
            training_progress=training_progress,
            privacy_impact=privacy_impact
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_training(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Reset the training process while maintaining configuration."""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="No session ID provided")
    
    session = session_manager.get_session(x_session_id)
    if not session or not session.fl_manager:
        raise HTTPException(status_code=401, detail="Invalid session or not initialized")
    
    try:
        # Store current configuration
        config = {
            'num_clients': session.fl_manager.num_clients,
            'local_epochs': session.fl_manager.local_epochs,
            'batch_size': session.fl_manager.batch_size,
            'noise_multiplier': session.fl_manager.privacy_mechanism.noise_multiplier,
            'l2_norm_clip': session.fl_manager.privacy_mechanism.l2_norm_clip
        }
        
        # Reinitialize with same configuration
        session.fl_manager = PrivateFederatedLearningManager(**config)
        
        return {
            "status": "success",
            "message": "Training reset successfully",
            "config": config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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