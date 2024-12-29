# models/federated/private_fl_manager.py

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from models.federated.fl_manager import FederatedLearningManager
from models.privacy.mechanisms import PrivacyMechanism, PrivacyMetrics
from utils import MNISTDataHandler
from api.utils.logger_config import logger

@dataclass
class TrainingMetrics:
    """Stores training and privacy metrics for visualization."""
    round_number: int
    client_metrics: Dict[int, Dict[str, float]]
    global_metrics: Dict[str, float]
    privacy_metrics: PrivacyMetrics
    privacy_budget: Dict[str, float]

class PrivateFederatedLearningManager(FederatedLearningManager):
    """Privacy-aware Federated Learning Manager with differential privacy."""
    
    def __init__(
        self,
        num_clients: int = 5,
        local_epochs: int = 1,
        batch_size: int = 32,
        rounds: int = 10,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0,
        test_mode: bool = False
    ):
        """
        Initialize the private federated learning manager.
        
        Args:
            num_clients: Number of clients participating in training
            local_epochs: Number of local training epochs per round
            batch_size: Training batch size for each client
            rounds: Number of federated learning rounds
            noise_multiplier: Scale of noise for differential privacy
            l2_norm_clip: Gradient clipping threshold
            test_mode: Whether to use reduced dataset for testing
        """
        super().__init__(
            num_clients=num_clients,
            local_epochs=local_epochs,
            batch_size=batch_size,
            rounds=rounds,
            test_mode=test_mode
        )

        self.current_round = 0

        self.local_epochs = 1 if test_mode else local_epochs  # Always use 1 epoch in test mode
        self.batch_size = 16 if test_mode else batch_size
        
        # Initialize privacy mechanism
        self.privacy_mechanism = PrivacyMechanism(
            noise_multiplier=noise_multiplier,
            l2_norm_clip=l2_norm_clip
        )
        
        # Track total training steps for privacy accounting
        self.total_steps = 0
        
        # Enhanced training history with privacy metrics
        self.history = {
            'rounds': [],
            'training_metrics': [],
            'privacy_metrics': [],
            'privacy_budget': []
        }
    
    def _aggregate_weights(self, client_weights: List[List[np.ndarray]]) -> None:
        """
        Aggregate weights using privacy mechanism.
        
        Args:
            client_weights: List of weight updates from each client
        """
        # Convert weights to updates (difference from global weights)
        global_weights = self.global_model.get_weights()
        weight_updates = [
            [cw - gw for cw, gw in zip(client_w, global_weights)]
            for client_w in client_weights
        ]
        
        # Apply privacy mechanisms
        private_update, privacy_metrics = self.privacy_mechanism.apply_privacy(
            weight_updates,
            self.batch_size
        )
        
        # Apply private update to global model
        new_weights = [gw + upd for gw, upd in zip(global_weights, private_update)]
        self.global_model.set_weights(new_weights)
        
        return privacy_metrics
    
    @router.post("/train_round")
    @with_retry(max_retries=3)
    async def train_round(x_session_id: Optional[str] = Header(None)) -> Dict[str, Any]:
        try:
            logger.info(f"Starting train_round for session {x_session_id}")
            
            # First verify session exists and load it
            session = validate_session(x_session_id)
            
            # Verify FL manager exists
            if not session.fl_manager:
                logger.error(f"FL manager not found in session {x_session_id}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Training not initialized - please initialize first"
                )
            
            # Log current state before training
            logger.info(f"Current round before training: {session.fl_manager.current_round}")
            
            # Verify FL manager is ready
            if not session.fl_manager.is_ready_for_training():
                logger.warning(f"FL manager not ready in session {x_session_id}, attempting reload")
                # Try to reload session from disk
                reloaded_session = session_manager._load_session(x_session_id)
                if reloaded_session:
                    session = Session.from_dict(reloaded_session)
                    session_manager.sessions[x_session_id] = session
                    logger.info("Successfully reloaded session from disk")
                    
                if not session.fl_manager or not session.fl_manager.is_ready_for_training():
                    logger.error("Training not properly initialized even after reload attempt")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Training not properly initialized"
                    )
                
            # Run training round
            metrics = session.fl_manager.train_round()
            logger.info(f"Training completed, round after: {session.fl_manager.current_round}")
            
            # Immediately persist updated session
            session_manager._persist_session(session)
            logger.info(f"Session persisted after training round {session.fl_manager.current_round-1}")
            
            return {
                "status": "success",
                "metrics": metrics
            }
        except HTTPException as he:
            logger.error(f"HTTP exception in train_round: {he.detail}")
            return JSONResponse(
                status_code=he.status_code,
                content={
                    "status": "error",
                    "message": he.detail
                }
            )
        except Exception as e:
            logger.error(f"Error in train_round: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "status": "error",
                    "message": str(e),
                    "detail": "Training failed - please check initialization status"
                }
            )
    
    def update_privacy_parameters(
        self,
        noise_multiplier: Optional[float] = None,
        l2_norm_clip: Optional[float] = None
    ) -> None:
        """
        Update privacy parameters during training.
        
        Args:
            noise_multiplier: New noise scale (optional)
            l2_norm_clip: New clipping threshold (optional)
        """
        try:
            if noise_multiplier is not None or l2_norm_clip is not None:
                logger.info(f"Updating privacy parameters: noise={noise_multiplier}, clip={l2_norm_clip}")
                
                # Get current values for any parameter not being updated
                current_noise = self.privacy_mechanism.noise_multiplier
                current_clip = self.privacy_mechanism.l2_norm_clip
                
                # Update with new or current values
                self.privacy_mechanism.update_parameters(
                    noise_multiplier=noise_multiplier if noise_multiplier is not None else current_noise,
                    l2_norm_clip=l2_norm_clip if l2_norm_clip is not None else current_clip
                )
                
                logger.info(f"Updated privacy parameters: noise={self.privacy_mechanism.noise_multiplier}, clip={self.privacy_mechanism.l2_norm_clip}")
        except Exception as e:
            logger.error(f"Error updating privacy parameters: {str(e)}")
            raise
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Get current privacy metrics and budget.
        
        Returns:
            Dictionary containing privacy metrics
        """
        if not self.history['privacy_metrics']:
            return {
                'current_epsilon': 0.0,
                'current_delta': 1e-5,
                'noise_multiplier': self.privacy_mechanism.noise_multiplier,
                'l2_norm_clip': self.privacy_mechanism.l2_norm_clip,
                'total_steps': self.total_steps,
                'clipped_updates_history': []
            }
            
        latest_budget = self.history['privacy_budget'][-1]
        latest_metrics = self.history['privacy_metrics'][-1]
        
        return {
            'current_epsilon': latest_budget['epsilon'],
            'current_delta': latest_budget['delta'],
            'noise_multiplier': self.privacy_mechanism.noise_multiplier,
            'l2_norm_clip': self.privacy_mechanism.l2_norm_clip,
            'total_steps': self.total_steps,
            'clipped_updates_history': [
                m.clipped_updates for m in self.history['privacy_metrics']
            ]
        }

    def is_ready_for_training(self) -> bool:
        """Check if manager is properly initialized and ready for training."""
        try:
            return (
                self.data_handler is not None and
                self.global_model is not None and
                len(self.client_models) == self.num_clients and
                self.privacy_mechanism is not None
            )
        except Exception as e:
            logger.error(f"Error checking training readiness: {str(e)}")
            return False

    def __getstate__(self):
        """Custom serialization."""
        try:
            state = self.__dict__.copy()
            # Handle client models
            if 'client_models' in state:
                state['client_models'] = {
                    k: v.get_weights() if v is not None else None
                    for k, v in state['client_models'].items()
                }
            # Handle global model
            if 'global_model' in state:
                state['global_model'] = (
                    self.global_model.get_weights() 
                    if self.global_model is not None else None
                )
            # Store privacy mechanism state explicitly
            if 'privacy_mechanism' in state:
                state['privacy_mechanism_state'] = {
                    'noise_multiplier': self.privacy_mechanism.noise_multiplier,
                    'l2_norm_clip': self.privacy_mechanism.l2_norm_clip
                }
                state.pop('privacy_mechanism')
            
            # Remove data handler - will be reinitialized
            if 'data_handler' in state:
                state.pop('data_handler')
                
            logger.info(f"Serializing state with keys: {list(state.keys())}")
            return state
        except Exception as e:
            logger.error(f"Error in serialization: {str(e)}", exc_info=True)
            raise

    def __setstate__(self, state):
        """Custom deserialization."""
        try:
            # Extract model weights and privacy state before update
            client_weights = state.pop('client_models', {})
            global_weights = state.pop('global_model', None)
            privacy_state = state.pop('privacy_mechanism_state', None)
            
            # Update state
            self.__dict__.update(state)
            
            # Reinitialize components
            self._initialize_setup()
            
            # Restore model weights
            if global_weights is not None:
                self.global_model.set_weights(global_weights)
                
            for client_id, weights in client_weights.items():
                if weights is not None and client_id in self.client_models:
                    self.client_models[client_id].set_weights(weights)
            
            # Restore privacy mechanism state
            if privacy_state:
                self.privacy_mechanism.update_parameters(
                    noise_multiplier=privacy_state['noise_multiplier'],
                    l2_norm_clip=privacy_state['l2_norm_clip']
                )
                
            logger.info("Successfully deserialized state")
        except Exception as e:
            logger.error(f"Error in deserialization: {str(e)}", exc_info=True)
            raise
    
    def validate_state(self):
        """Validate internal state consistency."""
        try:
            is_valid = (
                self.data_handler is not None and
                self.global_model is not None and
                len(self.client_models) == self.num_clients and
                self.privacy_mechanism is not None and
                isinstance(self.history, dict) and
                all(k in self.history for k in ['rounds', 'training_metrics', 'privacy_metrics', 'privacy_budget'])
            )
            
            if not is_valid:
                logger.warning("Invalid state detected during validation")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating state: {str(e)}")
            return False

    def reset(self):
        try:
            # Store current configuration
            config = {
                'num_clients': self.num_clients,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'rounds': self.rounds,
                'noise_multiplier': self.privacy_mechanism.noise_multiplier,  # This is correct
                'l2_norm_clip': self.privacy_mechanism.l2_norm_clip,         # This is correct
                'test_mode': self.test_mode
            }
            
            # Need to store these before reinitializing
            current_noise = self.privacy_mechanism.noise_multiplier
            current_clip = self.privacy_mechanism.l2_norm_clip
            
            # Reinitialize models and data
            self._initialize_setup()
            
            # Reset training state
            self.current_round = 0
            self.total_steps = 0
            self.history = {
                'rounds': [],
                'training_metrics': [],
                'privacy_metrics': [],
                'privacy_budget': []
            }
            
            # Restore privacy configuration using stored values
            self.privacy_mechanism.update_parameters(
                noise_multiplier=current_noise,
                l2_norm_clip=current_clip
            )
        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")
            raise
