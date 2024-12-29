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
        # Initialize privacy mechanism
        self.privacy_mechanism = PrivacyMechanism(
            noise_multiplier=noise_multiplier,
            l2_norm_clip=l2_norm_clip
        )

        self.current_round = 0
        self.total_steps = 0

        # Enhanced training history with privacy metrics
        self.history = {
            'rounds': [],
            'training_metrics': [],
            'privacy_metrics': [],
            'privacy_budget': []
        }

        super().__init__(
            num_clients=num_clients,
            local_epochs=local_epochs,
            batch_size=batch_size,
            rounds=rounds,
            test_mode=test_mode
        )
        
        self.local_epochs = 1 if test_mode else local_epochs  # Always use 1 epoch in test mode
        self.batch_size = 16 if test_mode else batch_size


    def _initialize_setup(self) -> None:
        """Initialize data handler and models."""
        try:
            # Store current privacy mechanism if it exists
            if hasattr(self, 'privacy_mechanism'):
                current_privacy = self.privacy_mechanism
            else:
                current_privacy = None
            
            # Call parent's initialize setup to handle models and data
            super()._initialize_setup()
            
            # Restore privacy mechanism if it existed, otherwise create new one
            if current_privacy is not None:
                self.privacy_mechanism = current_privacy
            else:
                self.privacy_mechanism = PrivacyMechanism(
                    noise_multiplier=1.0,
                    l2_norm_clip=1.0
                )
                
            logger.debug("Successfully initialized private FL setup")
            
        except Exception as e:
            logger.error(f"Error in private FL _initialize_setup: {str(e)}")
            raise


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
    
    def train_round(self) -> TrainingMetrics:
        """Execute one round of private federated learning."""
        try:
            logger.info(f"Starting training round, current_round={self.current_round}")
        
            if not self.validate_state():
                logger.error("Invalid state detected, attempting recovery")
                try:
                    self._initialize_setup()
                    if not self.validate_state():
                        raise ValueError("Training state invalid and recovery failed")
                except Exception as e:
                    logger.error(f"Recovery failed: {str(e)}")
                    raise

            # Distribute global model weights to all clients
            self._distribute_weights()
            
            # Train each client locally
            client_metrics = {}
            client_weights = []
            for client_id, client_model in self.client_models.items():
                # Train client
                metrics = self._train_client(client_id)
                client_metrics[client_id] = metrics
                
                # Collect weights
                client_weights.append(client_model.get_weights())
                
                # Update step count for privacy accounting
                self.total_steps += self.local_epochs * (len(self.data_handler.get_client_data(client_id)['x_train']) // self.batch_size)
            
            # Aggregate weights with privacy
            privacy_metrics = self._aggregate_weights(client_weights)
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            
            # Calculate privacy budget
            privacy_budget = self.privacy_mechanism.get_privacy_spent(
                num_steps=self.total_steps,
                batch_size=self.batch_size,
                dataset_size=len(self.data_handler.x_train)
            )
            
            # Create metrics for this round using current_round
            round_metrics = TrainingMetrics(
                round_number=self.current_round,
                client_metrics=client_metrics,
                global_metrics=global_metrics,
                privacy_metrics=privacy_metrics,
                privacy_budget=privacy_budget
            )
            
            # Update history
            self.history['rounds'].append(self.current_round)
            self.history['training_metrics'].append(round_metrics)
            self.history['privacy_metrics'].append(privacy_metrics)
            self.history['privacy_budget'].append(privacy_budget)
            
            self.current_round += 1
            
            return round_metrics

        except Exception as e:
            logger.error(f"Error in train_round: {str(e)}")
            raise
    
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
            basic_checks = (
                self.data_handler is not None and
                self.global_model is not None and
                len(self.client_models) == self.num_clients and
                self.privacy_mechanism is not None
            )
            
            privacy_checks = (
                hasattr(self.privacy_mechanism, 'noise_multiplier') and
                hasattr(self.privacy_mechanism, 'l2_norm_clip')
            )
            
            history_checks = isinstance(self.history, dict) and all(
                k in self.history for k in ['rounds', 'training_metrics', 'privacy_metrics', 'privacy_budget']
            )
            
            if not all([basic_checks, privacy_checks, history_checks]):
                logger.error(f"Validation failed - Basic: {basic_checks}, Privacy: {privacy_checks}, History: {history_checks}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking training readiness: {str(e)}")
            return False

    def __getstate__(self):
        """Custom serialization."""
        try:
            state = self.__dict__.copy()

            required_attrs = ['current_round', 'total_steps', 'privacy_mechanism', 'history']
            missing = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing:
                raise ValueError(f"Missing required attributes: {missing}")

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
        try:
            # Extract states first
            client_weights = state.pop('client_models', {})
            global_weights = state.pop('global_model', None)
            privacy_state = state.pop('privacy_mechanism_state', None)
            
            # First create privacy mechanism with privacy state values (not default values)
            if privacy_state:
                self.privacy_mechanism = PrivacyMechanism(
                    noise_multiplier=privacy_state['noise_multiplier'],
                    l2_norm_clip=privacy_state['l2_norm_clip']
                )
            else:
                self.privacy_mechanism = PrivacyMechanism(
                    noise_multiplier=1.0,
                    l2_norm_clip=1.0
                )

            # Update state
            self.__dict__.update(state)
            
            # Setup after state is updated
            self._initialize_setup()
            
            # Restore model weights
            if global_weights is not None:
                self.global_model.set_weights(global_weights)
                    
            for client_id, weights in client_weights.items():
                if weights is not None and client_id in self.client_models:
                    self.client_models[client_id].set_weights(weights)
                    
            logger.info("Successfully deserialized state")
        except Exception as e:
            logger.error(f"Error in deserialization: {str(e)}", exc_info=True)
            raise
    
    def validate_state(self):
        """Validate internal state consistency."""
        try:
            if not self._validate_privacy_state():
                return False

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

    def _validate_privacy_state(self) -> bool:
        """Validate privacy mechanism state."""
        try:
            if not hasattr(self, 'privacy_mechanism'):
                logger.error("No privacy mechanism found")
                return False
                
            required_attrs = ['noise_multiplier', 'l2_norm_clip']
            for attr in required_attrs:
                if not hasattr(self.privacy_mechanism, attr):
                    logger.error(f"Privacy mechanism missing attribute: {attr}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating privacy state: {str(e)}")
            return False

    def reset(self):
        try:
            if not hasattr(self, 'privacy_mechanism'):
                raise ValueError("Cannot reset: privacy mechanism not initialized")
            
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
