# models/federated/private_fl_manager.py

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from models.federated.fl_manager import FederatedLearningManager
from models.privacy.mechanisms import PrivacyMechanism, PrivacyMetrics
from utils import MNISTDataHandler

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
    
    def train_round(self) -> TrainingMetrics:
        """
        Execute one round of private federated learning.
        
        Returns:
            TrainingMetrics containing round results
        """
        try:
            current_round = len(self.history['rounds'])

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
            
            # Create metrics for this round
            round_metrics = TrainingMetrics(
                round_number=current_round,
                client_metrics=client_metrics,
                global_metrics=global_metrics,
                privacy_metrics=privacy_metrics,
                privacy_budget=privacy_budget
            )
            
            # Update history
            self.history['rounds'].append(current_round)
            self.history['training_metrics'].append(round_metrics)
            self.history['privacy_metrics'].append(privacy_metrics)
            self.history['privacy_budget'].append(privacy_budget)
            
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
        if noise_multiplier is not None or l2_norm_clip is not None:
            current_noise = self.privacy_mechanism.noise_multiplier
            current_clip = self.privacy_mechanism.l2_norm_clip
            
            self.privacy_mechanism.update_parameters(
                noise_multiplier=noise_multiplier if noise_multiplier is not None else current_noise,
                l2_norm_clip=l2_norm_clip if l2_norm_clip is not None else current_clip
            )
    
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
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom deserialization."""
        self.__dict__.update(state)

    def reset(self):
        """Reset training while maintaining configuration."""
        current_privacy_config = {
            'noise_multiplier': self.privacy_mechanism.noise_multiplier,
            'l2_norm_clip': self.privacy_mechanism.l2_norm_clip
        }
        
        # Initialize with original config
        super().__init__(
            num_clients=self.num_clients,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            rounds=self.rounds,
            test_mode=self.test_mode
        )
        
        # Restore privacy settings
        self.update_privacy_parameters(**current_privacy_config)
        
        # Clear history
        self.history = {
            'rounds': [],
            'training_metrics': [],
            'privacy_metrics': [],
            'privacy_budget': []
        }
