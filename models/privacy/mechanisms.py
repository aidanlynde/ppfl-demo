# models/privacy/mechanisms.py

import numpy as np
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass

@dataclass
class PrivacyMetrics:
    """Stores privacy-related metrics for visualization and analysis."""
    noise_scale: float
    clip_norm: float
    clipped_updates: int
    original_update_norms: List[float]
    clipped_update_norms: List[float]

class PrivacyMechanism:
    """Implements differential privacy mechanisms for federated learning."""
    
    def __init__(self, noise_multiplier: float = 1.0, l2_norm_clip: float = 1.0):
        """
        Initialize privacy mechanism.
        
        Args:
            noise_multiplier: Scale of noise to add (higher = more privacy)
            l2_norm_clip: Maximum L2 norm for gradient clipping (lower = more privacy)
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        
    def compute_update_norm(self, update: List[np.ndarray]) -> float:
        """Compute the L2 norm of an update."""
        squared_sum = 0.0
        for layer in update:
            squared_sum += np.sum(np.square(layer))
        return np.sqrt(squared_sum)
    
    def apply_privacy(self, 
                     model_updates: List[List[np.ndarray]], 
                     batch_size: int) -> Tuple[List[np.ndarray], PrivacyMetrics]:
        """
        Apply privacy mechanisms to model updates.
        
        Args:
            model_updates: List of weight updates from each client
            batch_size: Size of training batch (for noise scaling)
            
        Returns:
            Tuple of (private_aggregate, privacy_metrics)
        """
        # Track metrics for visualization
        original_norms = []
        clipped_norms = []
        num_clipped = 0
        
        # Clip gradients
        clipped_updates = []
        for update in model_updates:
            original_norm = self.compute_update_norm(update)
            original_norms.append(float(original_norm))
            
            # Determine if clipping is needed (with numerical tolerance)
            needs_clipping = original_norm > self.l2_norm_clip * (1 + 1e-6)
            if needs_clipping:
                num_clipped += 1
                scaling_factor = self.l2_norm_clip / original_norm
            else:
                scaling_factor = 1.0
                
            clipped_update = [w * scaling_factor for w in update]
            clipped_norm = self.compute_update_norm(clipped_update)
            clipped_norms.append(float(clipped_norm))
            
            clipped_updates.append(clipped_update)
        
        # Average the updates
        num_clients = len(model_updates)
        averaged_update = [
            sum(update[i] for update in clipped_updates) / num_clients
            for i in range(len(clipped_updates[0]))
        ]
        
        # Add noise
        noise_stddev = self.noise_multiplier * self.l2_norm_clip / batch_size
        noised_update = [
            param + np.random.normal(0, noise_stddev, param.shape)
            for param in averaged_update
        ]
        
        # Collect metrics
        metrics = PrivacyMetrics(
            noise_scale=noise_stddev,
            clip_norm=self.l2_norm_clip,
            clipped_updates=num_clipped,
            original_update_norms=original_norms,
            clipped_update_norms=clipped_norms
        )
        
        return noised_update, metrics
    
    def get_privacy_spent(self, num_steps: int, batch_size: int, dataset_size: int) -> Dict[str, float]:
        """
        Calculate current privacy spending (epsilon).
        This is a simplified privacy accounting method for demonstration.
        
        Args:
            num_steps: Number of training steps
            batch_size: Size of training batches
            dataset_size: Total size of dataset
            
        Returns:
            Dictionary containing privacy budget spent
        """
        # Simplified privacy calculation for demonstration
        # In practice, you'd use a more sophisticated privacy accountant
        sampling_rate = batch_size / dataset_size
        effective_noise = self.noise_multiplier / sampling_rate
        
        # Approximate privacy loss
        epsilon = np.sqrt(2 * np.log(1/1e-5)) * (1 / effective_noise)
        epsilon *= np.sqrt(num_steps)
        
        return {
            'epsilon': float(epsilon),
            'delta': 1e-5,  # Fixed delta for demonstration
            'noise_multiplier': self.noise_multiplier,
            'l2_norm_clip': self.l2_norm_clip
        }
    
    def update_parameters(self, noise_multiplier: float, l2_norm_clip: float) -> None:
        """
        Update privacy parameters.
        
        Args:
            noise_multiplier: New noise scale
            l2_norm_clip: New clipping threshold
        """
        if noise_multiplier is not None:
            self.noise_multiplier = float(noise_multiplier)
        if l2_norm_clip is not None:
            self.l2_norm_clip = float(l2_norm_clip)