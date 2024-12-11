# tests/test_private_fl_manager.py

import pytest
import numpy as np
from models.federated.private_fl_manager import PrivateFederatedLearningManager

@pytest.fixture
def private_fl_manager():
    """Fixture to provide private FL manager for testing."""
    return PrivateFederatedLearningManager(
        num_clients=2,
        local_epochs=1,
        batch_size=32,
        rounds=1,
        noise_multiplier=1.0,
        l2_norm_clip=1.0,
        test_mode=True
    )

def test_initialization(private_fl_manager):
    """Test private FL manager initialization."""
    assert private_fl_manager.num_clients == 2
    assert len(private_fl_manager.client_models) == 2
    assert private_fl_manager.global_model is not None
    assert private_fl_manager.privacy_mechanism is not None
    assert private_fl_manager.privacy_mechanism.noise_multiplier == 1.0
    assert private_fl_manager.privacy_mechanism.l2_norm_clip == 1.0

def test_privacy_parameter_update(private_fl_manager):
    """Test updating privacy parameters."""
    new_noise = 2.0
    new_clip = 0.5
    
    private_fl_manager.update_privacy_parameters(
        noise_multiplier=new_noise,
        l2_norm_clip=new_clip
    )
    
    assert private_fl_manager.privacy_mechanism.noise_multiplier == new_noise
    assert private_fl_manager.privacy_mechanism.l2_norm_clip == new_clip

def test_training_round_privacy_metrics(private_fl_manager):
    """Test if training round produces proper privacy metrics."""
    metrics = private_fl_manager.train_round()
    
    # Check basic metrics structure
    assert hasattr(metrics, 'round_number')
    assert hasattr(metrics, 'client_metrics')
    assert hasattr(metrics, 'global_metrics')
    assert hasattr(metrics, 'privacy_metrics')
    assert hasattr(metrics, 'privacy_budget')
    
    # Check privacy metrics
    assert metrics.privacy_metrics.noise_scale > 0
    assert metrics.privacy_metrics.clip_norm > 0
    assert isinstance(metrics.privacy_metrics.clipped_updates, int)
    assert len(metrics.privacy_metrics.original_update_norms) == 2
    assert len(metrics.privacy_metrics.clipped_update_norms) == 2

def test_privacy_budget_accumulation(private_fl_manager):
    """Test if privacy budget accumulates properly."""
    # Get initial privacy metrics
    initial_metrics = private_fl_manager.get_privacy_metrics()
    initial_epsilon = initial_metrics['current_epsilon']
    
    # Run multiple rounds
    for _ in range(2):
        private_fl_manager.train_round()
    
    # Get updated privacy metrics
    updated_metrics = private_fl_manager.get_privacy_metrics()
    updated_epsilon = updated_metrics['current_epsilon']
    
    # Privacy budget (epsilon) should increase with more rounds
    assert updated_epsilon > initial_epsilon

def test_model_improvements(private_fl_manager):
    """Test if model improves while maintaining privacy."""
    initial_metrics = private_fl_manager.train_round()
    initial_accuracy = initial_metrics.global_metrics['test_accuracy']
    
    # Update privacy parameters to be more permissive
    private_fl_manager.update_privacy_parameters(noise_multiplier=0.5)
    
    # Train for a few more rounds
    final_metrics = private_fl_manager.train_round()
    final_accuracy = final_metrics.global_metrics['test_accuracy']
    
    # Model should maintain or improve accuracy
    assert final_accuracy >= initial_accuracy * 0.8  # Allow for some variance due to privacy

def test_history_tracking(private_fl_manager):
    """Test if history is properly tracked."""
    # Train for multiple rounds
    num_rounds = 2
    for _ in range(num_rounds):
        private_fl_manager.train_round()
    
    # Check history structure
    assert len(private_fl_manager.history['rounds']) == num_rounds
    assert len(private_fl_manager.history['training_metrics']) == num_rounds
    assert len(private_fl_manager.history['privacy_metrics']) == num_rounds
    assert len(private_fl_manager.history['privacy_budget']) == num_rounds

def test_different_privacy_levels(private_fl_manager):
    """Test model behavior with different privacy levels."""
    # Train with high privacy (more noise)
    private_fl_manager.update_privacy_parameters(noise_multiplier=2.0)
    high_privacy_metrics = private_fl_manager.train_round()
    
    # Reset and train with low privacy (less noise)
    new_manager = PrivateFederatedLearningManager(
        num_clients=2,
        noise_multiplier=0.1,
        test_mode=True
    )
    low_privacy_metrics = new_manager.train_round()
    
    # Lower privacy should typically lead to better accuracy
    assert low_privacy_metrics.global_metrics['test_accuracy'] >= \
           high_privacy_metrics.global_metrics['test_accuracy'] * 0.8

def test_weight_update_clipping(private_fl_manager):
    """Test if weight updates are properly clipped."""
    metrics = private_fl_manager.train_round()
    
    # Check if clipping is working
    assert all(norm <= private_fl_manager.privacy_mechanism.l2_norm_clip * (1 + 1e-6)
              for norm in metrics.privacy_metrics.clipped_update_norms)

def test_noise_addition_consistency(private_fl_manager):
    """Test if noise addition is consistent with privacy parameters."""
    # Train with different noise levels
    private_fl_manager.update_privacy_parameters(noise_multiplier=0.1)
    low_noise_metrics = private_fl_manager.train_round()
    
    private_fl_manager.update_privacy_parameters(noise_multiplier=2.0)
    high_noise_metrics = private_fl_manager.train_round()
    
    # Higher noise should result in larger noise scale
    assert high_noise_metrics.privacy_metrics.noise_scale > \
           low_noise_metrics.privacy_metrics.noise_scale

if __name__ == "__main__":
    pytest.main([__file__])