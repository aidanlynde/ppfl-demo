# tests/test_privacy_mechanisms.py

import pytest
import numpy as np
from models.privacy.mechanisms import PrivacyMechanism

@pytest.fixture
def privacy_mechanism():
    """Fixture to provide privacy mechanism for testing."""
    return PrivacyMechanism(noise_multiplier=1.0, l2_norm_clip=1.0)

@pytest.fixture
def mock_updates():
    """Fixture to provide mock model updates for testing."""
    # Create mock updates with known L2 norms
    update1 = [np.array([[2.0, 0.0], [0.0, 0.0]])]  # L2 norm = 2
    update2 = [np.array([[0.1, 0.1], [0.1, 0.1]])]  # L2 norm ≈ 0.2
    return [update1, update2]

def test_clipping(privacy_mechanism, mock_updates):
    """Test gradient clipping mechanism."""
    clipped_updates, metrics = privacy_mechanism.apply_privacy(
        mock_updates,
        batch_size=32
    )
    
    # Check if updates were properly clipped
    assert metrics.clipped_updates == 1  # Only the first update should be clipped
    assert len(metrics.original_update_norms) == 2
    assert len(metrics.clipped_update_norms) == 2
    
    # Check original norms
    np.testing.assert_almost_equal(metrics.original_update_norms[0], 2.0, decimal=5)
    np.testing.assert_almost_equal(metrics.original_update_norms[1], 0.2, decimal=5)
    
    # Check clipped norms
    assert all(norm <= privacy_mechanism.l2_norm_clip * (1 + 1e-6) 
              for norm in metrics.clipped_update_norms)

def test_noise_addition(privacy_mechanism):
    """Test noise addition mechanism."""
    # Create identical updates
    identical_updates = [
        [np.ones((2, 2))] for _ in range(3)
    ]
    
    # Apply privacy multiple times
    results = [
        privacy_mechanism.apply_privacy(identical_updates, batch_size=32)[0]
        for _ in range(5)
    ]
    
    # Check if noise was actually added (updates should differ)
    assert not all(
        np.array_equal(results[0][0], result[0])
        for result in results[1:]
    )

def test_privacy_budget(privacy_mechanism):
    """Test privacy budget calculation."""
    privacy_spent = privacy_mechanism.get_privacy_spent(
        num_steps=1000,
        batch_size=32,
        dataset_size=1000
    )
    
    assert 'epsilon' in privacy_spent
    assert 'delta' in privacy_spent
    assert privacy_spent['epsilon'] > 0
    assert privacy_spent['delta'] > 0

def test_parameter_update(privacy_mechanism):
    """Test privacy parameter updates."""
    new_noise = 2.0
    new_clip = 0.5
    
    privacy_mechanism.update_parameters(new_noise, new_clip)
    
    assert privacy_mechanism.noise_multiplier == new_noise
    assert privacy_mechanism.l2_norm_clip == new_clip

def test_different_noise_levels(privacy_mechanism, mock_updates):
    """Test effect of different noise levels."""
    # Test with low noise
    privacy_mechanism.update_parameters(noise_multiplier=0.1, l2_norm_clip=1.0)
    low_noise_result, _ = privacy_mechanism.apply_privacy(mock_updates, batch_size=32)
    
    # Test with high noise
    privacy_mechanism.update_parameters(noise_multiplier=2.0, l2_norm_clip=1.0)
    high_noise_result, _ = privacy_mechanism.apply_privacy(mock_updates, batch_size=32)
    
    # High noise should lead to more deviation from original updates
    low_noise_var = np.var(low_noise_result[0])
    high_noise_var = np.var(high_noise_result[0])
    assert high_noise_var > low_noise_var

if __name__ == "__main__":
    pytest.main([__file__])