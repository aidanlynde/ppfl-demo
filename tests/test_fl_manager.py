# tests/test_fl_manager.py

import pytest
import numpy as np
from models.federated.fl_manager import FederatedLearningManager

@pytest.fixture
def fl_manager():
    """Fixture to provide FL manager for testing."""
    manager = FederatedLearningManager(
        num_clients=2,  # Using 2 clients consistently
        local_epochs=1,
        batch_size=32,
        rounds=1,
        test_mode=True
    )
    return manager

def test_initialization(fl_manager):
    """Test FL manager initialization."""
    assert fl_manager.num_clients == 2  # Changed from 3 to 2
    assert len(fl_manager.client_models) == 2  # Changed from 3 to 2
    assert fl_manager.global_model is not None

def test_weight_distribution(fl_manager):
    """Test weight distribution to clients."""
    # Get global weights
    global_weights = fl_manager.global_model.get_weights()
    
    # Distribute weights
    fl_manager._distribute_weights()
    
    # Check if all clients have the same weights as global model
    for client_model in fl_manager.client_models.values():
        client_weights = client_model.get_weights()
        for g_w, c_w in zip(global_weights, client_weights):
            assert np.allclose(g_w, c_w, rtol=1e-5)

def test_client_training(fl_manager):
    """Test individual client training."""
    client_id = 0
    metrics = fl_manager._train_client(client_id)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['loss'], float)
    assert isinstance(metrics['accuracy'], float)
    assert 0 <= metrics['accuracy'] <= 1

def test_weight_aggregation(fl_manager):
    """Test weight aggregation."""
    # Store initial weights
    initial_weights = [np.array(w) for w in fl_manager.global_model.get_weights()]
    
    # Run one round of training
    fl_manager.train_round()
    
    # Get new weights
    new_weights = fl_manager.global_model.get_weights()
    
    # Verify weights have changed
    for init_w, new_w in zip(initial_weights, new_weights):
        assert not np.allclose(init_w, new_w, rtol=1e-5)

def test_complete_round(fl_manager):
    """Test a complete training round."""
    metrics = fl_manager.train_round()
    
    assert 'client_metrics' in metrics
    assert 'global_metrics' in metrics
    assert len(metrics['client_metrics']) == 2  # Changed from 3 to 2
    assert 'test_accuracy' in metrics['global_metrics']
    assert 'test_loss' in metrics['global_metrics']

if __name__ == "__main__":
    pytest.main([__file__])