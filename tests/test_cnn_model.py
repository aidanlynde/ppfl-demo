# tests/test_cnn_model.py

import pytest
import numpy as np
from models.federated.cnn_model import MNISTModel
from utils import MNISTDataHandler

@pytest.fixture
def data_handler():
    """Fixture to provide data for testing."""
    handler = MNISTDataHandler(num_clients=1)
    handler.load_and_preprocess_data()
    return handler

@pytest.fixture
def model(data_handler):
    """Fixture to provide model for testing."""
    return MNISTModel(
        input_shape=data_handler.get_input_shape(),
        num_classes=data_handler.get_num_classes()
    )

def test_model_creation(model):
    """Test if model is created with correct structure."""
    assert model.model is not None
    # Check if model has correct input shape
    assert model.model.input_shape[1:] == (28, 28, 1)
    # Check if model has correct output shape
    assert model.model.output_shape[1] == 10

def test_weight_operations(model):
    """Test getting and setting weights."""
    # Get initial weights
    initial_weights = model.get_weights()
    
    # Create dummy weights (same shape as initial weights)
    dummy_weights = [np.random.randn(*w.shape).astype(np.float32) for w in initial_weights]
    
    # Set weights
    model.set_weights(dummy_weights)
    
    # Get new weights
    new_weights = model.get_weights()
    
    # Check if weights were updated (using allclose for float comparison)
    for w1, w2 in zip(dummy_weights, new_weights):
        assert np.allclose(w1, w2, rtol=1e-5, atol=1e-8)

def test_training(model, data_handler):
    """Test if model can train on a batch of data."""
    # Get a small batch of data
    x_train = data_handler.x_train[:32]
    y_train = data_handler.y_train[:32]
    
    # Train on batch
    loss, accuracy = model.train_on_batch(x_train, y_train)
    
    # Check if loss and accuracy are valid
    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1

def test_prediction(model, data_handler):
    """Test if model can make predictions."""
    # Get a small batch of data
    x_test = data_handler.x_test[:10]
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Check predictions shape and values
    assert predictions.shape == (10, 10)  # 10 samples, 10 classes
    assert np.all(predictions >= 0)  # Probabilities should be non-negative
    assert np.all(predictions <= 1)  # Probabilities should be <= 1
    assert np.allclose(np.sum(predictions, axis=1), 1)  # Each row should sum to 1

if __name__ == "__main__":
    pytest.main([__file__])