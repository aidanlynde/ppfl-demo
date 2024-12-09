# tests/test_data_handler.py

import pytest
import numpy as np
from utils.data_handler import MNISTDataHandler

def test_data_loading():
    """Test basic data loading and preprocessing."""
    handler = MNISTDataHandler(num_clients=5)
    handler.load_and_preprocess_data()
    
    # Check data shapes
    assert handler.x_train.shape[1:] == (28, 28, 1)
    assert handler.y_train.shape[1] == 10
    
    # Check normalization
    assert np.all(handler.x_train >= 0.0)
    assert np.all(handler.x_train <= 1.0)
    
def test_data_partitioning():
    """Test IID data partitioning."""
    num_clients = 5
    handler = MNISTDataHandler(num_clients=num_clients)
    handler.load_and_preprocess_data()
    handler.partition_data(strategy='iid')
    
    # Check number of clients
    assert len(handler.client_data) == num_clients
    
    # Check that all clients have roughly equal amount of data
    client_data_sizes = [len(handler.client_data[i]['x_train']) for i in range(num_clients)]
    assert len(set(client_data_sizes)) <= 1  # All clients should have same size (or differ by 1)
    
    # Check that client data maintains proper shapes
    client_data = handler.get_client_data(0)
    assert client_data['x_train'].shape[1:] == (28, 28, 1)
    assert client_data['y_train'].shape[1] == 10

def test_validation_split():
    """Test validation data splitting."""
    handler = MNISTDataHandler(validation_split=0.1)
    handler.load_and_preprocess_data()
    
    x_val, y_val = handler.get_validation_data()
    assert len(x_val) > 0
    assert len(x_val) == len(y_val)
    
def test_invalid_client_id():
    """Test error handling for invalid client ID."""
    handler = MNISTDataHandler(num_clients=5)
    handler.load_and_preprocess_data()
    handler.partition_data()
    
    with pytest.raises(ValueError):
        handler.get_client_data(999)  # Invalid client ID

if __name__ == "__main__":
    pytest.main([__file__])