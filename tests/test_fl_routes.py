# tests/test_fl_routes.py

from fastapi.testclient import TestClient
import pytest
from api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    """Reset FL manager before and after each test."""
    from api.routers.fl_routes import reset_fl_manager
    reset_fl_manager()
    yield
    reset_fl_manager()

@pytest.fixture
def initialized_fl():
    """Initialize FL before testing."""
    config = {
        "num_clients": 2,
        "local_epochs": 1,
        "batch_size": 32,
        "noise_multiplier": 1.0,
        "l2_norm_clip": 1.0,
        "test_mode": True
    }
    response = client.post("/api/fl/initialize", json=config)
    assert response.status_code == 200
    return response.json()

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_initialization():
    """Test FL initialization endpoint."""
    config = {
        "num_clients": 2,
        "local_epochs": 1,
        "batch_size": 32,
        "noise_multiplier": 1.0,
        "l2_norm_clip": 1.0
    }
    response = client.post("/api/fl/initialize", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "initial_metrics" in data

def test_train_round(initialized_fl):
    """Test training round endpoint."""
    response = client.post("/api/fl/train_round")
    assert response.status_code == 200
    data = response.json()
    assert "client_metrics" in data
    assert "global_metrics" in data
    assert "privacy_metrics" in data

def test_update_privacy(initialized_fl):
    """Test privacy parameter update endpoint."""
    config = {
        "noise_multiplier": 2.0,
        "l2_norm_clip": 0.5
    }
    response = client.post("/api/fl/update_privacy", json=config)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    metrics = data["current_metrics"]
    assert metrics["noise_multiplier"] == 2.0
    assert metrics["l2_norm_clip"] == 0.5

def test_get_current_state(initialized_fl):
    """Test current state endpoint."""
    response = client.get("/api/fl/current_state")
    assert response.status_code == 200
    data = response.json()
    assert "current_round" in data
    assert "privacy_settings" in data
    assert "training_active" in data

def test_client_info(initialized_fl):
    """Test client information endpoint."""
    # Train one round to have some data
    client.post("/api/fl/train_round")
    
    response = client.get("/api/fl/client_info/0")
    assert response.status_code == 200
    data = response.json()
    assert "client_id" in data
    assert "data_size" in data
    assert "training_progress" in data
    assert "privacy_impact" in data

def test_reset_training(initialized_fl):
    """Test training reset endpoint."""
    # Train one round
    client.post("/api/fl/train_round")
    
    # Reset training
    response = client.post("/api/fl/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Verify state is reset
    state = client.get("/api/fl/current_state").json()
    assert state["current_round"] == 0

def test_get_explanation():
    """Test explanation content endpoint."""
    response = client.get("/api/fl/explanations/federated_learning")
    assert response.status_code == 200
    data = response.json()
    assert "title" in data
    assert "content" in data
    assert "key_points" in data

def test_error_handling():
    """Test error handling for uninitialized FL."""
    from api.routers.fl_routes import fl_manager
    if fl_manager is not None:
        print(f"Warning: fl_manager was not None at start of test: {fl_manager}")

    response = client.get("/api/fl/metrics")

    print(f"Response status code: {response.status_code}")
    print(f"Response body: {response.json()}")
    
    assert response.status_code == 400, "Expected 400 status code for uninitialized FL"
    assert "not initialized" in response.json()["detail"].lower()

if __name__ == "__main__":
    pytest.main([__file__])