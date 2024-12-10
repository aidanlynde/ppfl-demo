# models/federated/fl_manager.py

import numpy as np
from typing import Dict, List, Tuple, Any
from .cnn_model import MNISTModel
from utils import MNISTDataHandler

class FederatedLearningManager:
    """Basic Federated Learning Manager without privacy mechanisms."""
    
    def __init__(
        self,
        num_clients: int = 5,
        local_epochs: int = 1,
        batch_size: int = 32,
        rounds: int = 10
    ):
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.rounds = rounds
        
        # Training history
        self.history = {
            'round_metrics': [],
            'global_metrics': []
        }
        
        # Initialize data and models
        self._initialize_setup()
    
    def _initialize_setup(self) -> None:
        """Initialize data handler and models."""
        # Set up data handler
        self.data_handler = MNISTDataHandler(num_clients=self.num_clients)
        self.data_handler.load_and_preprocess_data()
        self.data_handler.partition_data()
        
        # Get model parameters
        self.input_shape = self.data_handler.get_input_shape()
        self.num_classes = self.data_handler.get_num_classes()
        
        # Initialize global model
        self.global_model = MNISTModel(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Initialize client models
        self.client_models = {
            i: MNISTModel(
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
            for i in range(self.num_clients)
        }
    
    def train_round(self) -> Dict[str, Any]:
        """Execute one round of federated learning."""
        round_metrics = {}
        
        # Distribute global weights to clients
        self._distribute_weights()
        
        # Train each client
        client_metrics = {}
        for client_id in range(self.num_clients):
            metrics = self._train_client(client_id)
            client_metrics[client_id] = metrics
        
        # Aggregate weights
        self._aggregate_weights()
        
        # Evaluate global model
        global_metrics = self._evaluate_global_model()
        
        # Store metrics
        round_metrics = {
            'client_metrics': client_metrics,
            'global_metrics': global_metrics
        }
        self.history['round_metrics'].append(round_metrics)
        self.history['global_metrics'].append(global_metrics)
        
        return round_metrics
    
    def _distribute_weights(self) -> None:
        """Distribute global model weights to all clients."""
        global_weights = self.global_model.get_weights()
        for client_model in self.client_models.values():
            client_model.set_weights(global_weights)
    
    def _train_client(self, client_id: int) -> Dict[str, float]:
        """Train a single client for one round."""
        client_data = self.data_handler.get_client_data(client_id)
        client_model = self.client_models[client_id]
        
        x_train, y_train = client_data['x_train'], client_data['y_train']
        metrics_history = []
        
        # Train for specified number of local epochs
        for _ in range(self.local_epochs):
            for i in range(0, len(x_train), self.batch_size):
                batch_x = x_train[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]
                metrics = client_model.train_on_batch(batch_x, batch_y)
                metrics_history.append(metrics)
        
        # Calculate average metrics for this client
        avg_metrics = {
            'loss': np.mean([m[0] for m in metrics_history]),
            'accuracy': np.mean([m[1] for m in metrics_history])
        }
        
        return avg_metrics
    
    def _aggregate_weights(self) -> None:
        """Aggregate client weights using FedAvg algorithm."""
        # Get weights from all clients
        client_weights = [
            model.get_weights() 
            for model in self.client_models.values()
        ]
        
        # Calculate average weights
        avg_weights = []
        for weights_list_tuple in zip(*client_weights):
            avg_weights.append(
                np.array([np.array(w).astype(float) for w in weights_list_tuple])
                .mean(axis=0)
            )
        
        # Update global model
        self.global_model.set_weights(avg_weights)
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on test data."""
        x_test, y_test = self.data_handler.get_test_data()
        loss, accuracy = self.global_model.evaluate(x_test, y_test)
        
        return {
            'test_loss': loss,
            'test_accuracy': accuracy
        }
    
    def get_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.history
    
    def get_global_model(self) -> MNISTModel:
        """Get the global model."""
        return self.global_model