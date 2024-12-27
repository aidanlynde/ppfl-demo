# models/federated/fl_manager.py

import numpy as np
from typing import Dict, List, Tuple, Any
from .cnn_model import MNISTModel
from utils import MNISTDataHandler

class FederatedLearningManager:
    """Basic Federated Learning Manager without privacy mechanisms."""
    
    def __init__(
        self,
        num_clients: int = 3,
        local_epochs: int = 1,
        batch_size: int = 16,
        rounds: int = 10,
        test_mode: bool = False,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0
    ):
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.rounds = rounds
        self.test_mode = test_mode
        
        # Training history
        self.history = {
            'round_metrics': [],
            'global_metrics': []
        }
        
        # Initialize data and models
        self._initialize_setup()
    
    def _initialize_setup(self) -> None:
        """Initialize data handler and models."""
        self.data_handler = MNISTDataHandler(
            num_clients=self.num_clients,
            test_mode=self.test_mode
        )
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
        try:
            print("FL Manager: Starting training round")
            client_metrics = {}
            
            # Train each client
            for client_id in range(self.num_clients):
                print(f"Training client {client_id}")
                try:
                    metrics = self._train_client(client_id)
                    client_metrics[client_id] = metrics
                    print(f"Client {client_id} training complete: {metrics}")
                except Exception as e:
                    print(f"Error training client {client_id}: {str(e)}")
                    raise
            
            # Aggregate results
            print("Aggregating results")
            try:
                self._aggregate_weights()
                print("Weight aggregation complete")
            except Exception as e:
                print(f"Error during weight aggregation: {str(e)}")
                raise
            
            # Evaluate global model
            print("Evaluating global model")
            try:
                global_metrics = self._evaluate_global_model()
                print(f"Evaluation complete: {global_metrics}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                raise
            
            return {
                "client_metrics": client_metrics,
                "global_metrics": global_metrics
            }
            
        except Exception as e:
            print(f"Error in train_round: {str(e)}")
            raise
    
    def _distribute_weights(self) -> None:
        """Distribute global model weights to all clients."""
        global_weights = self.global_model.get_weights()
        for client_model in self.client_models.values():
            client_model.set_weights(global_weights)
    
    def _train_client(self, client_id: int) -> Dict[str, float]:
        """Train a single client for one round."""
        try:
            client_data = self.data_handler.get_client_data(client_id)
            x_train, y_train = client_data['x_train'], client_data['y_train']
            
            # Use smaller chunks for training
            chunk_size = 50  # Process only 50 samples at a time
            total_loss = 0
            total_accuracy = 0
            num_chunks = 0
            
            for i in range(0, len(x_train), chunk_size):
                chunk_x = x_train[i:i + chunk_size]
                chunk_y = y_train[i:i + chunk_size]
                
                # Train on smaller batches within the chunk
                for j in range(0, len(chunk_x), self.batch_size):
                    batch_x = chunk_x[j:j + self.batch_size]
                    batch_y = chunk_y[j:j + self.batch_size]
                    loss, accuracy = self.client_models[client_id].train_on_batch(batch_x, batch_y)
                    total_loss += loss
                    total_accuracy += accuracy
                    num_chunks += 1
                    
                # Force garbage collection after each chunk
                import gc
                gc.collect()
                
            return {
                'loss': total_loss / max(num_chunks, 1),
                'accuracy': total_accuracy / max(num_chunks, 1)
            }
            
        except Exception as e:
            print(f"Error training client {client_id}: {str(e)}")
            raise
    
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