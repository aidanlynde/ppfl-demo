# utils/data_handler.py

import numpy as np
from typing import Dict, Tuple
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class MNISTDataHandler:
    """Handles MNIST dataset loading, preprocessing, and partitioning for federated learning."""
    
    def __init__(self, num_clients: int = 5, validation_split: float = 0.1, test_mode: bool = False):
        self.num_clients = num_clients
        self.validation_split = validation_split
        self.test_mode = test_mode
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.client_data = {}
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        
    def load_and_preprocess_data(self) -> None:
        try:
            # Load MNIST data
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            
            # Always use minimal dataset in production to save memory
            x_train = x_train[:200]  
            y_train = y_train[:200]
            x_test = x_test[:50]   
            y_test = y_test[:50]
            
            # If in test mode, use even smaller subset
            if self.test_mode:
                x_train = x_train[:100]
                y_train = y_train[:100]
                x_test = x_test[:20]
                y_test = y_test[:20]
            
            # Normalize and reshape data
            self.x_train = self._preprocess_features(x_train)
            self.x_test = self._preprocess_features(x_test)
            
            # Convert labels to categorical
            self.y_train = to_categorical(y_train, self.num_classes)
            self.y_test = to_categorical(y_test, self.num_classes)
            
            # Split training data into train and validation sets
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train, self.y_train,
                test_size=self.validation_split,
                random_state=42
            )
            
            # Force garbage collection after processing
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def _preprocess_features(self, data: np.ndarray) -> np.ndarray:
        """Preprocess feature data."""
        # Normalize pixel values
        data = data.astype('float32') / 255.0
        
        # Add channel dimension for CNN
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)
            
        return data
        
    def partition_data(self, strategy: str = 'iid') -> None:
        """Partition data among clients using specified strategy."""
        if strategy == 'iid':
            self._partition_iid()
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")
            
    def _partition_iid(self) -> None:
        """Partition data in IID fashion among clients."""
        # Get number of samples per client
        samples_per_client = len(self.x_train) // self.num_clients
        
        # Randomly shuffle data
        indices = np.random.permutation(len(self.x_train))
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            
            # Get client indices
            client_indices = indices[start_idx:end_idx]
            
            # Assign data to client
            self.client_data[i] = {
                'x_train': self.x_train[client_indices],
                'y_train': self.y_train[client_indices]
            }
    
    def get_client_data(self, client_id: int) -> Dict[str, np.ndarray]:
        """Get training data for a specific client."""
        if client_id not in self.client_data:
            raise ValueError(f"Client ID {client_id} not found")
        return self.client_data[client_id]
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test dataset."""
        return self.x_test, self.y_test
    
    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation dataset."""
        return self.x_val, self.y_val
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get input shape of the data."""
        return self.input_shape
    
    def get_num_classes(self) -> int:
        """Get number of classes in the dataset."""
        return self.num_classes