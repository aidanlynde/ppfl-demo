# models/cnn_model.py

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from typing import Tuple, List

class MNISTModel:
    """CNN model for MNIST classification in federated learning setting."""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> Model:
        """
        Build and compile the CNN model.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                  input_shape=self.input_shape),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_weights(self) -> List[tf.Tensor]:
        """
        Get model weights.
        
        Returns:
            List of model weight tensors
        """
        return self.model.get_weights()
    
    def set_weights(self, weights: List[tf.Tensor]) -> None:
        """
        Set model weights.
        
        Args:
            weights: List of weight tensors to set
        """
        self.model.set_weights(weights)
    
    def train_on_batch(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[float, float]:
        """
        Train the model on a batch of data.
        
        Args:
            x: Input data batch
            y: Target labels batch
            
        Returns:
            Tuple of (loss, accuracy) for this batch
        """
        return self.model.train_on_batch(x, y)
    
    def evaluate(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            x: Test input data
            y: Test labels
            
        Returns:
            Tuple of (loss, accuracy) on test data
        """
        return self.model.evaluate(x, y, verbose=0)
    
    def predict(self, x: tf.Tensor) -> tf.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input data
            
        Returns:
            Model predictions
        """
        return self.model.predict(x, verbose=0)