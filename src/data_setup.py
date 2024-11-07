"""
Contains functionality for creating MNIST dataloaders for both finite network and NTK training
"""
import jax
import jax.numpy as jnp
from tensorflow.keras.datasets import mnist
import numpy as np
from typing import Tuple, Iterator, Callable
from functools import partial

def normalize_data(x: np.ndarray) -> np.ndarray:
    """Normalize data to [0,1] range"""
    return x.astype(np.float32) / 255.0

def prepare_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess MNIST dataset"""
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and reshape images
    x_train = normalize_data(x_train)
    x_test = normalize_data(x_test)
    
    # Reshape to (batch, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to one-hot
    y_train = jax.nn.one_hot(y_train, num_classes=10)
    y_test = jax.nn.one_hot(y_test, num_classes=10)
    
    return x_train, y_train, x_test, y_test

def create_batch_iterator(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> Iterator:
    """Creates a batch iterator for the dataset."""
    num_samples = len(images)
    indices = np.arange(num_samples)
    
    def iterator():
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                yield images[batch_idx], labels[batch_idx]
    
    return iterator()

def create_data_iterators(
    batch_size: int = 128,
    shuffle_train: bool = True
) -> Tuple[Iterator, Iterator, int]:
    """Creates training and testing data iterators.
    
    Args:
        batch_size: Number of samples per batch
        shuffle_train: Whether to shuffle training data
        
    Returns:
        train_iterator: Iterator for training data
        test_iterator: Iterator for test data
        num_classes: Number of classes in the dataset
    """
    # Load and preprocess MNIST
    x_train, y_train, x_test, y_test = prepare_mnist()
    
    # Create iterators
    train_iterator = create_batch_iterator(
        x_train, y_train, batch_size, shuffle=shuffle_train
    )
    
    test_iterator = create_batch_iterator(
        x_test, y_test, batch_size, shuffle=False
    )
    
    return train_iterator, test_iterator, 10  # 10 classes in MNIST 