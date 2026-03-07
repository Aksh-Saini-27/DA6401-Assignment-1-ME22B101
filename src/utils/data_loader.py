"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np

def load_data(dataset_name="mnist"):
    if dataset_name == "mnist":
        from .download_mnist import manual_mnist
        (X_train, y_train), (X_test, y_test) = manual_mnist()
    elif dataset_name == "fashion_mnist":
        import keras.datasets.fashion_mnist as fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return (X_train, y_train), (X_test, y_test)

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def create_mini_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        excerpt = indices[start_idx:start_idx + batch_size]
        yield X[excerpt], y[excerpt]
