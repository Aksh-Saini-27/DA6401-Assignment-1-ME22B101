import urllib.request
import gzip
import os
import numpy as np

def download_and_extract(url, filename, extract_shape):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16 if len(extract_shape) == 3 else 8)
    return data.reshape(extract_shape)

def manual_mnist():
    os.makedirs('data', exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    
    X_train = download_and_extract(base_url + "train-images-idx3-ubyte.gz", "data/train_images.gz", (-1, 28, 28))
    y_train = download_and_extract(base_url + "train-labels-idx1-ubyte.gz", "data/train_labels.gz", (-1,))
    X_test = download_and_extract(base_url + "t10k-images-idx3-ubyte.gz", "data/test_images.gz", (-1, 28, 28))
    y_test = download_and_extract(base_url + "t10k-labels-idx1-ubyte.gz", "data/test_labels.gz", (-1,))
    
    return (X_train, y_train), (X_test, y_test)
