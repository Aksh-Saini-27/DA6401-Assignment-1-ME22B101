"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .neural_layer import Dense
from .activations import Sigmoid, Tanh, ReLU, Softmax
from .objective_functions import MSE, CrossEntropy

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args=None, input_size=784, num_classes=10):
        """
        Initialize the neural network.
        
        Args:
            cli_args: Command-line arguments for configuring the network
        """
        self.layers = []
        if cli_args is None:
            return
            
        hidden_sizes = cli_args.hidden_size
        act_name = cli_args.activation.lower()
        init_method = cli_args.weight_init.lower()
        
        # Build architecture
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(Dense(prev_size, size, weight_init=init_method))
            if act_name == "relu":
                self.layers.append(ReLU())
            elif act_name == "sigmoid":
                self.layers.append(Sigmoid())
            elif act_name == "tanh":
                self.layers.append(Tanh())
            else:
                raise ValueError(f"Unknown activation: {act_name}")
            prev_size = size
            
        # Output layer
        self.layers.append(Dense(prev_size, num_classes, weight_init=init_method))
        self.layers.append(Softmax())
        
        if cli_args.loss == "mse":
            self.loss_fn = MSE()
        elif cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()
        else:
            raise ValueError(f"Unknown loss: {cli_args.loss}")

        self.optimizer = None
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits/probabilities
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            grad_w, grad_b
        """
        # The loss function's backward handles combining the gradient with Softmax
        grad = self.loss_fn.backward()
        
        # Go backwards through layers EXCEPT the Softmax,
        # since we combined the softmax and cross-entropy gradient
        # Actually, in our implementation, Softmax passes the gradient through:
        # returns grad_output directly. It works for both MSE and CE if we pass `(y_pred - y_true)/N`
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        if self.optimizer:
            self.optimizer.update(self.layers)
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        pass # Implementation in train loop
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        pass
    
    def predict(self, X):
        return self.forward(X)
        
    def save_weights(self, path):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                weights[f"W_{i}"] = layer.W
                weights[f"b_{i}"] = layer.b
        np.save(path, weights)
        
    def load_weights(self, path):
        weights = np.load(path, allow_pickle=True).item()
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                layer.W = weights[f"W_{i}"]
                layer.b = weights[f"b_{i}"]
