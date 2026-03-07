"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from .neural_layer import Dense
from .activations import Sigmoid, Tanh, ReLU
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

        prev_size = input_size

        # =========================
        # Hidden layers
        # =========================

        for size in hidden_sizes:

            self.layers.append(
                Dense(prev_size, size, weight_init=init_method)
            )

            if act_name == "relu":
                self.layers.append(ReLU())

            elif act_name == "sigmoid":
                self.layers.append(Sigmoid())

            elif act_name == "tanh":
                self.layers.append(Tanh())

            else:
                raise ValueError(f"Unknown activation: {act_name}")

            prev_size = size

        # =========================
        # Output layer (returns logits)
        # =========================

        self.layers.append(
            Dense(prev_size, num_classes, weight_init=init_method)
        )

        # =========================
        # Loss function
        # =========================

        if cli_args.loss == "mse":
            self.loss_fn = MSE()

        elif cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropy()

        else:
            raise ValueError(f"Unknown loss: {cli_args.loss}")

        self.optimizer = None

    # =========================
    # Forward
    # =========================

    def forward(self, X):
        """
        Forward propagation through the network.
        Returns logits (NOT softmax probabilities).
        """

        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out

    # =========================
    # Backward
    # =========================

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        """

        grad = self.loss_fn.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # =========================
    # Weight update
    # =========================

    def update_weights(self):

        if self.optimizer:
            self.optimizer.update(self.layers)

    # =========================
    # Prediction
    # =========================

    def predict(self, X):
        """
        Return logits for given input.
        """

        return self.forward(X)

    # =========================
    # Save weights
    # =========================

    def save_weights(self, path):

        weights = {}

        for i, layer in enumerate(self.layers):

            if hasattr(layer, "W"):
                weights[f"W_{i}"] = layer.W
                weights[f"b_{i}"] = layer.b

        np.save(path, weights)

    # =========================
    # Load weights
    # =========================

    def load_weights(self, path):

        weights = np.load(path, allow_pickle=True).item()

        for i, layer in enumerate(self.layers):

            if hasattr(layer, "W"):
                layer.W = weights[f"W_{i}"]
                layer.b = weights[f"b_{i}"]

    # =========================
    # Autograder required functions
    # =========================

    # def get_weights(self):
    #     """
    #     Return weights dictionary.
    #     Used by autograder and train script.
    #     """

    #     weights = {}

    #     for i, layer in enumerate(self.layers):

    #         if hasattr(layer, "W"):
    #             weights[f"W_{i}"] = layer.W
    #             weights[f"b_{i}"] = layer.b

    #     return weights

    # def set_weights(self, weights):
    #     """
    #     Set weights dictionary.
    #     Used by autograder forward pass check.
    #     """

    #     for i, layer in enumerate(self.layers):

    #         if hasattr(layer, "W"):
    #             layer.W = weights[f"W_{i}"]
    #             layer.b = weights[f"b_{i}"]

    def set_weights(self, weights):
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):

                W_key = f"W{i}"
                b_key = f"b{i}"
                if W_key not in weights or b_key not in weights:
                    raise ValueError("Missing weight keys")
                layer.W = weights[W_key]
                layer.b = weights[b_key]
            return
        raise ValueError(f"Unsupported weight format: {type(weights)}")
    
    def get_weights(self):
        """
        Return all layer weights and biases in a serializable format.
        """
        weights = []
        for layer in self.layers:
            weights.append({
                "W": layer.W,
                "b": layer.b
            })
        return weights