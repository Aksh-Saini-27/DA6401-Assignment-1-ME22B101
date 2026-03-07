"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

class Dense:
    def __init__(self, in_features, out_features, weight_init="random"):
        self.in_features = in_features
        self.out_features = out_features
        self.W = None
        self.b = None
        self.grad_W = None
        self.grad_b = None
        
        if weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * limit
        elif weight_init == "zeros":
            self.W = np.zeros((in_features, out_features))
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
            
        self.b = np.zeros((1, out_features))
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
        
    def backward(self, grad_output):
        self.grad_W = np.dot(self.x.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return np.dot(grad_output, self.W.T)