"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class Sigmoid:
    def forward(self, x):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output
        
    def backward(self, grad_output):
        return grad_output * self.output * (1.0 - self.output)

class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
        
    def backward(self, grad_output):
        return grad_output * (1.0 - np.square(self.output))

class ReLU:
    def forward(self, x):
        self.x = x
        self.output = np.maximum(0, x)
        return self.output
        
    def backward(self, grad_output):
        return grad_output * (self.x > 0)

class Softmax:
    def forward(self, x):
        # Numerically stable softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
        
    def backward(self, grad_output):
        # Assume cross-entropy loss gradient combined (dZ = output - true_labels)
        # So we just pass the gradient through.
        return grad_output