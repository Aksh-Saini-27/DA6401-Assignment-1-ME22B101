"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.square(y_pred - y_true))
        
    def backward(self):
        N = self.y_pred.shape[0]
        return 2.0 * (self.y_pred - self.y_true) / N / self.y_pred.shape[1]

class CrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        # Adding epsilon to avoid log(0)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Assuming y_true is one-hot encoded
        loss = -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
        return loss
        
    def backward(self):
        # Combined gradient for Softmax + Cross Entropy: (y_pred - y_true) / N
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / N