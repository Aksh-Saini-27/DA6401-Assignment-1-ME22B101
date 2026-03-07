"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

import numpy as np

class Optimizer:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def apply_weight_decay(self, layer):
        if self.weight_decay > 0:
            layer.grad_W += self.weight_decay * layer.W

class SGD(Optimizer):
    def update(self, layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                layer.W -= self.lr * layer.grad_W
                layer.b -= self.lr * layer.grad_b

class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.momentum = momentum
        self.v_W = {}
        self.v_b = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                if idx not in self.v_W:
                    self.v_W[idx] = np.zeros_like(layer.W)
                    self.v_b[idx] = np.zeros_like(layer.b)
                self.v_W[idx] = self.momentum * self.v_W[idx] - self.lr * layer.grad_W
                self.v_b[idx] = self.momentum * self.v_b[idx] - self.lr * layer.grad_b
                layer.W += self.v_W[idx]
                layer.b += self.v_b[idx]

class NAG(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.momentum = momentum
        self.v_W = {}
        self.v_b = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                if idx not in self.v_W:
                    self.v_W[idx] = np.zeros_like(layer.W)
                    self.v_b[idx] = np.zeros_like(layer.b)
                
                v_W_prev = self.v_W[idx]
                v_b_prev = self.v_b[idx]
                
                self.v_W[idx] = self.momentum * self.v_W[idx] - self.lr * layer.grad_W
                self.v_b[idx] = self.momentum * self.v_b[idx] - self.lr * layer.grad_b
                
                layer.W += -self.momentum * v_W_prev + (1 + self.momentum) * self.v_W[idx]
                layer.b += -self.momentum * v_b_prev + (1 + self.momentum) * self.v_b[idx]

class RMSProp(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = {}
        self.s_b = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                if idx not in self.s_W:
                    self.s_W[idx] = np.zeros_like(layer.W)
                    self.s_b[idx] = np.zeros_like(layer.b)
                self.s_W[idx] = self.beta * self.s_W[idx] + (1 - self.beta) * np.square(layer.grad_W)
                self.s_b[idx] = self.beta * self.s_b[idx] + (1 - self.beta) * np.square(layer.grad_b)
                layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[idx]) + self.epsilon)
                layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[idx]) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                if idx not in self.m_W:
                    self.m_W[idx] = np.zeros_like(layer.W)
                    self.v_W[idx] = np.zeros_like(layer.W)
                    self.m_b[idx] = np.zeros_like(layer.b)
                    self.v_b[idx] = np.zeros_like(layer.b)
                
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1 - self.beta1) * layer.grad_W
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1 - self.beta2) * np.square(layer.grad_W)
                
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1 - self.beta1) * layer.grad_b
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1 - self.beta2) * np.square(layer.grad_b)
                
                m_W_hat = self.m_W[idx] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[idx] / (1 - self.beta2 ** self.t)
                
                m_b_hat = self.m_b[idx] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[idx] / (1 - self.beta2 ** self.t)
                
                layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.t = 0

    def update(self, layers):
        self.t += 1
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                self.apply_weight_decay(layer)
                if idx not in self.m_W:
                    self.m_W[idx] = np.zeros_like(layer.W)
                    self.v_W[idx] = np.zeros_like(layer.W)
                    self.m_b[idx] = np.zeros_like(layer.b)
                    self.v_b[idx] = np.zeros_like(layer.b)
                
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1 - self.beta1) * layer.grad_W
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1 - self.beta2) * np.square(layer.grad_W)
                
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1 - self.beta1) * layer.grad_b
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1 - self.beta2) * np.square(layer.grad_b)
                
                m_W_hat = self.m_W[idx] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[idx] / (1 - self.beta2 ** self.t)
                
                m_b_hat = self.m_b[idx] / (1 - self.beta1 ** self.t)
                v_b_hat = self.v_b[idx] / (1 - self.beta2 ** self.t)
                
                m_W_prime = self.beta1 * m_W_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
                m_b_prime = self.beta1 * m_b_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)
                
                layer.W -= self.lr * m_W_prime / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.lr * m_b_prime / (np.sqrt(v_b_hat) + self.epsilon)