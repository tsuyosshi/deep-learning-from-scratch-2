import numpy as np

class MatMul:
    def __init__(self, W):
        self.params = W
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, _ = self.params
        y = np.dot(x, W)
        self.x = x
        return y
    
    def backward(self, dy):
        W, _ = self.params
        dx = np.dot(dy, W.T)
        dW = np.dot(self.x.T, dy)
        self.grads[0][...] = dW
        return dx
    
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    def backward(self, dy):
        dx = dy * (1.0 - self.y) * self.y
        return dx
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        y = np.dot(x, W) + b
        self.x = x
        return y
    
    def backward(self, dy):
        W, b = self.params
        dx = np.dot(dy, W.T)
        dW = np.dot(self.x.T, dy)
        db = np.sum(dy, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

