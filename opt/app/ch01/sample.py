import numpy as np

I = 2
H = 3
W = np.random.randn(I, H)

print(W)

b = [1, 2]
X = [W, b]

print(X)

W, b = X

print(W)
print(b)

