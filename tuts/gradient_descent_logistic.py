import numpy as np


# Example usage with binary y values:
x = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 1, 1, 1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cross_entropy_loss(y, p):
    loss = -np.mean( (y * np.log(p)) + (1-y) * (1 - np.log(p)))
    return loss

def compute_y(x,a,b):
    z = a * x + b
    p = sigmoid(z)
    
    return p

def compute_gradients(x,y,a,b):
    z = a * x + b
    p = sigmoid(z)
    
    da = np.mean((p - y) * x)
    db = np.mean(p - y)
    return da, db

def gradient_descent(x, y, starting_point, lr=0.01, iterations=50000):
    a,b = starting_point
    
    for _ in range(iterations):
        da,db = compute_gradients(x,y,a,b)
        a = a - (lr * da)
        b = b - (lr * db)
    
    return a, b

a,b = gradient_descent(x,y, (0, 0))

print(a, b)
print(compute_y(x, a, b))