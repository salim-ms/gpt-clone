import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

"""
Loss fn = 1/N SUM (y - y`)**2 
where y` = ax + b
"""
def compute_y(x, a, b):
    return a * x + b

def compute_gradients(x, y, a, b):
    N = float(len(x))
    da = (-2 / N) * np.sum(x * (y - (a*x + b)))
    db = (-1 / N) * np.sum(y - (a*x + b))
    
    return da, db

def gradient_descent(x, y, starting_point, lr= 0.01, iterations=10000):
    a,b = starting_point
    
    for _ in range(iterations):
        da, db = compute_gradients(x, y, a, b)
        print(da, db)
        a = a - (lr * da)
        b = b - (lr * db)
    
    return a, b

a, b = gradient_descent(x, y, (0, 0))

print(a, b)
print(compute_y(x, a, b))


