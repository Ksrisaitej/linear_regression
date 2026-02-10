import numpy as np
import math


def compute_cost(x, y, b, w):
    m = len(x)  # Fixed: number of samples
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost /= (2 * m)
    return cost


def compute_gradient(x, y, b, w):
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.
    for i in range(m):
        err = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_descent(x, y, learning_rate, iterations):
    m, n = x.shape
    w = np.zeros(n)
    b = 0.

    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, b, w)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        if i % math.ceil(iterations / 10) == 0:  # Fixed: indented inside loop
            print(f"Iteration {i}: cost = {compute_cost(x, y, b, w):.6f}")

    return w, b


def predict(x_test, w, b):
    return np.dot(x_test, w) + b


y = np.array([11,22,33,44,55,66,77])
x = np.array([1,2,3,4,5,6,7]).reshape(-1, 1)
x_test = np.array([9,10,11,12,13,14,15]).reshape(-1, 1)
w, b = gradient_descent(x, y, 0.05, 1000)

prediction = predict(x_test, w, b)
print(prediction)