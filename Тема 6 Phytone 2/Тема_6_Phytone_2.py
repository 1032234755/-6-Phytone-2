
import numpy as np

def f2(x):
    return 7*x[0]**2 + 2*x[0]*x[1] + 5*x[1]**2 + x[0] - 10*x[1]

def gradient_f2(x):
    df_dx1 = 14*x[0] + 2*x[1] + 1
    df_dx2 = 2*x[0] + 10*x[1] - 10
    return np.array([df_dx1, df_dx2])

def steepest_descent(f, grad_f, x0, lr=0.01, max_iter=100):
    x = np.array(x0)
    for _ in range(max_iter):
        grad = grad_f(x)
        x -= lr * grad
    return x

x0 = [0.5, 0.5]
solution = steepest_descent(f2, gradient_f2, x0)
print("Решение методом наискорейшего спуска:", solution)
