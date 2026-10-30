import numpy as np


def generate_data(n=50, w_true=2.5, b_true=1.0, noise=1.5, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    y = w_true * x + b_true + rng.normal(0.0, noise, n)
    return x, y


def mse(x, y, w, b):
    return np.mean((w * x + b - y) ** 2)


def normal_equation(x, y):
    X = np.column_stack((x, np.ones_like(x)))
    w, b = np.linalg.solve(X.T @ X, X.T @ y)
    return w, b


def gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    w, b = 0.0, 0.0
    history = []
    for _ in range(epochs):
        error = w * x + b - y
        grad_w = 2.0 * np.mean(error * x)
        grad_b = 2.0 * np.mean(error)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        history.append(mse(x, y, w, b))
    return w, b, history


if __name__ == "__main__":
    x, y = generate_data()
    w, b = normal_equation(x, y)
    print(f"normal equation: w={w:.4f} b={b:.4f} mse={mse(x, y, w, b):.4f}")
    for lr in (0.001, 0.005, 0.01):
        w, b, history = gradient_descent(x, y, learning_rate=lr)
        print(f"lr={lr}: w={w:.4f} b={b:.4f} mse={history[-1]:.4f}")
