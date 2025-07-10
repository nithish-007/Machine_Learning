import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        # Numerically stable sigmoid
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Better to initialize with zeros
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)

            # Gradients
            delw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            delb = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * delw
            self.bias -= self.lr * delb

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict_class(self, X):
        y_pred = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_pred]
