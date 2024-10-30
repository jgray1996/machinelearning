import numpy as np

class LinearRegression:
    
    theta = None
    costs = None

    def __repr__(self):
        return f"LinearRegression model, weights {self.theta}"


    def fit(self, X, y, alpha=0.5, lambda_=1, epochs=50):
        self.theta, self.costs = self.gradient_descent(X, y,
                                                       alpha=alpha,
                                                       lambda_=lambda_,
                                                       epochs=epochs)   
    
    def gradient_descent(self, X, y, theta, alpha=0.5, lambda_=1, epochs=50):
        theta = theta.copy()
        costs = []
        m = y.size
        for _ in range(epochs):
            y_hat = X @ theta
            error = y_hat - y
            theta -= alpha/m * (X.T @ error + lambda_ * theta)
            cost = self._compute_cost(error, lambda_)
            costs.append(cost)
        return theta, costs
    
    def compute_cost(self, X, y, theta, lambda_=1):
        theta = theta.copy()
        y_hat = X @ theta
        error = y_hat - y
        J = np.mean(lambda_ * error ** 2) / 2
        return J
    
    def _compute_cost(self, error, lambda_):
        return np.mean(lambda_ * error ** 2)/2
    
    def predict(self, X):
        if self.theta:
            return X @ self.theta
        else:
            raise ValueError("Model not fit yet, theta not found")
