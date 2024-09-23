import numpy as np

class LogisticRegression:

    theta = None
    loss_history = None

    def __init__(self):
        pass

    def fit(self, X, y, theta=None, 
            alpha=.01, num_iters=100, lambda_=0):
        if not self.theta: theta = np.array([0] * X.shape[1])
        pack = [None, None]
        pack = self.gradient_descent(X, y, theta, alpha,
                                     num_iters, lambda_)
        self.theta, self.loss_history = pack

    def sigmoid(self, z): return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta, lambda_):
        m = len(y)  # number of training examples
        # Compute the hypothesis
        h = self.sigmoid(X @ theta)
        cost = np.mean(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))
        return cost

    def gradient_descent(self, X, y, theta, alpha, num_iters, lambda_):
        cost_history = [] 
        for _ in range(num_iters):
            h = self.sigmoid(X @ theta)
            loss = h - y
            theta -= alpha / y.size * X.T @ loss
            cost_history.append(self.compute_cost(X, y, theta, lambda_))
        return theta, cost_history
    
    def predict(self, X, theta, threshold = 0.5): return self.sigmoid(X @ theta)
    
    def draw_costs(self):
        import matplotlib.pyplot as plt 
        """ function to draw historical cost"""
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show() 
