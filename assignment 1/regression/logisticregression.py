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

    def compute_cost(self, X, y, theta, lambda_):
        m = len(y)  # number of training examples
        # Compute the hypothesis
        h = X @ theta
        # Compute the squared errors
        square_errors = (h - y) ** 2
        # Compute the regularization term (excluding theta[0])
        regularization_term = (lambda_ / (2 * m)) * np.sum(theta ** 2)
        # Compute the cost function
        J_val_vec = (1 / (2 * m)) * np.sum(square_errors) + regularization_term
        return J_val_vec

    def gradient_descent(self, X, y, theta, alpha, num_iters, lambda_):
        # initialize list of costs
        cost_history = [] 
        m,n = X.shape
        for _ in range(num_iters):
            h = X @ theta.T
            loss = h - y
            grad = (X.T @ loss + lambda_ * theta) / m
            theta = theta - grad.T * alpha
            cost_history.append(self.compute_cost(X, y, theta, lambda_))
        return theta, cost_history
    
    def predict(self, x):
        return np.polyval(self.theta, x)
    
    def draw_costs(self):
        import matplotlib.pyplot as plt 
        """ function to draw historical cost"""
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show() 
