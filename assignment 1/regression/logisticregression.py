import numpy as np

class LogisticRegression:
    
    theta = None
    costs = None
    sigmoid = np.vectorize(lambda x:
                           1.0 / (1.0 + np.exp(-x))
                           if x > 0 else 
                           np.exp(x) / (1.0 + np.exp(x)))

    def __repr__(self):
        return f"LogisticRegression model, weights {self.theta}"


    def fit(self, X, y, alpha=0.5, lambda_=1, epochs=50):
        self.theta, self.costs = self.gradient_descent(X, y,
                                                       theta=np.zeros(X.shape[1]),
                                                       alpha=alpha,
                                                       lambda_=lambda_,
                                                       epochs=epochs)   
    
    def gradient_descent(self, X, y, theta,
                         alpha=0.1, lambda_=0.1, epochs=50):
        """
        Gradient descent for logistic regression with L2 regularization
        """
        theta = theta.copy()
        m = y.size
        costs = []

        for _ in range(epochs):
            # Step 1: Initial prediction
            z = X @ theta
            y_hat = self.sigmoid(z)  # sigmoid moved here for clarity
            
            # Step 2: Compute gradients
            error = y_hat - y
            gradients = (1/m) * X.T @ error
            
            # Add regularization term to gradients (excluding bias term)
            gradients[1:] += (lambda_/m) * theta[1:]
            
            # Step 3: Update parameters
            theta -= alpha * gradients
            
            # Step 4: Compute cost
            J = self.compute_cost(X, y, theta, lambda_)
            costs.append(J)

        return theta, costs


    def compute_cost(self, X, y, theta, lambda_):
        # parameters
        epsilon = 1e-15
        m = len(y)

        # step 1: Predition
        y_hat = self.sigmoid(X @ theta)
        # claude.ai suggestion to prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1-epsilon)

        # step 2: Calculate cost
        cost = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # step 3: Calculate regularization term
        reg_term = (lambda_/(2*m)) * np.sum(theta[1:] ** 2)
        
        # step 4: Add regularization term to calculated cost
        J = cost + reg_term
        return J 
    
    def predict(self, X):
        return self.sigmoid(X @ self.theta)
