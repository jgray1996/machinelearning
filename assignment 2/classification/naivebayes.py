import numpy as np

class NaiveBayesClassifyer:

    var = None
    theta = None
    priors = None
    classes = None

    def fit(self, X, y):
        self.get_priors(y)
        self.get_likelihoods(X, y)

    def predict(self, X, prob="gaus"):
        if prob == "gaus":
            y_hats = self.predict_proba_gaussian(X)
            return self.classes[np.argmax(y_hats, axis=1)]
        if prob == "log":
            y_hats = self.predict_proba_log(X)
            return self.classes[np.argmax(y_hats, axis=1)]

    def get_priors(self, y):
        classes, counts = np.unique(y, return_counts=True)
        self.classes = classes
        self.priors = counts/len(y)

    def get_likelihoods(self, X, y):
        classes = np.unique(y)
        self.var = [np.var(X[y == c], axis=0) for c in classes]
        self.theta = [np.std(X[y == c], axis=0) for c in classes]
        return self.var, self.theta

    def predict_proba_gaussian(self, X):
        Pyx = list()
        for c in range(len(self.classes)):
            Pxy = np.exp(
                -0.5 * (X - self.theta[c]) ** 2 / self.var[c]
                ) / np.sqrt(2.0 * np.pi * self.var[c])
            Py = self.priors[c]
            Pyx.append(Pxy.prod(axis=1) * Py)
        Pyx = np.array(Pyx).T
        return Pyx / Pyx.sum(axis=1, keepdims=True)
    
    def predict_proba_log(self, X):
        log_Pyx = list()
        for c in range(len(self.classes)):
            log_Pxy = -0.5 * (np.log(self.var[c]) + (X - self.theta[c]
                                                     ) ** 2 / self.var[c])
            log_Py = np.log(self.priors[c])
            log_Pyx.append(log_Pxy.sum(axis=1) + log_Py)
        log_Pyx = np.array(log_Pyx).T
        log_Pyx -= log_Pyx.max(axis=1, keepdims=True)
        Pyx = np.exp(log_Pyx)
        return Pyx / Pyx.sum(axis=1, keepdims=True)
