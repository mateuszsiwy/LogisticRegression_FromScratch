import numpy as np

def Sigmoid(x):
    return 1 / (1 + np.exp(-x)) ## where x is an array of scalar values

class LogisticRegression():
    """
    Parameters:
        n_iter, maximum number of iterations to be used by batch gradient descent
        lr: learning rate determining the size of steps in batch gradient descent
    
    """

    def __init__(self, n_iter=1000, lr=0.001):
        self.n_iter = n_iter
        self.lr = lr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) ## albo samo features
        self.bias = 0

        for _ in range(self.n_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            pred = Sigmoid(linear_pred)
            
            dw = (1/n_samples) * np.dot(X.T, (pred-y))
            db = (1/n_samples) * np.sum(pred-y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
            
            
            
            
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = Sigmoid(linear_pred)
        
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
