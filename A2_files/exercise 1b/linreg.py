import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        self.w = None
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        X_augmented = numpy.concatenate([numpy.ones((n_samples, 1)), X], axis=1)
        
        self.w = numpy.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ t

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        X_augmented = numpy.concatenate([numpy.ones((n_samples, 1)), X], axis=1)
        
        predictions = X_augmented @ self.w
        
        return predictions