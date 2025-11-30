import numpy

class WeightedLinearRegression():
    """
    Weighted linear regression implementation.
    Uses weights alpha_n for each data point in the loss function.
    """

    def __init__(self):
        self.w = None
            
    def fit(self, X, t, alpha):
        """
        Fits the weighted linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1] or [n_samples,]
        alpha : Array of shape [n_samples,] - weights for each sample
        """        
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Flatten t if needed
        if t.ndim == 2:
            t = t.flatten()

        n_samples = X.shape[0]
        
        # Augment X with column of ones for intercept
        X_augmented = numpy.concatenate([numpy.ones((n_samples, 1)), X], axis=1)
        
        # Create diagonal weight matrix A
        A = numpy.diag(alpha)
        
        # Compute optimal weights using formula: w = (X^T A X)^(-1) X^T A t
        # This is the solution derived in Exercise 1a
        XtAX = X_augmented.T @ A @ X_augmented
        XtAt = X_augmented.T @ A @ t
        
        self.w = numpy.linalg.inv(XtAX) @ XtAt

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples,]
        """                     
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        X_augmented = numpy.concatenate([numpy.ones((n_samples, 1)), X], axis=1)
        
        predictions = X_augmented @ self.w
        
        return predictions