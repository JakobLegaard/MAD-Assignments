import numpy as np

class RegularizedLinearRegression():
    """
    Regularized linear regression implementation.
    Adds L2 penalty term λ * w^T * w to the loss function.
    """

    def __init__(self, lambda_reg=0.0):
        """
        Parameters
        ----------
        lambda_reg : float
            Regularization parameter λ (default: 0.0 for no regularization)
        """
        self.w = None
        self.lambda_reg = lambda_reg
            
    def fit(self, X, t):
        """
        Fits the regularized linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples,] or [n_samples, 1]
        """        
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Flatten t if needed
        if t.ndim == 2:
            t = t.flatten()

        n_samples, n_features = X.shape
        
        # Augment X with column of ones for intercept
        X_augmented = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        
        # Regularized least squares solution: w = (X^T X + λI)^(-1) X^T t
        # Note: We don't regularize the intercept term (first weight)
        # Create regularization matrix
        reg_matrix = self.lambda_reg * np.eye(X_augmented.shape[1])
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Compute optimal weights
        XtX = X_augmented.T @ X_augmented
        Xtt = X_augmented.T @ t
        
        self.w = np.linalg.inv(XtX + reg_matrix) @ Xtt

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
        X_augmented = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        
        predictions = X_augmented @ self.w
        
        return predictions
    
    def get_weights(self):
        """
        Returns the learned weights.
        
        Returns
        -------
        weights : Array containing [w0, w1, ..., wD]
        """
        return self.w