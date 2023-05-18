from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # first we will obtain number of classes in y
        self.classes_ = np.array(list(set(y)))
        # calculate probabilities of each class
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        self.pi_ = class_probabilities
        # The expectancy is the average over each class as ew have seen in class
        self.mu_ = []
        for c in self.classes_:
            # Get indices of the class
            indices = np.where(y == c)
            mean_class = np.mean(X[indices], axis=0)
            self.mu_.append(mean_class)
        # Convert list to numpy array
        self.mu_ = np.array(self.mu_)
        # calculate unbiased estimator
        unbiased_division = (len(X) - len(self.classes_))
        # creates a list y_mu that contains the mean vectors corresponding to each value in y
        y_mu = [self.mu_[np.where(self.classes_ == y_val)][0] for y_val in y]
        y_mu = np.array(y_mu)
        x_mu = X - y_mu
        self.cov_ = np.einsum("ab,ac->abc", x_mu, x_mu).sum(axis=0) / unbiased_division
        self.cov_inv_ = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # calculate the liklihood for the sample to belong each class
        likelihood = self.likelihood(X)
        # finds the indices of the maximum likelihood values for each sample
        class_indices = np.argmax(likelihood, axis=1)
        # maps the predicted class indices to their corresponding class labels
        prediction = self.classes_[class_indices]
        return prediction

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        dominator = (2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_)
        dominator_sqrt = np.sqrt(dominator)
        # we want the mu and x would have the same dimensions , so we would be able to subtract
        x_mu = np.expand_dims(X, axis=1) - self.mu_
        inner_product = x_mu @ self.cov_inv_
        nominator = inner_product * x_mu
        sum_nominator = np.sum(nominator, axis=2)
        exp_nominator = np.exp(-0.5 * sum_nominator)
        likelihood = (exp_nominator / dominator_sqrt) * self.pi_
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
