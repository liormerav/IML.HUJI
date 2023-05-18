from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.array(list(set(y)))
        # calculate probabilities of each class
        class_counts = np.bincount(y)
        class_probabilities = class_counts / len(y)
        self.pi_ = class_probabilities
        self.mu_ = []
        self.vars_ = []
        # we will calculate the mean and the variance for each class
        for c in self.classes_:
            # samples of the current class
            X_current_class = X[y == c]

            # mean for the current class
            mean = np.mean(X_current_class, axis=0)
            self.mu_.append(mean)
            # variance for the current class
            var = np.var(X_current_class, axis=0, ddof=1)
            self.vars_.append(var)
        # convert list to array
        self.mu_ = np.array(self.mu_)
        self.vars_ = np.array(self.vars_)

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
        likelihood = self.likelihood(X)
        # choose maximum likelihood for each class
        class_index_max = np.argmax(likelihood, axis=1)
        predict = self.classes_[class_index_max]
        return predict

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

        num_samples = X.shape[0]
        number_classes = self.mu_.shape[0]
        # define array
        likelihood_arr = np.zeros((num_samples, number_classes))

        for i in range(number_classes):
            x_mu = X - self.mu_[i]
            dominator = np.sqrt(2 * np.pi * self.vars_[i])
            product = np.exp(-0.5 * (x_mu ** 2) / self.vars_[i]) / dominator
            # calculate the multiplication of each row
            x_class_likelihood = np.prod(product, axis=1)
            # assigns the likelihood values to the i-th column
            likelihood_arr[:, i] = x_class_likelihood * self.pi_[i]

        return np.array(likelihood_arr)

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
        return misclassification_error(y, self._predict(X))
