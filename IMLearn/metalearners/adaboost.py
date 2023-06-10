import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # we will implement Adaboost algorithm
        # 1) a. Initialize the sample weights D to have
        # equal weights for all samples
        # b. weights stores the weights correspond to each weak learner in the AdaBoost
        # algorithm, while Each iteration corresponds  a new weak learner
        # c. models is empty
        self.D_ = np.ones(len(y)) / len(y)
        self.weights_ = np.zeros(self.iterations_)
        self.models_ = []
        # 2) loop over iterations number
        iter_number = max(0, self.iterations_)
        for ind in range(iter_number):
            # a. Fit a weak learner on the training data
            new_wl = self.wl_()
            # fit
            model = new_wl.fit(X, y * self.D_)
            self.models_.append(model)
            pred_y = model.predict(X)
            # b. Calculate the weighted error of the weak learner by summing the weights of misclassified samples
            err = 0
            for i in range(len(y)):
                if y[i] != pred_y[i]:
                    err += self.D_[i]
            # c. Calculate the learner weight using the formula  0.5 * log((1 - err) / err).
            amountOfSay = 0.5 * np.log(float(1. - err) / err)
            self.weights_[ind] = amountOfSay
            # d. Update the sample weights D by the equation
            sample_weight_eq = np.exp(-pred_y * y * self.weights_[ind])
            self.D_ = self.D_ * sample_weight_eq
            # e. Normalize the sample weights, so they sum up to 1
            self.D_ = self.D_ / np.sum(self.D_)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # represents the weighted predictions accumulated during the partial prediction process
        pred_weights = np.zeros(X.shape[0])
        # should be the minimum between T and self.iterations
        iter_num = min(T, self.iterations_)
        for i in range(0, iter_num):
            prediction = self.models_[i].predict(X)
            pred_weights += self.weights_[i] * prediction
        # calculate the sign of each sample according the pred_wight that had already calculated
        responses = np.zeros(X.shape[0])
        for i in range(len(pred_weights)):
            if pred_weights[i] > 0:
                responses[i] = 1
            elif pred_weights[i] < 0:
                responses[i] = -1
        return responses

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ..metrics import misclassification_error
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y, y_pred)
