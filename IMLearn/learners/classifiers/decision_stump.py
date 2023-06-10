from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # we need to find the beast feature and threshold, so we will use the product function
        error = np.inf
        # generates pairs of values from the Cartesian product of range(X.shape[1]) and [-1, 1]
        iterator = iter(product(range(X.shape[1]), [-1, 1]))
        while True:
            try:
                # pair of j and sign
                j, sign = next(iterator)
            except StopIteration:
                break

            current_threshold, threshold_error_cur = self._find_threshold(X[:, j], y, sign)
            if threshold_error_cur < error:
                self.threshold_ = current_threshold
                self.j_ = j
                self.sign_ = sign
                error = threshold_error_cur

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        mask = X[:, self.j_] < self.threshold_
        result = np.where(mask, -self.sign_, self.sign_)
        return result

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Calculate the initial loss
        loss = np.sum(np.sign(labels) != np.sign(-sign))
        c_loss = loss

        # Combine values and labels into a single array
        concatenated_data = np.column_stack((values, labels))

        # Sort the data based on the values in the first column
        sort_data = concatenated_data[concatenated_data[:, 0].argsort()]

        # Initialize variables
        thr = np.inf
        i = len(values)

        # Iterate over the sorted data
        while i > 0:
            # Calculate the current difference
            cur_diff = np.sign(sort_data[i - 1, 1]) * sign

            # Update cumulative loss
            c_loss = c_loss - cur_diff

            # Update the loss if the cumulative loss is smaller
            loss = min(c_loss, loss)

            # Update the threshold if the cumulative loss is smaller or equal to the loss
            if c_loss <= loss:
                thr = sort_data[i - 1, 0]

            i -= 1

        # Calculate the normalized loss
        normalized_loss = loss / len(values)

        return thr, normalized_loss

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
