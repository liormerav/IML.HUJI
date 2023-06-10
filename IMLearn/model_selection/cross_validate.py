from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator
from sklearn.model_selection import KFold


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    kf = KFold(n_splits=cv)

    train_scores, validation_scores = [], []
    for train_index, validation_index in kf.split(X):
        # Split the data into training and validation sets for the current fold
        X_train = np.concatenate([X[train_index]])
        X_val = np.concatenate([X[validation_index]])
        y_train = np.concatenate([y[train_index]])
        y_val = np.concatenate([y[validation_index]])

        # Fit the estimator on the training data
        fit = deepcopy(estimator).fit(X_train, y_train)

        # Compute the scores for the training and validation sets
        train_scores.append(scoring(y_train, fit.predict(X_train)))
        validation_scores.append(scoring(y_val, fit.predict(X_val)))

    # Compute the average scores over all folds
    train_score = np.mean(train_scores)
    validation_score = np.mean(validation_scores)

    return train_score, validation_score
