from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    train_proportion = round(train_proportion, 2)

    # The idea is to take all indices and then shuffle them to get randomly indices order
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_samples_num = int(np.ceil(train_proportion * len(X)))
    # Consider computer's numbers representation
    test_samples_num = int(np.floor(round((1 - round(train_proportion, 1)) * len(X))))
    train_indices = indices[:train_samples_num]
    test_indices = indices[train_samples_num:train_samples_num + test_samples_num]

    train_X = X.iloc[train_indices, :]
    train_y = y.iloc[train_indices]
    test_X = X.iloc[test_indices, :]
    test_y = y.iloc[test_indices]

    return train_X, train_y, test_X, test_y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
