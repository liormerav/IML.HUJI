from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    # Randomly select 50 samples , we don't need to randomize because the order is randomize
    indices = np.arange(0, 50)

    # Split the data into training and testing portions using the randomly selected indices
    train_X, train_y = x[indices], y[indices]
    test_X, test_y = np.delete(x, indices, axis=0), np.delete(y, indices, axis=0)

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions

    ridge_alpha_range = np.linspace(0.005, 1, num=n_evaluations)
    lasso_alpha_range = np.linspace(0.05, 2, num=n_evaluations)
    ridge_arr = np.asarray(
        list(map(lambda lam: cross_validate(RidgeRegression(lam), np.asarray(train_X), np.asarray(train_y),
                                            mean_square_error, 5), ridge_alpha_range)))
    lasso_arr = np.asarray(list(map(lambda lam: cross_validate(Lasso(alpha=lam, max_iter=5000), np.asarray(train_X),
                                                               np.asarray(train_y), mean_square_error, 5),
                                    lasso_alpha_range)))

    fig = make_subplots(rows=2, cols=1, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"],
                        shared_xaxes=False)
    fig.update_layout(title=r"$\text{K fold MSE over Test and Validation Lasso and Ridge}$", width=500, height=400)
    fig.update_xaxes(title=r"$\lambda\$")

    # Add the scatter plots

    fig.add_trace(go.Scatter(x=ridge_alpha_range, y=ridge_arr[:, 1], name="Ridge Validation Error"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ridge_alpha_range, y=ridge_arr[:, 0], name="Ridge Train Error"), row=1, col=1)

    fig.add_trace(go.Scatter(x=lasso_alpha_range, y=lasso_arr[:, 1], name="Lasso Validation Error"), row=2, col=1)
    fig.add_trace(go.Scatter(x=lasso_alpha_range, y=lasso_arr[:, 0], name="Lasso Train Error"), row=2, col=1)

    fig.write_image("../figures/kfold.png")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = ridge_alpha_range[np.where(ridge_arr[:, 1] == np.min(ridge_arr[:, 1]))[0][0]]
    best_lasso = lasso_alpha_range[np.where(lasso_arr[:, 1] == np.min(lasso_arr[:, 1]))[0][0]]
    least_sqaure = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)
    ridge_error = RidgeRegression(lam=best_ridge).fit(train_X, train_y).loss(test_X, test_y)
    lasso_error = mean_square_error(test_y, Lasso(alpha=best_lasso).fit(train_X, train_y).predict(test_X))
    print(f"Best Ridge with alpha Parameter of", best_ridge)
    print(f"Best Lasso with alpha Parameter of", best_lasso)
    print(f"___________MODEL ERRORS__________________")
    print(f"Least squares ", least_sqaure)
    print(f"Ridge error ", ridge_error)
    print(f"Lasso error ", lasso_error)


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
