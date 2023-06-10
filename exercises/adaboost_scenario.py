import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)
    arr_train = []
    arr_test = []
    # calculate the error for each model
    for ind in range(n_learners):
        arr_train_val_ind = adaBoost.partial_loss(train_X, train_y, ind + 1)
        arr_train.append(arr_train_val_ind)

        arr_test_val_ind = adaBoost.partial_loss(test_X, test_y, ind + 1)
        arr_test.append(arr_test_val_ind)
        range_ = list(range(1, n_learners + 1))
        fig = go.Figure(
            [go.Scatter(x=range_, y=arr_test, name="Error of Test", fill=None, mode="lines",
                        line=dict(color="green")),
             go.Scatter(x=range_, y=arr_train, name="Error of Train", fill='tonexty', mode="lines",
                        line=dict(color="blue")), ],
            layout=go.Layout(title="Training- and Test Errors as a Function of The Number of Fitted Learners",
                             xaxis=dict(title="Number of Iterations"),
                             yaxis=dict(title="Error "),
                             showlegend=True,
                             plot_bgcolor='white',
                             paper_bgcolor='white',
                             font=dict(size=12, color='black')
                             ))
    if noise ==0.4:
        fig.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = ['diamond' if y == 1 else 'circle' for y in test_y]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf Number of classifiers: {{{t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=0.03)
    for i, t in enumerate(T):
        """"
        This inner function returns the calculation of adaBoost.partial_predict with a given x as a parameter
        """

        def prediction_func(X, t=t):
            return adaBoost.partial_predict(X, t)

        decision_surf = decision_surface(prediction_func, lims[0], lims[1], density=60, showscale=False,
                                         colorscale='Greens')
        scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                             marker=dict(color=test_y, symbol=symbols,
                                         colorscale=[[0, 'gray'], [1, 'green']]))

        fig.add_traces([decision_surf, scatter], rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models}}$", margin=dict(t=100),
                      height=800, width=800).update_xaxes(visible=False).update_yaxes(visible=False)
    # fig.show()

    # Question 3: Decision surface of best performing ensemble
    # find the model's number of the minimum error
    best_err = np.argmin(arr_test) + 1
    """"
     This inner function returns the calculation of adaBoost.partial_predict with a given the best model
    """

    def best_prediction_func(X):
        return adaBoost.partial_predict(X, best_err)

    # Calculate accuracy
    accuracy = 1 - arr_test[best_err - 1]

    fig = go.Figure([
        decision_surface(best_prediction_func, lims[0], lims[1], density=60, showscale=False),
        go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=test_y,
                symbol=symbols,
                colorscale=[[0, 'red'], [1, 'blue']],
                cmin=0,
                cmax=1
            )
        )
    ],
        layout=go.Layout(
            width=400,
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title=f"Best Ensemble: {best_err} Accuracy: {accuracy}",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
    )
    #    fig.show()

    # Question 4: Decision surface with weighted samples

    updated_D = adaBoost.D_ / adaBoost.D_.max() * 5
    fig = go.Figure()

    # Add decision surface trace
    fig.add_trace(decision_surface(adaBoost.predict, lims[0], lims[1], density=60, showscale=False))

    # Add scatter plot trace
    fig.add_trace(go.Scatter(
        x=train_X[:, 0],
        y=train_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=updated_D,
            color=train_y,
            symbol=symbols
        )
    ))

    # Set layout
    fig.update_layout(
        width=400,
        height=400,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Training - point size proportional to weight"
    )

    # Display the figure
    if noise==0.4:
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # no noise
    fit_and_evaluate_adaboost(0)
    # noise of 0.4
    fit_and_evaluate_adaboost(0.4)

