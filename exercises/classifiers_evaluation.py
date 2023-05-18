from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        # array for storing the loss values
        loss_array = []

        # This function is  callback function which receives the object and uses it is loss
        # function to calculate the loss over the training set
        def callback_loss_function(fit, ignore1, ignore2):
            loss_array.append(fit.loss(X, y))

        Perceptron(callback=callback_loss_function).fit(X, y)

        fig = go.Figure(
            go.Scatter(x=list(range(len(loss_array))), y=loss_array, name="Misclassification Error", mode="lines",
                       marker=dict(color="blue")),
            layout=go.Layout(title="Perceptron algorithm's misclassification error\nover " + n + " Training data",
                             xaxis=dict(title="Training iteration"),
                             yaxis=dict(title="Misclassification Error"),
                             showlegend=True,
                             plot_bgcolor='white',
                             paper_bgcolor='white',
                             font=dict(size=12, color='black')
                             ))
        fig.write_image("Perceptron_Loss{}.png".format(f))


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        gaussian_model = GaussianNaiveBayes().fit(X, y)
        LDA_model = LDA().fit(X, y)
        # prediction:
        gaussian_predict = gaussian_model.predict(X)
        LDA_predict = LDA_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        accuracy_gaussian = round(accuracy(y, gaussian_predict) * 100, 3)
        accuracy_LDA = round(accuracy(y, LDA_predict) * 100, 3)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                "  Prediction by Gaussian classifier,"
                                " accuracy= {}%".format(accuracy_gaussian), "Prediction by LDA Classifier, "
                                                                            "accuracy= {}%".format(accuracy_LDA)),
                            horizontal_spacing=0.2)
        X_values = X[:, 0]
        y_values = X[:, 1]
        # Add the traces to the figure
        traceGaussianData = go.Scatter(x=X_values, y=y_values, mode='markers',
                                       marker=dict(color=gaussian_predict, colorscale='Bluered',
                                                   line=dict(width=1, color='black')))

        traceLDAData = go.Scatter(x=X_values, y=y_values, mode='markers',
                                  marker=dict(color=LDA_predict, colorscale='Jet', symbol='square',
                                              line=dict(width=1, color='black')))

        fig.add_traces([traceGaussianData, traceLDAData], rows=[1, 1], cols=[1, 2])

        # set subplot title Size
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=12)

        # Add `X` dots specifying fitted Gaussians' means
        gaussian_X = gaussian_model.mu_[:, 0]
        gaussian_y = gaussian_model.mu_[:, 1]
        traceXGaussian = go.Scatter(x=gaussian_X, y=gaussian_y, mode="markers",
                                    marker=dict(symbol="x", color="black", size=16))
        LDA_X = LDA_model.mu_[:, 0]
        LDA_y = LDA_model.mu_[:, 1]

        traceXLDA = go.Scatter(x=LDA_X, y=LDA_y, mode="markers", marker=dict(symbol="x", color="black", size=16))

        fig.add_traces([traceXGaussian, traceXLDA], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gaussian_model.classes_)):
            traceGaussianElipse = get_ellipse(gaussian_model.mu_[i], np.diag(gaussian_model.vars_[i]))
            traceLDAElipse = get_ellipse(LDA_model.mu_[i], LDA_model.cov_)
            fig.add_traces([traceGaussianElipse, traceLDAElipse], rows=[1, 1], cols=[1, 2])

        fig.update_layout(title_text="Guassian Model vs. LDA model - Over {}".format(f),
                          width=700, height=400, showlegend=False)
        fig.write_image("Gaussian_compared_LDA_{}.png".format(f))


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
