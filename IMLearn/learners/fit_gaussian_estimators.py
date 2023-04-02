import numpy as np
import plotly.graph_objects as go

from IMLearn.learners import gaussian_estimators, UnivariateGaussian

""""
@author: lior.merav
This file uses the gaussian_estimator.py file 
"""


def question1(instance):
    """"
    This function fit a univariate Gaussian. Print the estimated expectation and variance
    """
    samples = np.random.normal(10, 1, size=1000)
    instance.fit(samples)
    print("(" + str(instance.mu_), str(instance.var_) + ")")
    return samples


def question2(instance, samples):
    """"
    this function calculates the mu estimation for each sample size and then plotting it
    """
    list_of_estimators = []
    for i in range(10, 1001, 10):
        list_of_estimators.append(np.abs(UnivariateGaussian().fit(samples[:i]).mu_ - instance.mu_))
    fig = go.Figure(
        go.Scatter(x=list(range(len(list_of_estimators))),
                   y=list_of_estimators,
                   mode="markers",
                   marker=dict(color="blue"))
    )
    fig.update_layout(
        template="simple_white",
        title="Absolute Distance Between Estimated and True Expectation",
        xaxis_title=r"Sample Size",
        yaxis_title=r"Sample Expectancy value estimator"
    )
    fig.show()


def question3(instance, samples):
    """"
    this function creates an scatter plot of the empirical PDF function under the fitted model
    """
    x_pdf_values = instance.pdf(samples)
    sort_indices = np.argsort(x_pdf_values)
    # Sort x and y arrays by their y values
    x_sorted = samples[sort_indices]
    y_sorted = x_pdf_values[sort_indices]
    fig = go.Figure(
        go.Scatter(x=x_sorted,
                   y=y_sorted,
                   mode="markers",
                   marker=dict(color="blue"))
    )
    fig.update_layout(
        title=r"empirical PDF function of univariate gaussian",
        xaxis_title=r"Samples",
        yaxis_title=r"Empirical PDF"
    )

    fig.show()


def question4(instance_mult, cov):
    """"
     this function Fit a multivariate Gaussian and print the estimated expectation and covariance matrix
    """
    mean = np.array([0, 0, 4, 0])
    samples = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
    instance_mult.fit(samples)
    print("M=")
    print(instance_mult.mu_)
    print("COV=")
    print(instance_mult.cov_)
    return samples


def question5_6(instance_mult, samples_mult, cov):
    """"
    This function calculate the log-likelihood for models with
    expectation µ = [ f1,0, f3,0] and the true covariance matrix defined above
    This function also prints the model which achieved the maximum log-likelihood value
    """
    f1_vals = np.linspace(-10, 10, 200)
    f3_vals = np.linspace(-10, 10, 200)
    matrix = np.zeros((200, 200))
    for index_f1 in range(f1_vals.shape[0]):
        for index_f3 in range(f3_vals.shape[0]):
            mean = np.array([f1_vals[index_f1], 0, f3_vals[index_f3], 0])
            matrix[index_f1, index_f3] = instance_mult.log_likelihood(mean, cov, samples_mult)
    fig = go.Figure(go.Heatmap(x=f3_vals, y=f1_vals, z=matrix))
    fig.update_layout(
        title=r"Log likelihood for models with f1, f3 features for multivariate gaussian",
        xaxis_title=r"f3 feature",
        yaxis_title=r"f1 feature"
    )

    indices = np.unravel_index(matrix.argmax(), matrix.shape)
    first_value = indices[0]
    second_value = indices[1]
    print("Vector is")
    vec = np.array([round(f1_vals[first_value], 3), 0, round(f3_vals[second_value], 3), 0])
    column_vector = vec.reshape(-1, 1)
    print(column_vector)
    fig.show()


def main():
    """"
    This function activates the functions for questions 1-6
    """
    instance_uni = gaussian_estimators.UnivariateGaussian()
    instance_mult = gaussian_estimators.MultivariateGaussian()
    samples_uni = question1(instance_uni)
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    question2(instance_uni, samples_uni)
    question3(instance_uni, samples_uni)

    samples_mult = question4(instance_mult, cov)
    question5_6(instance_mult, samples_mult, cov)


if __name__ == "__main__":
    main()
