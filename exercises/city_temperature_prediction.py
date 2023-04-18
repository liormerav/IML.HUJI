import math
import random

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date'])
    columns = ['Country', 'City', 'Date', 'Year', 'Month', 'Day', 'Temp']
    data.columns = columns
    data = data.dropna().drop_duplicates()
    # if Month is not in 1,2..12 return abs(x)%12
    data['Month'] = data['Month'].apply(lambda x: abs(x) % 12 if x not in range(1, 13) else x)
    data = data[data['Year'].astype(str).str.len() == 4]
    # if Day is not in 1,2..31 return abs(x)%12
    data['Day'] = data['Day'].apply(lambda x: abs(x) % 31 if x not in range(1, 32) else x)
    data = data[data['Temp'] > 0]
    # add new DayOfYear column
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data


def specific_from_israel(data):
    import plotly.graph_objs as go

    # Subset data for Israel only
    israel_data = data[data['Country'] == 'Israel']
    israel_data['Year'] = israel_data['Year'].astype(str)
    # Create a scatter plot with color-coded dots for each year
    fig = px.scatter(x=israel_data['DayOfYear'], y=israel_data['Temp'], color=israel_data['Year'],
                     title=f"Average Daily Temperature in Israel as a Function of Day of Year",
                     labels={"x": f"DayOfYear", "y": "Temperature"}, )

    israel_data['std'] = israel_data.groupby('Month')['Temp'].agg('std')
    israel_data_std = israel_data.groupby(["Month"]).agg(std=("Temp", "std"))
    fig = px.bar(x=range(1, 13), y=israel_data_std['std'],
                 title=f"SD of Temp by The Months Over the Years",
                 labels={"x": f"Month", "y": " SD of Temperature"}, )


def question_3(df):
    data_grouped = df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std"))
    fig = px.line(data_grouped, x=data_grouped['Month'], y=data_grouped['mean'], color='Country',
                  error_y=data_grouped['std'], title='Average Monthly Temperature by Country',
                  labels={'avg_temp': 'Temperature (C)', 'Month': 'Month', "x": f"Month", "y": "Mean Temperature"})
    fig.show()


def question_4(df):
    israel_data = df[df['Country'] == 'Israel']
    array_of_loss = np.zeros(10)
    train_X, train_Y, test_X, test_Y = split_train_test(pd.DataFrame(israel_data['DayOfYear']),
                                                        pd.Series(israel_data['Temp']))
    for k in range(1, 11):
        polynomial_fitting = PolynomialFitting(k)
        polynomial_fitting.fit(train_X.values.flatten(), train_Y.values)
        loss = polynomial_fitting.loss(test_X.values.flatten(), test_Y.values)
        loss = round(loss, 2)
        array_of_loss[k - 1] = loss
    print(array_of_loss)
    fig = px.bar(x=range(1, 11), y=array_of_loss, title=f"Loss of Polynomial Model Over the Test Set",
                 color_discrete_sequence=['red'],
                 labels={"x": f"K value", "y": " Loss"}, text=[f"Loss: {loss}" for loss in array_of_loss])



def question_5(df):
    israel_data = df[df['Country'] == 'Israel']
    model = PolynomialFitting(k=5).fit(israel_data['DayOfYear'].values, israel_data['Temp'].values)
    jordan_loss = round(model.loss(df[df.Country == "Jordan"].DayOfYear, df[df.Country == "Jordan"].Temp), 2)
    south_africa_loss = round(
        model.loss(df[df.Country == "South Africa"].DayOfYear, df[df.Country == "South Africa"].Temp), 2)
    netherlands_loss = round(
        model.loss(df[df.Country == "The Netherlands"].DayOfYear, df[df.Country == "The Netherlands"].Temp), 2)

    data = pd.DataFrame({"Country": ["Jordan", "South Africa", "The Netherlands"],
                         "Loss": [south_africa_loss, netherlands_loss,jordan_loss]})

    fig = px.bar(data, x="Country", y="Loss", text="Loss", color="Country",
                 title="Loss value of different Countries with K=5")

    fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    df = load_data('C:/Users/liorm/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv')
    # Question 1 - Load and preprocessing of city temperature dataset

    # Question 2 - Exploring data for specific country
    # specific_from_israel(df)

    # Question 3 - Exploring differences between countries
    # question_3(df)

    # Question 4 - Fitting model for different values of `k`
    question_4(df)

    # Question 5 - Evaluating fitted model on different countries
    question_5(df)
