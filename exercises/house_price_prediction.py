import plotly

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

train_col = []


def clean_data(x, y):
    """"
    function cleans the data in the training set (remove duplicates and na values)
    """
    m = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    # drop Na just in the relevant columns
    m.replace(['NA', 'N/A', None], np.nan, inplace=True)
    m = m.dropna(subset=m.columns[~m.columns.isin(['sqft_living15', 'sqft_lot15', 'lat', 'long', 'id', 'date'])])
    m = m.drop_duplicates()
    return m


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X_columns = ['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
                 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                 'lat', 'long', 'sqft_living15', 'sqft_lot15']

    # define the desired column order
    # use the loc method to reorder the columns
    X = X.loc[:, X_columns]
    # if its train we would "clean" the data
    global train_col
    if y is not None:
        data = clean_data(X, y)
        # limit room number to 30
        if pd.api.types.is_numeric_dtype(data['bedrooms']):
            data = data[data["bedrooms"] < 30]
        # limit sqft living and sqft lot to 15
        if pd.api.types.is_numeric_dtype(data['sqft_living']):
            data = data[data["sqft_living"] > 15]
        if pd.api.types.is_numeric_dtype(data['sqft_lot']):
            data = data[data["sqft_lot"] > 15]

    else:
        # test
        data = X


    # drop unesscary columns
    col_drop = ['sqft_living15', 'sqft_lot15', 'lat', 'long', 'id', 'date']
    data = data.drop(col_drop, axis=1)
    # fix values on dataset in order to preform learning
    # change negative values to the average of column
    # Identify columns with negative values

    data = data.apply(pd.to_numeric, errors='coerce')

    numerical_data = data.select_dtypes(include=np.number)
    negative_cols = numerical_data.columns[(numerical_data < 0).any()]

    # Loop over negative columns and replace negative values with the column average
    for col in negative_cols:
        mask = data[col] < 0  # Create a boolean mask of negative values
        col_mean = data.loc[~mask, col].mean()  # Calculate the mean of non-negative values
        data.loc[mask, col] = col_mean  # Replace negative values with the mean
    for col in numerical_data:  # replace Na values
        col_mean = max(data[col][data[col] >= 0].mean(), 0)
        data[col] = data[col].replace(['NA', 'N/A', None, np.nan], col_mean)

    # make yr_renovated binary column
    data['yr_renovated'] = data['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)
    # if water front is not 0 or 1 change to 0
    data['waterfront'] = data['waterfront'].apply(lambda x: 0 if x != 0 and x != 1 else x)
    if y is not None:
        # treat zip code with method: one hot encoding
        data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'],dummy_na=False)
        train_col = data.columns.tolist()
        return data.drop('price', axis=1), data['price']

    data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'], dummy_na=False)
    data = data.drop(columns=[col for col in data.columns if col not in train_col])
    # get set difference between columns in DataFrame and columns to check
    missing_cols = set(train_col).difference(data.columns)

    # add missing columns with default value of 0
    for col in missing_cols:
        data.insert(loc=len(data.columns), column=col, value=0)
    data = data.reindex(columns = train_col,fill_value=0)
    return data.drop('price', axis=1)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~X.columns.str.startswith('zipcode_')]
    pearson_lst = []

    for col in X:
        numerator = np.mean((X[col] - np.mean(X[col])) * (y - np.mean(y)))

        # calculate dominator
        pearson_domi = np.std(X[col]) * np.std(y)
        # lst pf pearson corrleation values
        pearson = numerator / pearson_domi
        pearson_lst.append(pearson)
        # Define a list of colors

        # Create a scatter plot with OLS trendline for each column in X
        fig = px.scatter(x=X[col], y=y, trendline="ols",
                         color_discrete_sequence=['blue'],
                         title=f"Correlation of {col} and Response\n Pearson equals to {pearson}",
                         labels={"x": f"{col}", "y": "Response"})

        fig.write_image(output_path + f"/pearson{col}.png", engine='orca')


if __name__ == '__main__':
    np.random.seed(0)

    df = pd.read_csv('C:/Users/liorm/PycharmProjects/IML.HUJI/datasets/house_prices.csv')

    # Question 1 - split data into train and test sets
    df_without_price = df.drop("price", axis=1)
    train_X, train_Y, test_X, test_Y = split_train_test(df_without_price, df['price'])

    # Question 2 - Preprocessing of housing prices dataset

    train_X, train_Y = preprocess_data(train_X, train_Y)
    test_X = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(test_X, test_Y, 'C:/Users/liorm/PycharmProjects/IML.HUJI/datasets')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # Step 1: Compute loss for each percentage value
    array_var_avg = np.array([])
    array_loss_avg = np.array([])

    for p in range(10, 101):

        # compute loss for each sample
        loss_array = np.array([])
        for j in range(10):
            X = train_X.sample(frac=p / 100.0)
            Y = train_Y.loc[X.index]
            linear_reg = LinearRegression(include_intercept=True).fit(X, Y)

            loss = linear_reg.loss(test_X, test_Y)
            loss_array = np.append(loss_array, loss)

        # compute mean and std of loss array
        loss_mean = np.mean(loss_array)
        loss_std = np.std(loss_array)

        # append mean and std to arrays
        array_loss_avg = np.append(array_loss_avg, loss_mean)
        array_var_avg = np.append(array_var_avg, loss_std)

    # Step 2: Plot mean loss with error bars

    range_ = list(range(10, 101))
    fig = go.Figure(
        [go.Scatter(x=range_, y=array_loss_avg + 2 * array_var_avg, name="", fill=None, mode="lines",
                    line=dict(color="blue")),
         go.Scatter(x=range_, y=array_loss_avg - 2 * array_var_avg, name="", fill='tonexty', mode="lines",
                    line=dict(color="blue")),
         go.Scatter(x=range_, y=array_loss_avg, name="Loss", mode="markers+lines", marker=dict(color="black"))],
        layout=go.Layout(title="Mean Squared Error (MSE) as a Function Of Training Size",
                         xaxis=dict(title="Sample Size (%)"),
                         yaxis=dict(title="MSE"),
                         showlegend=True,
                         plot_bgcolor='white',
                         paper_bgcolor='white',
                         font=dict(size=12, color='black')
                         ))
    #fig.write_image('C:/Users/liorm/PycharmProjects/IML.HUJI/datasets' + f"/Loss.png", engine='orca')
