from IMLearn.utils import split_train_test, utils
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
    df = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    df = df.dropna().drop_duplicates()
    return df


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
    columns = ['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront',
               'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
               'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']
    # if its train we would "clean" the data
    global train_col
    if y is not None:
        data = clean_data(X, y)

        data.columns = columns
        # treat zip code with method: one hot encoding
        data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'])
        train_col = data.columns.tolist()
        # limit room number to 30
        data = data[data["bedrooms"] < 30]


    else:
        # test
        data = X
        data = pd.get_dummies(data, prefix='zipcode_', columns=['zipcode'])
        data = data.drop(columns=[col for col in data.columns if col not in train_col])
        # get set difference between columns in DataFrame and columns to check
        missing_cols = set(train_col).difference(data.columns)

        # add missing columns with default value of 0
        for col in missing_cols:
            data.insert(loc=len(data.columns), column=col, value=0)

    # drop unesscary columns
    col_drop = ['sqft_living15', 'sqft_lot15', 'lat', 'long', 'id', 'date']
    data = data.drop(col_drop, axis=1)
    # fix values on dataset in order to preform learning
    # change negative values to the average of column
    # Identify columns with negative values

    numerical_data = data.select_dtypes(include=np.number)
    negative_cols = numerical_data.columns[(numerical_data < 0).any()]

    # Loop over negative columns and replace negative values with the column average
    for col in negative_cols:
        mask = data[col] < 0  # Create a boolean mask of negative values
        col_mean = data.loc[~mask, col].mean()  # Calculate the mean of non-negative values
        data.loc[mask, col] = col_mean  # Replace negative values with the mean


    if y is not None:
        return data.drop('price', axis=1), data['price']
    return data.drop('price', axis=1)


""""
    # We expect non-negative values in all the columns which are number typed
    data = data[data.select_dtypes(include=[np.number]).ge(0).all(1)]
    # drop if the date is not in the correct format
    for i in range(len(data)):
        try:
            data.strptime(data.loc[i, 'date'], '%Y%m%dT000000')
        except ValueError:
            data.drop(i, inplace=True)
    # check year format in column yr_built
    data = data[pd.to_numeric(data['yr_built'], errors='coerce').notnull()]
    data['yr_built'] = data['yr_built'].astype(int)
    data = data[(data['yr_built'] >= 0) & (data['yr_built'] <= 2023)]
    # check year format in column yr_renovated
    data = data[pd.to_numeric(data['yr_renovated'], errors='coerce').notnull()]
    data['yr_renovated'] = data['yr_renovated'].astype(int)
    data = data[(data['yr_renovated'] >= 0) & (data['yr_renovated'] <= 2023)]
"""


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
    X.drop(['sqft_living15', 'sqft_lot15', 'lat', 'long', 'id', 'date'], axis='columns', inplace=True)
    pearson_lst = []

    for col in X:
        numerator = np.mean((X[col] - np.mean(X[col])) * (y - np.mean(y)))

        # calculate dominator
        pearson_domi = np.std(X[col]) * np.std(y)
        # lst pf pearson corrleation values
        pearson = numerator / pearson_domi
        pearson_lst.append(pearson)
        # Define a list of colors
        colors = ["blue"] * len(X.columns)

        # Create a scatter plot with OLS trendline for each column in X
        scatter_plots = [px.scatter(x=X[col], y=y, trendline="ols",
                                    color_discrete_sequence=[colors[i]],
                                    title=f"Correlation of {col} and Response\n Pearson equals to {pearson}",
                                    labels={"x": f"{col}", "y": "Response"})
                         for i, col in enumerate(X.columns)]

        # Show all scatter plots
        for fig in scatter_plots:
            fig.write_image(output_path + f"correlation_pearson{col}.png")


if __name__ == '__main__':
    np.random.seed(0)

    df = pd.read_csv('C:/Users/liorm/PycharmProjects/IML.HUJI/datasets/house_prices.csv')

    # Question 1 - split data into train and test sets
    df_without_price = df.drop("price", axis=1)
    train_X, train_Y, test_X, test_Y = split_train_test(df_without_price, df['price'])

    # Question 2 - Preprocessing of housing prices dataset

    train_X,train_Y = preprocess_data(train_X, train_Y)
    test_X= preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(test_X, test_Y,'C:/Users/liorm/PycharmProjects/IML.HUJI/datasets/house_prices.csv')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    import pandas as pd
    import numpy as np

    # Step 1: Compute loss for each percentage value
    array_var_avg = np.array([])
    array_loss_avg = np.array([])

    for p in range(10,101):

        # compute loss for each sample
        loss_array = np.array([])
        for j in range(10):
            X = train_X.sample(frac=p / 100.0)
            Y = train_Y.loc[X.index]
            lr = LinearRegression(include_intercept=True).fit(X, Y)

            loss = lr.loss(test_X, test_Y)
            loss_array =np.append(loss_array,loss)

        # compute mean and std of loss array
        loss_mean = np.mean(loss_array)
        loss_std = np.std(loss_array)

        # append mean and std to arrays
        array_loss_avg= np.append(array_loss_avg,loss_mean)
        array_var_avg =np.append(array_var_avg,2 * loss_std)

    # Step 2: Plot mean loss with error bars
    import matplotlib.pyplot as plt

    range = list(range(10, 101))

    fig = go.Figure(
        [go.Scatter(x=range, y=array_loss_avg - 2 * array_var_avg, fill=None, mode="lines", line=dict(color="blue")),
         # changed line color to blue
         go.Scatter(x=range, y=array_loss_avg + 2 * array_var_avg, fill='tonexty', mode="lines", line=dict(color="blue")),
         # changed line color to blue
         go.Scatter(x=range, y=array_var_avg, mode="markers+lines", marker=dict(color="black"))],
        layout=go.Layout(title="Mean Squared Error (MSE) as a Function Of Training Size",
                         xaxis=dict(title="Sample Size (%)"),
                         yaxis=dict(title="MSE"),
                         showlegend=False,
                         plot_bgcolor='white',  # added plot background color
                         paper_bgcolor='white',  # added paper background color
                         font=dict(size=12, color='black')  # modified font properties
                         ))

    fig.show()

    """
    import pandas as pd
        import numpy as np
    
        array_var_avg = []
        array_loss_avg = []
        for i in range(10, 101):
            # compute size of sample
            n = int(i / 100 * len(train_X))
    
            # concat data and sample from it
            df = pd.concat([train_X.reset_index(drop=True), train_Y.reset_index(drop=True)], axis=1)
            samples = df.sample(n)
    
            # compute loss for each sample
            loss_array = []
            for j in range(10):
                Y = samples[['price']]
                X = samples.drop('price', axis=1)
                lr = LinearRegression(include_intercept=True).fit(X, Y)
                loss = lr.loss(test_X, test_Y)
                loss_array.append(loss)
    
            # compute mean and std of loss array
            loss_mean = np.mean(loss_array)
            loss_std = np.std(loss_array)
    
            # append mean and std to arrays
            array_loss_avg.append(loss_mean)
            array_var_avg.append(2 * loss_std)
    
        # plot mean loss with error bars
        plt.errorbar(range(10, 101), array_loss_avg, yerr=array_var_avg)
        plt.xlabel('Training data size (%)')
        plt.ylabel('Average loss')
        plt.show()
    """
