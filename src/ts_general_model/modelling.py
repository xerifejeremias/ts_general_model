import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from linearmodels.panel import PanelOLS
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from linearmodels.panel import PanelOLS
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

def fit_sarima_model(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Fits a SARIMA model for each market and product_code.
    
    Parameters:
    - df: DataFrame, the data to model
    - order: tuple, the (p, d, q) order of the model
    - seasonal_order: tuple, the (P, D, Q, s) seasonal order of the model
    
    Returns:
    - Dictionary of fitted models
    """
    models = {}
    for (market, product_code), group in df.groupby(['market', 'product_code']):
        model = SARIMAX(group['y'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        models[(market, product_code)] = results
        print(f"Fitted SARIMA model for market: {market}, product_code: {product_code}")
    
    return models

def test_autocorrelation(df, target_column):
    """
    Tests for autocorrelation in the target column and plots the ACF graph.
    
    Parameters:
    - df: DataFrame, the data containing the target variable
    - target_column: str, the name of the target column to test
    
    Returns:
    - DataFrame with autocorrelation values
    """
    # Calculate autocorrelation values
    acf_values = acf(df[target_column], fft=True)
    
    # Plot ACF
    plot_acf(df[target_column])
    plt.title(f'ACF Plot for {target_column}')
    plt.show()
    
    # Create a DataFrame for autocorrelation values
    acf_df = pd.DataFrame({'Lag': range(len(acf_values)), 'Autocorrelation': acf_values})
    
    return acf_df

def fit_panel_data_model(df, index_columns=['market', 'product_code', 'ds'], dependent_var='y', independent_vars=['var1', 'var2', 'var3', 'var4', 'var5']):
    """
    Fits a panel data model (Fixed Effects) for all market and product_code combinations.
    
    Parameters:
    - df: DataFrame, the data to model
    
    Returns:
    - Fitted panel data model
    """
    # Set the index for panel data
    df = df.set_index(index_columns)
    
    # Define the dependent and independent variables
    y = df[dependent_var]
    X = df[independent_vars]
    X = add_constant(X)
    
    # Fit the Fixed Effects model
    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit()
    
    print("Fitted Panel Data Model (Fixed Effects):")
    print(results.summary)
    
    return results

def fit_ols_model(df, group_columns=['market', 'product_code'], dependent_var='y', independent_vars=['var1', 'var2', 'var3', 'var4', 'var5']):
    """
    Fits an OLS model for each market and product_code.
    
    Parameters:
    - df: DataFrame, the data to model
    
    Returns:
    - Dictionary of fitted models
    """
    models = {}
    for group_keys, group in df.groupby(group_columns):
        X = group[independent_vars]
        X = add_constant(X)  # Adds a constant term to the predictor
        y = group[dependent_var]
        model = OLS(y, X).fit()
        models[group_keys] = model
        print(f"Fitted OLS model for group: {group_keys}")
        print(model.summary())  # Print the summary of the model
    
    return models

def train_test_split_by_date(df, date_str, date_column='ds'):
    """
    Splits the DataFrame into train and test sets based on a specific date.
    
    Parameters:
    - df: DataFrame, the data to split
    - date_str: str, the date to split the data on in "YYYY-MM-DD" format
    - date_column: str, the name of the column containing datetime information
    
    Returns:
    - train_df: DataFrame, the training set
    - test_df: DataFrame, the testing set
    """
    # Convert the date string to a Timestamp
    split_date = pd.Timestamp(date_str)
    
    # Ensure the date_column exists in the DataFrame
    if date_column not in df.columns:
        raise ValueError(f"Column '{date_column}' not found in DataFrame.")
    
    # Split the DataFrame
    train_df = df[df[date_column] < split_date]
    test_df = df[df[date_column] >= split_date]
    
    return train_df, test_df

def test_heteroskedasticity(series, plot=False):
    """
    Tests for heteroskedasticity using the Breusch-Pagan test.
    
    Parameters:
    - series: pd.Series, the time series data to test
    - plot: bool, whether to plot residuals
    
    Returns:
    - p-value: float, the p-value of the test
    
    Interpretation:
    - A p-value < 0.05 suggests the presence of heteroskedasticity.
    """
    # Fit a simple linear model
    X = np.arange(len(series))
    X = add_constant(X)
    model = OLS(series, X).fit()
    residuals = model.resid
    
    # Perform Breusch-Pagan test
    _, p_value, _, _ = het_breuschpagan(residuals, X)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(residuals)
        plt.title('Residuals Plot')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.show()
    
    return p_value

def test_stationarity(series, plot=False):
    """
    Tests for stationarity using the Augmented Dickey-Fuller test.
    
    Parameters:
    - series: pd.Series, the time series data to test
    - plot: bool, whether to plot the series and rolling statistics
    
    Returns:
    - p-value: float, the p-value of the test
    
    Interpretation:
    - A p-value < 0.05 suggests the series is stationary.
    """
    # Perform ADF test
    result = adfuller(series)
    p_value = result[1]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(series, label='Original')
        plt.plot(series.rolling(window=12).mean(), label='Rolling Mean')
        plt.plot(series.rolling(window=12).std(), label='Rolling Std')
        plt.title('Time Series and Rolling Statistics')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    return p_value

def differentiate_series(series, order=1):
    """
    Differentiates the time series to achieve stationarity.
    
    Parameters:
    - series: pd.Series, the time series data to differentiate
    - order: int, the order of differentiation (default is 1)
    
    Returns:
    - differentiated_series: pd.Series, the differentiated series
    
    Interpretation:
    - Differentiation helps in stabilizing the mean of a time series by removing changes in the level of a time series, and thus eliminating (or reducing) trend and seasonality.
    """
    differentiated_series = series.diff(periods=order).dropna()
    return differentiated_series

def check_multicollinearity(df, features):
    """
    Checks for multicollinearity between features using Variance Inflation Factor (VIF).
    
    Parameters:
    - df: DataFrame, the data containing the features
    - features: list, the list of feature column names to check
    
    Returns:
    - DataFrame with VIF values for each feature
    """
    # Add a constant to the DataFrame
    X = add_constant(df[features])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X.values, i + 1) for i in range(len(features))]
    
    return vif_data

def add_trend_feature(df, target_var):
    """
    Adds a trend feature to the DataFrame based on the target variable.
    
    Parameters:
    - df: DataFrame, the data containing the target variable
    - target_var: str, the name of the target variable column
    
    Returns:
    - DataFrame with an additional trend feature column
    """
    df['trend'] = np.arange(len(df))
    return df

def add_seasonality_feature(df, target_var, period=12):
    """
    Adds a seasonality feature to the DataFrame based on the target variable.
    
    Parameters:
    - df: DataFrame, the data containing the target variable
    - target_var: str, the name of the target variable column
    - period: int, the period of seasonality (default is 12 for monthly data)
    
    Returns:
    - DataFrame with an additional seasonality feature column
    """
    df['seasonality'] = np.sin(2 * np.pi * df.index / period)
    return df
