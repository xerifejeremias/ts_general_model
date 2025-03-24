import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.
    
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)

def perform_eda(df):
    """
    Performs Exploratory Data Analysis on the DataFrame.
    
    Parameters:
    - df: DataFrame, the data to analyze
    
    Returns:
    - None, displays plots and prints statistics
    """
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Histograms for numerical features
    df[['var1', 'var2', 'var3', 'var4', 'var5', 'y']].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

def create_lags(df, lag=1):
    """
    Creates lagged features for the target variable at market and product_code level.
    
    Parameters:
    - df: DataFrame, the data to process
    - lag: int, the number of periods to lag
    
    Returns:
    - DataFrame with lagged features
    """
    df = df.copy()
    df.sort_values(by=['market', 'product_code', 'ds'], inplace=True)
    df[f'y_lag_{lag}'] = df.groupby(['market', 'product_code'])['y'].shift(lag)
    return df

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
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    models = {}
    for (market, product_code), group in df.groupby(['market', 'product_code']):
        model = SARIMAX(group['y'], order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)
        models[(market, product_code)] = results
        print(f"Fitted SARIMA model for market: {market}, product_code: {product_code}")
    
    return models

def fit_panel_data_model(df):
    """
    Fits a panel data model (Fixed Effects) for all market and product_code combinations.
    
    Parameters:
    - df: DataFrame, the data to model
    
    Returns:
    - Fitted panel data model
    """
    from linearmodels.panel import PanelOLS
    from linearmodels import RandomEffects
    
    # Set the index for panel data
    df = df.set_index(['market', 'product_code', 'ds'])
    
    # Define the dependent and independent variables
    y = df['y']
    X = df[['var1', 'var2', 'var3', 'var4', 'var5']]
    X = add_constant(X)
    
    # Fit the Fixed Effects model
    model = PanelOLS(y, X, entity_effects=True)
    results = model.fit()
    
    print("Fitted Panel Data Model (Fixed Effects):")
    print(results.summary)
    
    return results

def fit_ols_model(df):
    """
    Fits an OLS model for each market and product_code.
    
    Parameters:
    - df: DataFrame, the data to model
    
    Returns:
    - Dictionary of fitted models
    """
    from statsmodels.api import OLS, add_constant
    
    models = {}
    for (market, product_code), group in df.groupby(['market', 'product_code']):
        X = group[['var1', 'var2', 'var3', 'var4', 'var5']]
        X = add_constant(X)  # Adds a constant term to the predictor
        y = group['y']
        model = OLS(y, X).fit()
        models[(market, product_code)] = model
        print(f"Fitted OLS model for market: {market}, product_code: {product_code}")
    
    return models
    
    # Correlation matrix
    corr_matrix = df[['var1', 'var2', 'var3', 'var4', 'var5', 'y']].corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Scatter plots for each feature against the target variable
    for var in ['var1', 'var2', 'var3', 'var4', 'var5']:
        plt.figure(figsize=(6, 4))
        plt.scatter(df[var], df['y'])
        plt.title(f'Scatter Plot: {var} vs y')
        plt.xlabel(var)
        plt.ylabel('y')
        plt.show()
