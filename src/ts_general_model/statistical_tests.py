import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, List

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS, add_constant

def test_autocorrelation(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Tests for autocorrelation in the target column and plots the ACF graph.

    Parameters:
    - df: pd.DataFrame, the data containing the target variable.
    - target_column: str, the name of the target column to test.

    Returns:
    - pd.DataFrame with autocorrelation values.
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


def test_heteroskedasticity(df: pd.DataFrame, target_column: str, plot: bool = False) -> float:
    """
    Tests for heteroskedasticity using the Breusch-Pagan test.

    Parameters:
    - df: pd.DataFrame, the data containing the target variable.
    - target_column: str, the name of the target variable.
    - plot: bool, whether to plot residuals.

    Returns:
    - float: the p-value of the test.

    Interpretation:
    - A p-value < 0.05 suggests the presence of heteroskedasticity.
    """
    series = df[target_column]
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



def test_stationarity(df: pd.DataFrame, target_column: str, plot: bool = False) -> float:
    """
    Tests for stationarity using the Augmented Dickey-Fuller test.

    Parameters:
    - df: pd.DataFrame, the data containing the target variable.
    - target_column: str, the name of the target variable column on which the ADF test is performed.
    - plot: bool, whether to plot the series and rolling statistics.

    Returns:
    - float: the p-value of the test.

    Interpretation:
    - A p-value < 0.05 suggests the target variable is stationary.
    """
    series = df[target_column]
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


def test_multicollinearity(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Checks for multicollinearity between features using Variance Inflation Factor (VIF).

    Parameters:
    - df: pd.DataFrame, the data containing the features.
    - features: list of str, the list of feature column names to check.

    Returns:
    - pd.DataFrame with VIF values for each feature.
    """
    # Add a constant to the DataFrame
    X = add_constant(df[features])
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X.values, i + 1) for i in range(len(features))]
    
    return vif_data

