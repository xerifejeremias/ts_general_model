import pandas as pd
import numpy as np
from typing import List, Optional

def add_trend_feature(df: pd.DataFrame, target_var: str) -> pd.DataFrame:
    """
    Adds a trend feature to the DataFrame based on the target variable.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - target_var: str, the name of the target variable (not used in computation).
    
    Returns:
    - pd.DataFrame with an added 'trend' column.
    """
    df = df.copy()
    df['trend'] = np.arange(len(df))
    return df

def add_seasonality_feature(df: pd.DataFrame, target_var: str, period: int = 12) -> pd.DataFrame:
    """
    Adds a seasonality feature to the DataFrame based on the target variable.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - target_var: str, the name of the target variable (not used in computation).
    - period: int, the seasonality period.
    
    Returns:
    - pd.DataFrame with an added 'seasonality' column.
    """
    df = df.copy()
    df['seasonality'] = np.sin(2 * np.pi * df.index / period)
    return df

def add_lag_features(df: pd.DataFrame, lags: List[int], target: str) -> pd.DataFrame:
    """
    Creates lagged variables for the specified target variable for each lag in the list.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - lags: List[int], a list of integers specifying the lag periods.
    - target: str, the name of the target variable column.
    
    Returns:
    - pd.DataFrame with new columns in the form "{target}_lag_{lag}".
    """
    df = df.copy()
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df

def add_log_target_feature(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Applies a natural logarithm transformation to the target variable and adds it as a new column.
    
    Parameters:
    - df: pd.DataFrame, the input data.
    - target_column: str, the name of the target variable column.
    
    Returns:
    - pd.DataFrame with an added column "log_{target_column}" containing the log-transformed values.
    """
    df = df.copy()
    df[f"log_{target_column}"] = np.log(df[target_column])
    return df

def add_differentiated_series_feature(df: pd.DataFrame, target_column: str, order: int = 1) -> pd.DataFrame:
    """
    Differentiates a target variable in a DataFrame to achieve stationarity.

    Parameters:
    - df: pd.DataFrame, the data containing the target variable.
    - target_column: str, the name of the column to differentiate.
    - order: int, the order of differentiation (default is 1).

    Returns:
    - pd.DataFrame: the DataFrame with an additional column containing the differentiated series.
      The new column is named as '{target_column}_diff_{order}'.
    """
    series = df[target_column]
    differentiated_series = series.diff(periods=order).dropna()
    new_column = f"{target_column}_diff_{order}"
    # Align the differentiated series with the original DataFrame by reindexing
    df = df.copy()
    df[new_column] = differentiated_series
    return df

def apply_pipeline_feature_engineering(df: pd.DataFrame,
                                       target_var: str,
                                       lags: Optional[List[int]] = None,
                                       seasonality_period: int = 12,
                                       diff_order: int = 1) -> pd.DataFrame:
    
    
    if lags is None:
        lags = [1]

    df = add_trend_feature(df, target_var)
    df = add_seasonality_feature(df, target_var, seasonality_period)
    df = add_lag_features(df, lags, target_var)
    df = add_log_target_feature(df, target_var)
    df = add_differentiated_series_feature(df, target_var, diff_order)

    return df
