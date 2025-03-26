import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple, List



from statsmodels.tsa.statespace.sarimax import SARIMAX
from linearmodels.panel import PanelOLS
from statsmodels.api import OLS, add_constant


def fit_sarima_model(df: pd.DataFrame, target_var: str = 'y', order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)) -> Any:
    """
    Fits a SARIMA model for the entire dataframe using the specified target variable.

    Parameters:
    - df: pd.DataFrame, the data to model.
    - target_var: str, the name of the target variable.
    - order: tuple, the (p, d, q) order of the model.
    - seasonal_order: tuple, the (P, D, Q, s) seasonal order of the model.

    Returns:
    - The fitted SARIMA model.
    """
    model = SARIMAX(df[target_var], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    print(f"Fitted SARIMA model for target: {target_var}")
    return results


def fit_panel_data_model(df: pd.DataFrame, index_columns: List[str] = ['market', 'product_code', 'ds'], dependent_var: str = 'y', independent_vars: List[str] = ['var1', 'var2', 'var3', 'var4', 'var5']) -> Any:
    """
    Fits a panel data model (Fixed Effects) for all market and product_code combinations.

    Parameters:
    - df: pd.DataFrame, the data to model.
    - index_columns: list of str, columns to set as index.
    - dependent_var: str, the name of the dependent variable.
    - independent_vars: list of str, the names of the independent variables.

    Returns:
    - Fitted panel data model.
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

def fit_ols_model(df: pd.DataFrame, dependent_var: str = 'y', independent_vars: List[str] = ['var1', 'var2', 'var3', 'var4', 'var5']) -> Any:
    """
    Fits an OLS model for the entire dataframe using the specified dependent and independent variables.

    Parameters:
    - df: pd.DataFrame, the data to model.
    - dependent_var: str, the name of the dependent variable.
    - independent_vars: list of str, the names of the independent variables.

    Returns:
    - The fitted OLS model.
    """
    X = df[independent_vars]
    X = add_constant(X)
    y = df[dependent_var]
    model = OLS(y, X).fit()
    print("Fitted OLS model")
    print(model.summary())
    return model

def train_test_split_by_date(df: pd.DataFrame, date_str: str, date_column: str = 'ds') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into train and test sets based on a specific date.

    Parameters:
    - df: pd.DataFrame, the data to split.
    - date_str: str, the date to split the data on in "YYYY-MM-DD" format.
    - date_column: str, the name of the column containing datetime information.

    Returns:
    - Tuple containing:
      - train_df: pd.DataFrame, the training set.
      - test_df: pd.DataFrame, the testing set.
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

