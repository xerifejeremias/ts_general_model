import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from linearmodels.panel import PanelOLS
from linearmodels import RandomEffects
from statsmodels.api import OLS, add_constant

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.
    
    Parameters:
    - file_path: str, path to the CSV file
    
    Returns:
    - DataFrame containing the CSV data
    """
    return pd.read_csv(file_path)

def perform_eda(df, numerical_features):
    """
    Performs Exploratory Data Analysis on the DataFrame.
    
    Parameters:
    - df: DataFrame, the data to analyze.
    - numerical_features: list, the column names to use for generating histograms.
    
    Returns:
    - None, displays plots and prints statistics
    """
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Histograms for numerical features
    df[numerical_features].hist(bins=15, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()

def get_dataframe_shape(df):
    """
    Returns the shape and size of a DataFrame.

    Parameters:
    - df: pandas DataFrame.

    Returns:
    - A tuple containing:
      - The shape (rows, columns) of the DataFrame.
      - The total number of elements in the DataFrame.
    """
    return df.shape, df.size

def get_dataframe_summary(df):
    """
    Returns a summary of the DataFrame.

    For numerical columns, descriptive statistics are obtained with describe().
    For categorical columns (object, category), value_counts() is computed for each column.

    Parameters:
    - df: pandas DataFrame.

    Returns:
    - A tuple containing:
      - A DataFrame with descriptive statistics for numerical columns.
      - A dictionary mapping categorical column names to their value counts.
    """
    numerical_summary = df.describe()
    categorical_summary = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        categorical_summary[col] = df[col].value_counts()
    return numerical_summary, categorical_summary
