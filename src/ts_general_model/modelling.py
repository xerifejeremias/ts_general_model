import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from linearmodels.panel import PanelOLS
from statsmodels.api import OLS, add_constant

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

def fit_panel_data_model(df):
    """
    Fits a panel data model (Fixed Effects) for all market and product_code combinations.
    
    Parameters:
    - df: DataFrame, the data to model
    
    Returns:
    - Fitted panel data model
    """
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
    models = {}
    for (market, product_code), group in df.groupby(['market', 'product_code']):
        X = group[['var1', 'var2', 'var3', 'var4', 'var5']]
        X = add_constant(X)  # Adds a constant term to the predictor
        y = group['y']
        model = OLS(y, X).fit()
        models[(market, product_code)] = model
        print(f"Fitted OLS model for market: {market}, product_code: {product_code}")
    
    return models
