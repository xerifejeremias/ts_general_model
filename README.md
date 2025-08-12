import pandas as pd
from abc import ABC, abstractmethod
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Theta
from orbit.models import DLT
from orbit.estimators import StanEstimatorMAP
import warnings

warnings.filterwarnings("ignore")

# Abstract base class for forecasters
class TimeSeriesForecaster(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        pass

# Adapter for MLForecast (LightGBM and XGBoost)
class MLForecasterAdapter(TimeSeriesForecaster):
    def __init__(self):
        self.mlf = MLForecast(
            models={
                'LightGBM': LGBMRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
            },
            freq='M',
            lags=[1, 2, 3, 6, 12],
            date_features=['year', 'month'],
            num_threads=1
        )

    def fit(self, df: pd.DataFrame):
        self.mlf.fit(df, id_col='unique_id', time_col='ds', target_col='y')

    def predict(self, horizon: int) -> pd.DataFrame:
        return self.mlf.predict(horizon)

# Adapter for StatsForecast (AutoARIMA and Theta)
class StatsForecasterAdapter(TimeSeriesForecaster):
    def __init__(self):
        self.models = [
            AutoARIMA(season_length=12),
            Theta(season_length=12)
        ]
        self.freq = 'M'
        self.sf = None

    def fit(self, df: pd.DataFrame):
        self.sf = StatsForecast(df=df, models=self.models, freq=self.freq, n_jobs=1)

    def predict(self, horizon: int) -> pd.DataFrame:
        fcst = self.sf.forecast(h=horizon)
        fcst = fcst.rename(columns={'AutoARIMA': 'AutoArima', 'Theta': 'Theta'})
        return fcst.reset_index()

# Adapter for Orbit (using DLT model for each series)
class OrbitAdapter(TimeSeriesForecaster):
    def __init__(self):
        self.models = {}
        self.last_ds = {}

    def fit(self, df: pd.DataFrame):
        self.df = df
        unique_ids = df['unique_id'].unique()
        for uid in unique_ids:
            df_id = df[df['unique_id'] == uid][['ds', 'y']].sort_values('ds').reset_index(drop=True)
            model = DLT(
                response_col='y',
                date_col='ds',
                seasonality=12,
                estimator_type=StanEstimatorMAP,
                n_bootstrap_draws=-1  # MAP estimation
            )
            model.fit(df_id)
            self.models[uid] = model
            self.last_ds[uid] = df_id['ds'].max()

    def predict(self, horizon: int) -> pd.DataFrame:
        fcsts = []
        for uid, model in self.models.items():
            last_date = self.last_ds[uid]
            future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=horizon, freq='M')
            future_df = pd.DataFrame({'ds': future_dates})
            pred = model.predict(future_df)
            pred['unique_id'] = uid
            pred = pred[['unique_id', 'ds', 'prediction']]
            pred.rename(columns={'prediction': 'Orbit'}, inplace=True)
            fcsts.append(pred)
        return pd.concat(fcsts, ignore_index=True)

# Function to generate combined forecasts
def generate_forecasts(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    # Initialize adapters
    ml_adapter = MLForecasterAdapter()
    stats_adapter = StatsForecasterAdapter()
    orbit_adapter = OrbitAdapter()

    # Fit models
    ml_adapter.fit(df)
    stats_adapter.fit(df)
    orbit_adapter.fit(df)

    # Generate predictions
    ml_fcst = ml_adapter.predict(horizon)
    stats_fcst = stats_adapter.predict(horizon)
    orbit_fcst = orbit_adapter.predict(horizon)

    # Merge forecasts on unique_id and ds
    combined_fcst = ml_fcst.merge(stats_fcst, on=['unique_id', 'ds'], how='outer')
    combined_fcst = combined_fcst.merge(orbit_fcst, on=['unique_id', 'ds'], how='outer')

    # Reorder columns if needed
    columns = ['unique_id', 'ds', 'LightGBM', 'XGBoost', 'AutoArima', 'Theta', 'Orbit']
    combined_fcst = combined_fcst[columns]

    return combined_fcst

# Example usage (assuming df is your DataFrame)
# forecasts = generate_forecasts(df)
# print(forecasts)