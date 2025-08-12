from orbit.diagnostics.backtest import BackTester
from orbit.models.dlt import DLT
import pandas as pd


def orbit_cv_per_id(
    train_aux: pd.DataFrame,
    regressors=None,
    min_train_len: int = 12,
    forecast_len: int = 12,
    incremental_len: int = 6,
    n_splits: int = 6,
    seasonality: int = 12,
    model_col_name: str = "orbit_dlt",
) -> pd.DataFrame:
    """
    Run Orbit backtests per unique_id and return a Nixtla-style dataframe:
    columns = ['unique_id', 'ds', 'y', 'cutoff', <model_col_name>]

    Expects `train_aux` with at least: ['unique_id', 'ds', 'y'] and optional regressors.
    """

    all_cv_results = []

    # loop over series
    for uid in train_aux["unique_id"].unique():
        print(uid)
        # keep this series; your code also filtered out non-positive y
        df_uid = train_aux[(train_aux["unique_id"] == uid) & (train_aux["y"] > 0.0)].copy()

        # Discard series that are flat (Orbit error when initializing stan-map)
        if df_uid["y"].nunique() == 1:
            continue

        # Skip short series (you had 24 due to seasonality=12)
        if len(df_uid) <= 24:
            continue

        try:
            model = DLT(
                response_col="y",
                date_col="ds",
                regressor_col=regressors if regressors else None,
                seasonality=seasonality,
                estimator="stan-map",
            )

            bt = BackTester(
                model=model,
                df=df_uid,
                min_train_len=min_train_len,
                forecast_len=forecast_len,
                incremental_len=incremental_len,
                n_splits=n_splits,
                window_type="rolling",  # or "expanding"
            )

            bt.fit_predict()
            cv_df = bt.get_predicted_df()

            # -------------- Nixtla-style post-processing --------------
            # cutoff per fold = last training date within that split
            cutoffs = (
                cv_df.loc[cv_df["training_data"], ["split_key", "date"]]
                     .groupby("split_key", as_index=False)["date"].max()
                     .rename(columns={"date": "cutoff"})
            )

            # attach cutoff, keep only test rows, and rename to Nixtla schema
            cv_df = (
                cv_df.merge(cutoffs, on="split_key", how="left")
                     .loc[~cv_df["training_data"], :]
                     .rename(columns={"date": "ds", "actual": "y", "prediction": model_col_name})
            )

            # add the uid and select final columns
            cv_df["unique_id"] = uid
            cv_df = cv_df[["unique_id", "ds", "y", "cutoff", model_col_name]].reset_index(drop=True)
            # ----------------------------------------------------------

            all_cv_results.append(cv_df)

        except Exception as e:
            print(f"Orbit CV failed for {uid}: {e}")

    if not all_cv_results:
        return pd.DataFrame(columns=["unique_id", "ds", "y", "cutoff", model_col_name])

    return pd.concat(all_cv_results, ignore_index=True)