"""
predictive_tool.py
------------------
LangChain tool for predictive analytics with automatic model selection.
All numeric calculations are delegated to specialized libraries
to avoid LLM hallucination on numeric data.

Supported models:
- Linear Regression      (sklearn)
- Polynomial Regression  (numpy/sklearn)
- Moving Average         (numpy)
- Exponential Smoothing  (statsmodels)
- ARIMA                  (statsmodels)
- SARIMA                 (statsmodels) — seasonal ARIMA
- Random Forest          (sklearn)
- XGBoost                (xgboost)
- Prophet                (prophet)
- Ensemble               (average of linear + exponential_smoothing + arima)

Model selection is automatic based on data size and trend pattern.
Confidence intervals are returned for all models where available.
"""

import json
import warnings
from typing import Union

import numpy as np
import pandas as pd
from langchain.tools import tool
from database.db_connector import DatabaseConnector

warnings.filterwarnings("ignore")


def detect_best_model(df: pd.DataFrame, value_col: str) -> str:
    """
    Automatically selects the best forecasting model based on data characteristics.

    Rules:
    - < 10 points       → moving_average (not enough data for ML)
    - 10-15 points      → linear or polynomial depending on trend linearity
    - 16-30 points      → arima or polynomial (classic time series range)
    - 30-100 points     → sarima (if seasonal), exponential_smoothing, or random_forest
    - > 100 points      → xgboost or prophet
    - strong seasonality (autocorr > 0.6 at lag 12) → sarima or prophet
    """
    n = len(df)
    values = df[value_col].values

    if n < 10:
        return "moving_average"

    # Check linearity using R² of linear fit
    x = np.arange(n)
    coeffs = np.polyfit(x, values, 1)
    linear_pred = np.polyval(coeffs, x)
    ss_res = np.sum((values - linear_pred) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r2_linear = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    if n < 16:
        return "linear" if r2_linear > 0.85 else "polynomial"

    # Check for seasonality (autocorrelation at lag 12 = monthly seasonality)
    has_seasonality = False
    if n >= 24:
        try:
            autocorr = pd.Series(values).autocorr(lag=12)
            has_seasonality = autocorr > 0.5
        except Exception:
            pass

    if n < 30:
        # Good range for ARIMA: enough points, not too many
        return "sarima" if has_seasonality else "arima"

    if n < 100:
        if has_seasonality:
            return "sarima" if n >= 36 else "prophet"
        return "exponential_smoothing" if r2_linear > 0.7 else "random_forest"

    return "prophet" if has_seasonality else "xgboost"


def forecast_linear(values: np.ndarray, periods: int) -> list:
    """Linear regression forecast."""
    from sklearn.linear_model import LinearRegression
    x = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, values)
    future_x = np.arange(len(values), len(values) + periods).reshape(-1, 1)
    return model.predict(future_x).tolist()


def forecast_polynomial(values: np.ndarray, periods: int, degree: int = 2) -> list:
    """Polynomial regression forecast."""
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, degree)
    future_x = np.arange(len(values), len(values) + periods)
    return np.polyval(coeffs, future_x).tolist()


def forecast_moving_average(values: np.ndarray, periods: int) -> list:
    """Moving average forecast."""
    window = min(3, len(values))
    forecast = []
    recent = list(values[-window:])
    for _ in range(periods):
        next_val = float(np.mean(recent))
        forecast.append(next_val)
        recent = recent[1:] + [next_val]
    return forecast


def forecast_exponential_smoothing(values: np.ndarray, periods: int) -> list:
    """Exponential smoothing forecast using statsmodels."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(values, trend="add", seasonal=None)
    fit = model.fit()
    return fit.forecast(periods).tolist()


def forecast_arima(values: np.ndarray, periods: int) -> tuple[list, list, list]:
    """
    ARIMA forecast with automatic order selection via AIC minimization.
    Returns (forecast, lower_ci, upper_ci).
    Tries orders p in [0,1,2], d in [0,1], q in [0,1,2] and picks lowest AIC.
    Falls back to ARIMA(1,1,1) if auto-selection fails.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import itertools

    best_aic = np.inf
    best_order = (1, 1, 1)

    # Grid search over small ARIMA orders
    for p, d, q in itertools.product(range(3), range(2), range(3)):
        try:
            m = ARIMA(values, order=(p, d, q))
            fit = m.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order = (p, d, q)
        except Exception:
            continue

    try:
        model = ARIMA(values, order=best_order)
        fit = model.fit()
        forecast_result = fit.get_forecast(steps=periods)
        mean = forecast_result.predicted_mean.tolist()
        ci = forecast_result.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].tolist()
        upper = ci.iloc[:, 1].tolist()
        return mean, lower, upper
    except Exception:
        # Fallback: no confidence intervals
        forecast = forecast_linear(values, periods)
        return forecast, [], []


def forecast_sarima(values: np.ndarray, periods: int, seasonal_period: int = 12) -> tuple[list, list, list]:
    """
    Seasonal ARIMA (SARIMA) forecast for data with monthly seasonality.
    Uses order (1,1,1) x (1,1,0,12) as a robust default.
    Returns (forecast, lower_ci, upper_ci).
    Falls back to ARIMA if SARIMA fails (e.g. not enough seasonal cycles).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        model = SARIMAX(
            values,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        forecast_result = fit.get_forecast(steps=periods)
        mean = forecast_result.predicted_mean.tolist()
        ci = forecast_result.conf_int(alpha=0.05)
        lower = ci.iloc[:, 0].tolist()
        upper = ci.iloc[:, 1].tolist()
        return mean, lower, upper
    except Exception:
        # Fall back to standard ARIMA
        return forecast_arima(values, periods)


def forecast_ensemble(values: np.ndarray, periods: int) -> tuple[list, list, list]:
    """
    Ensemble forecast: averages predictions from linear regression,
    exponential smoothing, and ARIMA. Confidence interval is derived
    from the spread (min/max) across the three models.
    """
    forecasts = []

    try:
        forecasts.append(forecast_linear(values, periods))
    except Exception:
        pass

    try:
        forecasts.append(forecast_exponential_smoothing(values, periods))
    except Exception:
        pass

    try:
        arima_mean, _, _ = forecast_arima(values, periods)
        forecasts.append(arima_mean)
    except Exception:
        pass

    if not forecasts:
        return forecast_linear(values, periods), [], []

    arr = np.array(forecasts)
    mean = arr.mean(axis=0).tolist()
    lower = arr.min(axis=0).tolist()
    upper = arr.max(axis=0).tolist()
    return mean, lower, upper


def forecast_random_forest(values: np.ndarray, periods: int, lags: int = 3) -> list:
    """Random Forest forecast using lagged features."""
    from sklearn.ensemble import RandomForestRegressor

    lags = min(lags, len(values) - 1)
    X, y = [], []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i])
        y.append(values[i])

    if len(X) < 2:
        return forecast_linear(values, periods)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    forecast = []
    recent = list(values[-lags:])
    for _ in range(periods):
        pred = model.predict([recent])[0]
        forecast.append(float(pred))
        recent = recent[1:] + [pred]
    return forecast


def forecast_xgboost(values: np.ndarray, periods: int, lags: int = 5) -> list:
    """XGBoost forecast using lagged features."""
    import xgboost as xgb

    lags = min(lags, len(values) - 1)
    X, y = [], []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i])
        y.append(values[i])

    if len(X) < 2:
        return forecast_linear(values, periods)

    model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(np.array(X), np.array(y))

    forecast = []
    recent = list(values[-lags:])
    for _ in range(periods):
        pred = float(model.predict(np.array([recent]))[0])
        forecast.append(pred)
        recent = recent[1:] + [pred]
    return forecast


def forecast_prophet(df: pd.DataFrame, date_col: str, value_col: str, periods: int) -> list:
    """Prophet forecast for time series with seasonality."""
    from prophet import Prophet

    prophet_df = df[[date_col, value_col]].copy()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)
    return forecast["yhat"].tail(periods).tolist()


def get_predictive_tool(connector: DatabaseConnector):
    """Returns the predictive analytics tool for the LangChain agent."""

    @tool
    def predictive_tool(input_json: Union[str, dict]) -> str:
        """
        Use this tool whenever the user asks about future values, trends, forecasts,
        predictions, projections, or how a metric will develop over time.
        Do NOT answer such questions with SQL alone — always call this tool.

        This tool fetches historical data via SQL and applies an automatically
        selected statistical or machine learning model (linear regression,
        polynomial regression, exponential smoothing, random forest, XGBoost,
        or Prophet) to forecast future values.

        Input: JSON string OR dict with the following fields:
            - sql_query    : SQL query that returns historical time series data
                             (a date/time column + a numeric value column),
                             aggregated by time period (daily, monthly, yearly).
                             Provide at least 12 data points for reliable results.
            - date_column  : exact column name for the date/time in the SQL result
            - value_column : exact column name for the numeric value to forecast
            - periods      : number of future periods to predict (int, typically 3-12)
            - method       : "auto" (recommended) or one of: "linear", "polynomial",
                             "moving_average", "exponential_smoothing",
                             "arima", "sarima", "ensemble",
                             "random_forest", "xgboost", "prophet"

        Output: JSON with model_used, historical_points, forecast_periods,
                historical data, forecast values, and confidence intervals
                (confidence_lower / confidence_upper) where available.

        Example — forecast monthly revenue for the next 6 months:
        {
            "sql_query": "SELECT DATE_TRUNC('month', order_date) AS month, SUM(freight) AS total FROM orders GROUP BY DATE_TRUNC('month', order_date) ORDER BY month",
            "date_column": "month",
            "value_column": "total",
            "periods": 6,
            "method": "auto"
        }
        """
        # GPT-4o passes a native dict via function-calling — accept both forms
        try:
            if isinstance(input_json, dict):
                params = input_json
            else:
                params = json.loads(input_json)
        except json.JSONDecodeError as e:
            return f"Invalid JSON input: {e}"

        required = ["sql_query", "date_column", "value_column", "periods"]
        missing = [f for f in required if f not in params]
        if missing:
            return f"Missing required fields: {missing}"

        try:
            results = connector.execute_query(params["sql_query"])
            if not results:
                return "No data returned by SQL query."

            df = pd.DataFrame(results)
            date_col = params["date_column"]
            value_col = params["value_column"]
            periods = int(params["periods"])
            method = params.get("method", "auto")

            if date_col not in df.columns or value_col not in df.columns:
                return f"Column not found. Available columns: {list(df.columns)}"

            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df = df.dropna(subset=[value_col])

            if len(df) < 2:
                return "Not enough data points for forecasting (minimum 2 required)."

            values = df[value_col].values

            # Auto model selection
            if method == "auto":
                method = detect_best_model(df, value_col)

            # Run selected model
            # Models that return (forecast, lower_ci, upper_ci)
            confidence_lower: list = []
            confidence_upper: list = []

            if method == "linear":
                forecast = forecast_linear(values, periods)
            elif method == "polynomial":
                forecast = forecast_polynomial(values, periods)
            elif method == "moving_average":
                forecast = forecast_moving_average(values, periods)
            elif method == "exponential_smoothing":
                forecast = forecast_exponential_smoothing(values, periods)
            elif method == "arima":
                forecast, confidence_lower, confidence_upper = forecast_arima(values, periods)
            elif method == "sarima":
                forecast, confidence_lower, confidence_upper = forecast_sarima(values, periods)
            elif method == "ensemble":
                forecast, confidence_lower, confidence_upper = forecast_ensemble(values, periods)
            elif method == "random_forest":
                forecast = forecast_random_forest(values, periods)
            elif method == "xgboost":
                forecast = forecast_xgboost(values, periods)
            elif method == "prophet":
                forecast = forecast_prophet(df, date_col, value_col, periods)
            else:
                return (
                    f"Unknown method '{method}'. "
                    f"Use: auto, linear, polynomial, moving_average, "
                    f"exponential_smoothing, arima, sarima, ensemble, "
                    f"random_forest, xgboost, prophet"
                )

            # Build response
            historical = df[[date_col, value_col]].copy()
            historical[date_col] = historical[date_col].astype(str)
            historical[value_col] = historical[value_col].round(2)

            response = {
                "model_used": method,
                "historical_points": len(df),
                "forecast_periods": periods,
                "historical": historical.to_dict(orient="records"),
                "forecast": [round(v, 2) for v in forecast],
            }

            # Add confidence intervals if available
            if confidence_lower and confidence_upper:
                response["confidence_lower"] = [round(v, 2) for v in confidence_lower]
                response["confidence_upper"] = [round(v, 2) for v in confidence_upper]
                response["confidence_level"] = "95%"

            return json.dumps(response, ensure_ascii=False, indent=2)

        except Exception as e:
            return f"Forecasting error: {e}"

    return predictive_tool