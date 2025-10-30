from datetime import datetime, timezone, timedelta
from xml.parsers.expat import model

from matplotlib import dates
from constants import FIBO_VOLUME_FACTORS, NOISE_VARIANCE, START_TIME_HOUR,START_TIME_MINUTE,TRADE_WINDOW_TIME_HOURS,TRADE_WINDOW_TIME_MINUTES, ROLLING_PERIODS, PERIODS, MARGIN_Y, MARGIN_X, VOLUME_FACTOR
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import math
import numpy as np
import logging
import pandas as pd

def check_trading_time():
    now = datetime.now(timezone.utc)
    trading_time = False             
    trading_time_start = now.replace(hour=START_TIME_HOUR, minute=START_TIME_MINUTE, second=0, microsecond=0)
    # If the trading window starts in the future, use yesterday's start time
    if now < trading_time_start:
        trading_time_start -= timedelta(days=1)

    trading_time_end = trading_time_start + timedelta(hours=TRADE_WINDOW_TIME_HOURS, minutes=TRADE_WINDOW_TIME_MINUTES)

    # Handles windows that wrap past midnight
    if trading_time_start <= now < trading_time_end:
        trading_time = True
    else:
        trading_time = False


    return trading_time

# Define measurement function H (dynamic, depends on x_t)
def update_H(x_t):
    return np.array([[x_t, 1]])  # y_t = slope * x_t + intercept + noise

def run_kalman_filter_momentum(y, x):
    kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [slope, intercept], Measurement: y
    # Define state transition matrix (random walk for slope and intercept)
    kf.F = np.array([[1, 0],  # Slope stays constant (random walk)
                [0, 1]]) # Intercept stays constant (random walk)

    # Initial state and covariance
    kf.x = np.array([0.5, 0])  # Initial guess: slope = 0.5, intercept = 0
    kf.P *= 0.01  # Initial uncertainty
    kf.R = NOISE_VARIANCE   # Measurement noise variance
    kf.Q = np.eye(2) * 1e-5  # Process noise (small for slow evolution)

        # Step 3: Run Kalman Filter and compute dynamic spread
    spreads = []
    for t in range(PERIODS):
        kf.H = update_H(x[t])  # Update measurement matrix with current x_t
        kf.predict()
        kf.update(y[t])
        slope, intercept = kf.x
        spread = y[t] - (slope * x[t] + intercept)  # Dynamic spread
        spreads.append(spread)


    return slope, intercept, spreads

def get_residuals_zscore_stdev(asset1_prices, asset2_prices):
    pd.Series(asset1_prices), 
    pd.Series(asset2_prices)

    X = asset1_prices.values.reshape(-1, 1)
    y = asset2_prices.values

    model = LinearRegression()
    model.fit(X, y)

    # Predict log_price2 using the regression model
    log_price2_pred = model.predict(X)

    # Calculate residuals: actual - predicted
    residuals = asset2_prices - log_price2_pred

    return residuals

def get_dynamic_spread_zscores(asset1_prices,asset2_prices):

    dates = asset1_prices['time']

    correlation = asset1_prices['close'].corr(asset2_prices['close'])

    asset1_prices = np.array(asset1_prices['close'])
    asset2_prices = np.array(asset2_prices['close'])

        
    # Log-transform the prices
    log_asset1 = np.log(asset1_prices)
    log_asset2 = np.log(asset2_prices)


    print(f"Counts - Asset 1: {len(log_asset1)}, Asset 2: {len(log_asset2)}")

    # Run the Kalman Filter
    slope, intercept, spreads = run_kalman_filter_momentum(log_asset1, log_asset2)

    spreads = pd.Series(spreads, index=dates)
    slope = pd.Series(slope, index=dates)

    # Step 4: Compute z-scores of the spread
    rolling_mean_spread = spreads.rolling(window=ROLLING_PERIODS, min_periods=1).mean()
    rolling_std_spread = spreads.rolling(window=ROLLING_PERIODS, min_periods=1).std()
    rolling_z_scores = (spreads - rolling_mean_spread) / rolling_std_spread

    return rolling_z_scores, spreads, slope, correlation

def get_linear_regression_spread_zscores(asset1_prices, asset2_prices):

    dates = asset1_prices['time']

    correlation = asset1_prices['close'].corr(asset2_prices['close'])
        
    # Log-transform the prices
    log_asset1 = np.log(asset1_prices['close'])
    log_asset2 = np.log(asset2_prices['close'])

    X = log_asset2.values.reshape(-1, 1)
    y = log_asset1.values

    model = LinearRegression()
    model.fit(X, y)

    hedge_ratio = model.coef_[0]

    # Step 2: Compute fitted values and residuals
    fitted = model.predict(X)
    residuals = y - fitted

    # Step 3: Compute rolling z-score of residuals (window=60)
    rolling_mean = pd.Series(residuals).rolling(window=ROLLING_PERIODS).mean()
    rolling_std = pd.Series(residuals).rolling(window=ROLLING_PERIODS).std()
    z_scores = (pd.Series(residuals) - rolling_mean) / rolling_std

    return z_scores,residuals,hedge_ratio,correlation

def get_half_life(spread):
    # Convert `dynamic_spread` to a pandas Series
    spread_series = pd.Series(spread, name="dynamic_spread")

    # Calculate the lagged spread and the difference
    spread_lagged = spread_series.shift(1).fillna(0)  # Lagged spread (y_{t-1})
    spread_delta = spread_series - spread_lagged  # Change in spread (Delta y_t)

    # Reshape the data for linear regression
    X = spread_lagged.values.reshape(-1, 1)  # Independent variable
    y = spread_delta.values  # Dependent variable

    # Perform linear regression: y = kappa * X + noise
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    kappa = -model.coef_[0]  # Speed of mean reversion (kappa)

    # Calculate the half-life
    half_life = np.log(2) / kappa

    return half_life

def check_cointegration(spreads):
    # Perform Augmented Dickey-Fuller test on the spread
    adf_result = adfuller(spreads)
    p_value = adf_result[1]
    coint_t = adf_result[0]
    critical_value = adf_result[4]['5%']
    t_check = coint_t < critical_value
    coint_flag = p_value < 0.05 and t_check
    return coint_flag

def calculate_volumes(symbolY,symbolX,hedge_ratio,min_lot_Y,min_lot_X,total_max_lots,total_positions):
    print(f"Volume adjust for {symbolY} and {symbolX} with hedge ratio {hedge_ratio}")

    grid_lot_investment = total_max_lots/VOLUME_FACTOR

    grid_count = (total_positions/2)
    fibo_index = int(grid_count)

    print(f"Grid count {grid_count} and fibo index {fibo_index} max lots {total_max_lots} and grid lot investment {grid_lot_investment}")

    investment_asset_y = (grid_lot_investment/(1 + abs(hedge_ratio)))
    investment_asset_x = (grid_lot_investment - investment_asset_y)

    volume_y = max(investment_asset_y,min_lot_Y)*FIBO_VOLUME_FACTORS[fibo_index]
    volume_x = max(investment_asset_x,min_lot_X)*FIBO_VOLUME_FACTORS[fibo_index]

    volume_y = float(math.floor(volume_y))
    volume_x = float(math.floor(volume_x))
    
    print(f"hedge ratio {hedge_ratio} volume {symbolY} is {volume_y} and for {symbolX} is {volume_x}")
    return volume_y, volume_x

def get_correlation(assetY,assetX):
    # Calculate the Pearson correlation coefficient between the two assets
    correlation = assetY['close'].corr(assetX['close'])
    return correlation

def get_group_name(symbol):
    return symbol[:3]+'*'

    