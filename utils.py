from datetime import datetime, timezone, timedelta
from constants import ADDITIONAL_GRID, FIBO_VOLUME_FACTORS, NOISE_VARIANCE, START_TIME_HOUR,START_TIME_MINUTE,TRADE_WINDOW_TIME_HOURS,TRADE_WINDOW_TIME_MINUTES, ROLLING_PERIODS, PERIODS, MARGIN_Y, MARGIN_X, VOLUME_FACTOR, Z_SCORE_ENTRY_THRESHOLD, WAVELET_LEVEL
from kalman_filter import KalmanFilter, estimate_initial_hedge_ratio
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pywt
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

def get_linear_regression_spread_zscores(asset1_prices, asset2_prices):
     
    # Log-transform the prices
    log_asset1 = np.log(asset1_prices['close'])
    log_asset2 = np.log(asset2_prices['close'])

    # Get minimum length and trim both series to equal length
    min_length = min(len(log_asset1), len(log_asset2))
    log_asset1 = log_asset1.iloc[:min_length]
    log_asset2 = log_asset2.iloc[:min_length]

    X = log_asset2.values.reshape(-1, 1)
    y = log_asset1.values

    model = LinearRegression()
    model.fit(X, y)

    hedge_ratio = model.coef_[0]
    print(f"Olha o hedge ratio aqui -> {hedge_ratio}")

    # Step 2: Compute fitted values and residuals
    fitted = model.predict(X)
    residuals = y - fitted

    # Step 3: Compute rolling z-score of residuals (window=60)
    rolling_mean = pd.Series(residuals).rolling(window=ROLLING_PERIODS).mean()
    rolling_std = pd.Series(residuals).rolling(window=ROLLING_PERIODS).std()
    z_scores = (pd.Series(residuals) - rolling_mean) / rolling_std

    results = pd.DataFrame({
           'z_scores': z_scores,
           'spread': residuals,
           'hedge_ratio': hedge_ratio,
    })
    
    return results

def wavelet_denoise_spread(spread: np.ndarray, wavelet: str = 'db4', level: int = WAVELET_LEVEL) -> np.ndarray:
    """
    Denoise spread using Discrete Wavelet Transform (DWT).
    Zeroes out detail (high-frequency / noise) coefficients at all levels,
    keeping only the approximation (low-frequency / signal).
    Uses 'periodization' mode to preserve the original array length.
    """
    max_level = pywt.dwt_max_level(len(spread), wavelet)
    level = min(level, max_level)
    coeffs = pywt.wavedec(spread, wavelet, mode='periodization', level=level)
    # Zero all detail levels — keep only the approximation
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet, mode='periodization')
    return denoised[:len(spread)]

def get_dynamic_spread_zscores(asset1_prices, asset2_prices):
    # Log-transform the prices
    log_asset1 = np.log(asset1_prices['close'])
    log_asset2 = np.log(asset2_prices['close'])

    # Get minimum length and trim both series to equal length
    min_length = min(len(log_asset1), len(log_asset2))
    log_asset1 = log_asset1.iloc[:min_length]
    log_asset2 = log_asset2.iloc[:min_length]

    y = log_asset1.values
    x = log_asset2.values

    initial_beta = estimate_initial_hedge_ratio(y, x, lookback=PERIODS)
   
    # Initialize Kalman Filter
    kf = KalmanFilter(
            delta=1e-4,
            ve=1e-3,
            initial_state=initial_beta,
            initial_variance=1.0
        )

    # Run Kalman Filter to get dynamic hedge ratios
    filter_results = kf.filter_batch(y, x)

    # Extract spread and denoise with wavelet transform
    spread = filter_results['spread'].values
    spread = wavelet_denoise_spread(spread)

    # Compute rolling statistics for z-score
    spread_series = pd.Series(spread)
    rolling_mean = pd.Series(spread_series).rolling(window=ROLLING_PERIODS).mean()
    rolling_std = pd.Series(spread_series).rolling(window=ROLLING_PERIODS).std()
    z_scores = (pd.Series(spread_series) - rolling_mean) / rolling_std

    results = pd.DataFrame({
           'z_scores': z_scores,
           'spread': spread,
           'hedge_ratio': filter_results['kalman_hedge_ratio'].values[-1],
    })

    return results

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

def get_hurst_exponent(spread: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate Hurst exponent via variance-of-lags method.
    H < 0.5  -> mean-reverting spread (good for pairs trading)
    H ~ 0.5  -> random walk
    H > 0.5  -> trending / persistent
    """
    spread = np.asarray(spread, dtype=float)
    max_lag = min(max_lag, len(spread) // 2)
    lags = range(2, max_lag)
    tau = [np.std(spread[lag:] - spread[:-lag]) for lag in lags]
    # Remove zero-std lags to avoid log(0)
    valid = [(lag, t) for lag, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return 0.5  # not enough data, return neutral value
    log_lags = np.log([v[0] for v in valid])
    log_tau  = np.log([v[1] for v in valid])
    poly = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0])

def check_cointegration(spreads):
    # Perform Augmented Dickey-Fuller test on the spread
    print("Checking cointegration using ADF test")
    adf_result = adfuller(spreads)
    p_value = adf_result[1]
    coint_t = adf_result[0]
    critical_value = adf_result[4]['10%']
    t_check = coint_t < critical_value
    coint_flag = p_value < 0.15 and t_check
    print(f"ADF Test Result: p-value={p_value}, coint_t={coint_t}, critical_value(10%)={critical_value}, Cointegration Flag={coint_flag}")
    return coint_flag

def calculate_volumes(symbolY,symbolX,hedge_ratio,min_lot_Y,min_lot_X,total_max_lots,total_positions):
    print(f"Volume adjust for {symbolY} and {symbolX} with hedge ratio {hedge_ratio}")

    grid_lot_investment = total_max_lots/VOLUME_FACTOR

    grid_count = (total_positions/2)
    fibo_index = int(grid_count)

    print(f"Grid count {grid_count} and fibo index {fibo_index} max lots {total_max_lots} and grid lot investment {grid_lot_investment}")

    investment_asset_x = (grid_lot_investment/(1 + abs(hedge_ratio)))
    investment_asset_y = (grid_lot_investment - investment_asset_x)

    volume_y = max(investment_asset_y,min_lot_Y)*FIBO_VOLUME_FACTORS[fibo_index]
    volume_x = max(investment_asset_x,min_lot_X)*FIBO_VOLUME_FACTORS[fibo_index]

    volume_y = float(math.floor(volume_y))
    volume_x = float(math.floor(volume_x))
    
    print(f"hedge ratio {hedge_ratio} volume {symbolY} is {volume_y} and for {symbolX} is {volume_x}")
    return volume_y, volume_x

def get_correlation(assetY,assetX):
    # Calculate the Pearson correlation coefficient between the two assets
    print(f"Calculating correlation between {assetY['close'].iloc[0]} and {assetX['close'].iloc[0]}")
    correlation = assetY['close'].corr(assetX['close'])
    return correlation

def get_vecm_ect_zscore(asset1_prices, asset2_prices) -> float:
    """
    Compute the VECM Error Correction Term (ECT) z-score using the Johansen
    cointegrating vector.  Returns the most recent normalized ECT value.
    A high absolute value means prices are far from long-run equilibrium.
    """
    log_y = np.log(asset1_prices['close'])
    log_x = np.log(asset2_prices['close'])
    min_length = min(len(log_y), len(log_x))
    log_y = log_y.iloc[:min_length].values
    log_x = log_x.iloc[:min_length].values

    data = np.column_stack([log_y, log_x])
    johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)

    # First eigenvector is the dominant cointegrating vector
    ev = johansen_result.evec[:, 0]
    beta = ev[0] / ev[1]   # normalise so log_x coefficient = 1

    ect = log_y - beta * log_x
    ect_series = pd.Series(ect)
    rolling_mean = ect_series.rolling(window=ROLLING_PERIODS).mean()
    rolling_std  = ect_series.rolling(window=ROLLING_PERIODS).std()
    ect_zscore = (ect_series - rolling_mean) / rolling_std

    return float(ect_zscore.iloc[-1])

def get_group_name(symbol):
    return symbol[:3]+'*'

def updates_zscore_entry(highest_zscore_period,total_profit,total_traded_volumes,total_grids_history,current_grids):
    logger = logging.getLogger(__name__)
    updated_zscore_entry = 0.0
    grids_total = 0.0
    if current_grids > 0.0 or total_grids_history > 0.0:
        logger.info(f"Total open grids: {current_grids} and total grids history: {total_grids_history}")
        grids_total = max(current_grids, total_grids_history)

    logger.info(f"Total grids history: {total_grids_history}, Total traded volumes: {total_traded_volumes}, Total profit: {total_profit}, Highest zscore period: {highest_zscore_period}, Total grids: {grids_total}")
    if grids_total == 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
        updated_zscore_entry = float(highest_zscore_period) + ADDITIONAL_GRID
    elif grids_total == 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD
    elif grids_total > 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = float(highest_zscore_period) + (ADDITIONAL_GRID * grids_total)
    elif grids_total > 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + (ADDITIONAL_GRID * grids_total)

    logger.info(f"Updated z score is {updated_zscore_entry} for highest z score period {highest_zscore_period} and total grids {grids_total}")

    return updated_zscore_entry 

    