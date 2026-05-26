from datetime import datetime, timezone, timedelta
from constants import ADDITIONAL_GRID, FIBO_VOLUME_FACTORS, START_TIME_HOUR,START_TIME_MINUTE,TRADE_WINDOW_TIME_HOURS,TRADE_WINDOW_TIME_MINUTES, ROLLING_PERIODS, PERIODS, MARGIN_Y, MARGIN_X, VOLUME_FACTOR, Z_SCORE_ENTRY_THRESHOLD, WAVELET_LEVEL, COINTEGRATION_METHOD, JOHANSEN_CRIT_LEVEL, JOHANSEN_DET_ORDER, JOHANSEN_MAX_LAGS, ADF_PVALUE_THRESHOLD, ADF_CRIT_LEVEL, EG_PVALUE_THRESHOLD, EG_CRIT_LEVEL, OU_LAMBDA_MIN, KALMAN_ORDER
from kalman_filter import KalmanFilter, estimate_initial_hedge_ratio
# 2nd-order Kalman filter — imported at module level; only instantiated when KALMAN_ORDER == 2
from KalmanPairTrading2ndOrder import KalmanPairTrading2ndOrder
from cubic_spline_interest_rate import RateCurve
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
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

    if KALMAN_ORDER == 2:
        # 2nd-order Kalman: tracks beta velocity and acceleration
        kf = KalmanPairTrading2ndOrder(delta=1e-4, sigma_obs=1e-3, dt=1.0)
        # Seed the initial beta state
        kf.x[1, 0] = initial_beta
    else:
        # Default: 1st-order Kalman filter
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
           'hedge_ratio': filter_results['kalman_hedge_ratio'].values,
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

def get_ou_params(spread) -> dict:
    """
    Fit the Ornstein-Uhlenbeck process to the spread via discrete-time OLS:
        ΔX_t = a + b * X_{t-1} + ε_t
    Returns:
        lambda_ : mean-reversion speed  (λ = -b;  λ > 0 means mean-reverting)
        mu      : long-run mean          (μ = -a/b)
        sigma   : residual std
        is_mean_reverting : True when λ > OU_LAMBDA_MIN
    """
    spread_series = pd.Series(np.asarray(spread, dtype=float))
    spread_lagged = spread_series.shift(1).dropna()
    spread_delta  = (spread_series - spread_series.shift(1)).dropna()

    X = spread_lagged.values.reshape(-1, 1)
    y = spread_delta.values

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    b = float(model.coef_[0])
    a = float(model.intercept_)

    lambda_ = -b
    mu      = (-a / b) if b != 0 else float('nan')
    sigma   = float(np.std(y - model.predict(X)))
    is_mean_reverting = lambda_ > OU_LAMBDA_MIN

    print(
        f"OU params: λ={lambda_:.6f}, μ={mu:.6f}, σ={sigma:.6f}, "
        f"mean_reverting={is_mean_reverting} (min λ={OU_LAMBDA_MIN})"
    )
    return {
        'lambda_': lambda_,
        'mu': mu,
        'sigma': sigma,
        'is_mean_reverting': is_mean_reverting,
    }

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

def check_cointegration(
    asset1_prices,
    asset2_prices,
    method_override=None,
    johansen_crit_level_override=None,
    adf_pvalue_threshold_override=None,
    adf_crit_level_override=None,
    eg_pvalue_threshold_override=None,
    eg_crit_level_override=None,
):
    """
    Check cointegration with configurable method: Johansen, ADF, or Engle-Granger.
    """
    log_y = np.log(asset1_prices['close'])
    log_x = np.log(asset2_prices['close'])
    min_length = min(len(log_y), len(log_x))
    log_y = log_y.iloc[:min_length].values
    log_x = log_x.iloc[:min_length].values

    method = (method_override or COINTEGRATION_METHOD).lower()
    if method == "johansen":
        print("Checking cointegration using Johansen test")
        johansen_crit_level = johansen_crit_level_override or JOHANSEN_CRIT_LEVEL
        cvt_index = {"90%": 0, "95%": 1, "99%": 2}.get(johansen_crit_level, 1)
        data = np.column_stack([log_y, log_x])

        # Select optimal lag order via AIC to ensure VAR residuals are white noise
        best_aic = np.inf
        best_lag = 1
        max_lags = min(JOHANSEN_MAX_LAGS, (len(data) // 10) - 1)
        for lag in range(1, max(2, max_lags + 1)):
            try:
                from statsmodels.tsa.vector_ar.var_model import VAR
                aic = VAR(data).fit(lag, ic=None).aic
                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except Exception:
                break
        print(f"Johansen: selected k_ar_diff={best_lag} via AIC (max_lags={max_lags}), det_order={JOHANSEN_DET_ORDER}")

        johansen_result = coint_johansen(data, det_order=JOHANSEN_DET_ORDER, k_ar_diff=best_lag)

        trace_stat = johansen_result.lr1[0]
        trace_cv  = johansen_result.cvt[0, cvt_index]
        trace_ok  = trace_stat > trace_cv

        max_eigen_stat = johansen_result.lr2[0]
        max_eigen_cv   = johansen_result.cvm[0, cvt_index]
        max_eigen_ok   = max_eigen_stat > max_eigen_cv

        # Use OR: cointegrated if either trace OR max-eigenvalue test passes (standard practice)
        coint_flag = trace_ok or max_eigen_ok
        print(
            "Johansen Trace Result: trace_stat="
            f"{trace_stat}, critical_value({johansen_crit_level})={trace_cv}, trace_ok={trace_ok}"
        )
        print(
            "Johansen Max-Eigen Result: max_eigen_stat="
            f"{max_eigen_stat}, critical_value({johansen_crit_level})={max_eigen_cv}, max_eigen_ok={max_eigen_ok}"
        )
        print(f"Johansen Cointegration Flag (Trace OR Max-Eigen)={coint_flag}")
        return coint_flag

    if method == "engle":
        print("Checking cointegration using Engle-Granger test")
        eg_crit_level = eg_crit_level_override or EG_CRIT_LEVEL
        eg_pvalue_threshold = eg_pvalue_threshold_override or EG_PVALUE_THRESHOLD
        crit_index = {"1%": 0, "5%": 1, "10%": 2}.get(eg_crit_level, 1)
        t_stat, pval, crit = coint(log_y, log_x, trend='c', autolag='AIC')
        t_check = t_stat < crit[crit_index]
        coint_flag = pval < eg_pvalue_threshold and t_check
        print(
            "Engle-Granger Result: t-stat="
            f"{t_stat}, p-value={pval}, critical_value({eg_crit_level})={crit[crit_index]}, "
            f"Cointegration Flag={coint_flag}"
        )
        return coint_flag

    # Default to ADF on OLS residual spread
    print("Checking cointegration using ADF test")
    X = log_x.reshape(-1, 1)
    y = log_y
    model = LinearRegression()
    model.fit(X, y)
    residuals = y - model.predict(X)

    adf_result = adfuller(residuals)
    p_value = adf_result[1]
    coint_t = adf_result[0]
    adf_crit_level = adf_crit_level_override or ADF_CRIT_LEVEL
    adf_pvalue_threshold = adf_pvalue_threshold_override or ADF_PVALUE_THRESHOLD
    critical_value = adf_result[4].get(adf_crit_level, adf_result[4]['5%'])
    t_check = coint_t < critical_value
    coint_flag = p_value < adf_pvalue_threshold and t_check
    print(
        "ADF Test Result: p-value="
        f"{p_value}, coint_t={coint_t}, critical_value({adf_crit_level})={critical_value}, "
        f"Cointegration Flag={coint_flag}"
    )
    return coint_flag

def calculate_volumes(symbolY,symbolX,hedge_ratio,min_lot_Y,min_lot_X,total_max_lots,total_positions):
    print(f"Volume adjust for {symbolY} and {symbolX} with hedge ratio {hedge_ratio}")

    grid_lot_investment = total_max_lots/VOLUME_FACTOR

    grid_count = (total_positions/2)
    fibo_index = int(grid_count)
    # Clamp to the last defined Fibonacci factor to prevent IndexError when
    # grid_count exceeds the length of FIBO_VOLUME_FACTORS.
    fibo_index = min(fibo_index, len(FIBO_VOLUME_FACTORS) - 1)

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

def get_vecm_ect_zscore(asset1_prices, asset2_prices) -> dict:
    """
    Compute the VECM Error Correction Term (ECT) z-score and alpha (adjustment
    coefficients) using the Johansen cointegrating vector.

    Returns a dict with:
      - ect_zscore  : float  — most recent normalized ECT value
      - alpha       : np.ndarray shape (2,) — speed-of-adjustment coefficients
      - alpha_valid : bool   — True when signs indicate valid mean-reversion
                               (alpha[0] < 0 and alpha[1] > 0, or vice versa)
      - beta        : float  — cointegrating beta (log_x coefficient)
    """
    log_y = np.log(asset1_prices['close'])
    log_x = np.log(asset2_prices['close'])
    min_length = min(len(log_y), len(log_x))
    log_y = log_y.iloc[:min_length].values
    log_x = log_x.iloc[:min_length].values

    data = np.column_stack([log_y, log_x])

    # ── Johansen: cointegrating vector (beta) ─────────────────────────────
    johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)
    ev = johansen_result.evec[:, 0]
    beta = ev[0] / ev[1]   # normalise so log_x coefficient = 1

    ect = log_y - beta * log_x
    ect_series = pd.Series(ect)
    rolling_mean = ect_series.rolling(window=ROLLING_PERIODS).mean()
    rolling_std  = ect_series.rolling(window=ROLLING_PERIODS).std()
    ect_zscore = float((ect_series - rolling_mean).div(rolling_std).iloc[-1])

    # ── Full VECM: extract alpha (adjustment / loading coefficients) ──────
    # alpha shape: (k_vars=2, coint_rank=1) — how fast each variable corrects.
    # Correct stability condition (Granger Representation Theorem):
    #   trace(alpha @ beta') < 0  ⟺  dot(alpha, beta_vec) < 0
    # This works for BOTH positively-cointegrated (spread-type, c<0) AND
    # negatively-cointegrated (sum-type, c>0) pairs like WIN/WDO.
    # Do NOT use alpha[0]*alpha[1] < 0 — that only works for spread-type pairs.
    try:
        vecm_model = VECM(data, k_ar_diff=1, coint_rank=1, deterministic='ci')
        vecm_result = vecm_model.fit()
        alpha = vecm_result.alpha.flatten()           # shape (2,) after flatten
        beta_vec = vecm_result.beta[:2, 0]            # first 2 rows = variable coefficients
        # dot(alpha, beta_vec) is the eigenvalue of alpha@beta' (bivariate rank-1 case)
        alpha_valid = bool(np.dot(alpha, beta_vec) < 0)
        logging.getLogger(__name__).info(
            f"VECM alpha: y={alpha[0]:.6f}, x={alpha[1]:.6f}, "
            f"beta_vec={beta_vec}, trace={np.dot(alpha, beta_vec):.6f}, "
            f"valid_mean_reversion={alpha_valid}, beta={beta:.6f}"
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(f"VECM alpha estimation failed: {exc}")
        alpha = np.array([np.nan, np.nan])
        alpha_valid = False

    return {
        "ect_zscore":  ect_zscore,
        "alpha":       alpha,
        "alpha_valid": alpha_valid,
        "beta":        beta,
    }

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

    