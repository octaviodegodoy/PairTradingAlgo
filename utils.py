from datetime import datetime, timezone, timedelta
from constants import ADDITIONAL_GRID, FIBO_VOLUME_FACTORS, START_TIME_HOUR,START_TIME_MINUTE,TRADE_WINDOW_TIME_HOURS,TRADE_WINDOW_TIME_MINUTES, ROLLING_PERIODS, PERIODS, MARGIN_Y, MARGIN_X, VOLUME_FACTOR, Z_SCORE_ENTRY_THRESHOLD, WAVELET_LEVEL, COINTEGRATION_METHOD, JOHANSEN_CRIT_LEVEL, JOHANSEN_DET_ORDER, JOHANSEN_MAX_LAGS, ADF_PVALUE_THRESHOLD, ADF_CRIT_LEVEL, EG_PVALUE_THRESHOLD, EG_CRIT_LEVEL, OU_LAMBDA_MIN, KALMAN_ORDER, MAGIC_NUMBER, VOL_SKEW_WINDOW, VOL_SKEW_THRESHOLD
from kalman_filter import KalmanFilter, estimate_initial_hedge_ratio
# 2nd-order Kalman filter — imported at module level; only instantiated when KALMAN_ORDER == 2
from KalmanPairTrading2ndOrder import KalmanPairTrading2ndOrder
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
    # Correlate log returns (stationary) instead of price levels to avoid
    # spurious positive correlation from shared non-stationary trends.
    # WIN and WDO are negatively correlated in returns; using price levels
    # can produce the wrong sign when both series happen to share a drift.
    returns_y = np.log(assetY['close']).diff().dropna()
    returns_x = np.log(assetX['close']).diff().dropna()
    min_len = min(len(returns_y), len(returns_x))
    correlation = returns_y.iloc[:min_len].corr(returns_x.iloc[:min_len])
    print(f"Correlation (log returns) between {assetY['close'].iloc[0]} and {assetX['close'].iloc[0]}: {correlation:.4f}")
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

def updates_zscore_entry(highest_zscore_period, total_profit, total_traded_volumes, total_grids_history, current_grids, magic_number=MAGIC_NUMBER):
    logger = logging.getLogger(__name__)
    updated_zscore_entry = 0.0
    grids_total = 0.0
    if current_grids > 0.0 or total_grids_history > 0.0:
        logger.info(f"[magic={magic_number}] Total open grids: {current_grids} and total grids history: {total_grids_history}")
        grids_total = max(current_grids, total_grids_history)

    logger.info(f"[magic={magic_number}] Total grids history: {total_grids_history}, Total traded volumes: {total_traded_volumes}, Total profit: {total_profit}, Highest zscore period: {highest_zscore_period}, Total grids: {grids_total}")
    if grids_total == 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
        updated_zscore_entry = float(highest_zscore_period) + ADDITIONAL_GRID
    elif grids_total == 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD
    elif grids_total > 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = float(highest_zscore_period) + (ADDITIONAL_GRID * grids_total)
    elif grids_total > 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + (ADDITIONAL_GRID * grids_total)

    logger.info(f"[magic={magic_number}] Updated z score is {updated_zscore_entry} for highest z score period {highest_zscore_period} and total grids {grids_total}")

    return updated_zscore_entry


def get_residual_heteroscedasticity_trend(spread: np.ndarray, window: int = VOL_SKEW_WINDOW) -> dict:
    """
    Detect whether the regression model's residuals (spread) exhibit trending
    heteroscedasticity, and determine the directional bias of that trend.

    A well-specified pair-trading model requires homoscedastic residuals — i.e.
    the spread's variance should be constant over time.  When the variance
    itself starts to trend, the model assumption is violated and the spread is
    in a *directional* regime rather than pure mean-reversion.

    Method
    ------
    Step 1 — Variance trend (is the model heteroscedastic?):
        Compute rolling std of spread changes over ``window`` bars.
        Fit OLS of rolling_std vs time index → normalized slope (variance_slope).
        |variance_slope| > VOL_SKEW_THRESHOLD  →  heteroscedastic (trending variance).

    Step 2 — Directional bias (which way is the heteroscedasticity?):
        Over the trailing ``window`` bars split changes into up/down moves:
            σ_up   = std of positive spread changes
            σ_down = std of negative spread changes
            vol_asymmetry = (σ_up − σ_down) / (σ_up + σ_down)   ∈ [−1, +1]
        A positive asymmetry means upside moves are more volatile → uptrend bias.
        A negative asymmetry means downside moves are more volatile → downtrend bias.

    Step 3 — trend_signal:
        Heteroscedastic AND vol_asymmetry > 0  → +1  (spread trending UP,   BUY bias)
        Heteroscedastic AND vol_asymmetry < 0  → −1  (spread trending DOWN, SELL bias)
        Homoscedastic (flat variance)          →  0  (pure mean-reversion, use z-score)

    Parameters
    ----------
    spread : np.ndarray
        Regression residuals / Kalman spread series.
    window : int
        Rolling window length in bars (default: VOL_SKEW_WINDOW from constants).

    Returns
    -------
    dict with keys:
        variance_slope   : float     — normalized OLS slope of rolling std (>0 rising)
        vol_asymmetry    : float     — conditional variance asymmetry [−1, +1]
        trend_signal     : int       — +1 uptrend, −1 downtrend, 0 homoscedastic
        is_heteroscedastic : bool    — True when residual variance is trending
        rolling_std      : pd.Series — rolling std of spread changes
        sigma_up         : float     — std of positive spread changes in trailing window
        sigma_down       : float     — std of negative spread changes in trailing window
    """
    spread_arr = np.asarray(spread, dtype=float)
    if len(spread_arr) < window + 2:
        return {
            'variance_slope': 0.0,
            'vol_asymmetry': 0.0,
            'trend_signal': 0,
            'is_heteroscedastic': False,
            'rolling_std': pd.Series(dtype=float),
            'sigma_up': float('nan'),
            'sigma_down': float('nan'),
        }

    diff = pd.Series(np.diff(spread_arr))

    # ── Step 1: Variance trend ────────────────────────────────────────────
    rolling_std = diff.rolling(window=window).std().dropna()
    if len(rolling_std) < 3:
        variance_slope = 0.0
    else:
        t = np.arange(len(rolling_std), dtype=float).reshape(-1, 1)
        ols = LinearRegression(fit_intercept=True).fit(t, rolling_std.values)
        mean_std = float(rolling_std.mean())
        # Normalise slope by mean std so the threshold is scale-independent
        variance_slope = float(ols.coef_[0]) / mean_std if mean_std > 0 else 0.0

    is_heteroscedastic = abs(variance_slope) > VOL_SKEW_THRESHOLD

    # ── Step 2: Directional conditional variance ──────────────────────────
    recent_diff = diff.iloc[-window:]
    up_moves   = recent_diff[recent_diff > 0]
    down_moves = recent_diff[recent_diff < 0]

    sigma_up   = float(up_moves.std())   if len(up_moves)   >= 2 else 0.0
    sigma_down = float(down_moves.std()) if len(down_moves) >= 2 else 0.0

    denom = sigma_up + sigma_down
    vol_asymmetry = (sigma_up - sigma_down) / denom if denom > 0 else 0.0

    # ── Step 3: Composite trend signal ───────────────────────────────────
    if is_heteroscedastic and vol_asymmetry > 0:
        trend_signal = 1   # residuals heteroscedastic with upside bias → BUY spread
    elif is_heteroscedastic and vol_asymmetry < 0:
        trend_signal = -1  # residuals heteroscedastic with downside bias → SELL spread
    else:
        trend_signal = 0   # homoscedastic → mean-reverting, z-score drives entry

    return {
        'variance_slope': variance_slope,
        'vol_asymmetry': vol_asymmetry,
        'trend_signal': trend_signal,
        'is_heteroscedastic': is_heteroscedastic,
        'rolling_std': rolling_std,
        'sigma_up': sigma_up,
        'sigma_down': sigma_down,
    }


# Legacy alias kept for backward compatibility
get_volatility_skewness = get_residual_heteroscedasticity_trend


def get_spread_trade_direction(z_score: float, slope: float, threshold: float, trend_signal: int = 0) -> dict:
    """
    Determine whether to go LONG or SHORT the spread, and map that to
    concrete BUY/SELL actions on each leg, incorporating the volatility-skewness
    trend signal.

    Spread direction rules
    ----------------------
    slope > 0 (positively-correlated pair, e.g. WIN/WDO when spread-type):
        z_score < -threshold  →  LONG spread  → BUY Y,  SELL X
        z_score > +threshold  →  SHORT spread → SELL Y, BUY X
    slope < 0 (negatively-correlated pair):
        z_score < -threshold  →  LONG spread  → BUY Y,  BUY X
        z_score > +threshold  →  SHORT spread → SELL Y, SELL X

    Skew alignment with ``trend_signal`` (+1 uptrend, -1 downtrend, 0 neutral):
        LONG  spread + trend_signal == +1  → skew_confirms (spread rising  → reversion from below)
        SHORT spread + trend_signal == -1  → skew_confirms (spread falling → reversion from above)
        LONG  spread + trend_signal == -1  → skew_warns    (spread still falling → hold off)
        SHORT spread + trend_signal == +1  → skew_warns    (spread still rising  → hold off)

    Returns
    -------
    dict with keys:
        spread_direction : str  — "long_spread", "short_spread", or "no_trade"
        action_y         : str  — "BUY", "SELL", or "NONE"
        action_x         : str  — "BUY", "SELL", or "NONE"
        skew_confirms    : bool — vol skew agrees with z-score direction
        skew_warns       : bool — vol skew contradicts z-score direction
        skew_neutral     : bool — vol skew is neutral (trend_signal == 0)
    """
    if z_score < -threshold:
        spread_direction = "long_spread"
        action_y, action_x = ("BUY", "SELL") if slope > 0 else ("BUY", "BUY")
    elif z_score > threshold:
        spread_direction = "short_spread"
        action_y, action_x = ("SELL", "BUY") if slope > 0 else ("SELL", "SELL")
    else:
        return {
            'spread_direction': 'no_trade',
            'action_y': 'NONE',
            'action_x': 'NONE',
            'skew_confirms': False,
            'skew_warns': False,
            'skew_neutral': trend_signal == 0,
        }

    skew_confirms = (
        (spread_direction == "long_spread"  and trend_signal ==  1) or
        (spread_direction == "short_spread" and trend_signal == -1)
    )
    skew_warns = (
        (spread_direction == "long_spread"  and trend_signal == -1) or
        (spread_direction == "short_spread" and trend_signal ==  1)
    )

    return {
        'spread_direction': spread_direction,
        'action_y': action_y,
        'action_x': action_x,
        'skew_confirms': skew_confirms,
        'skew_warns': skew_warns,
        'skew_neutral': trend_signal == 0,
    }