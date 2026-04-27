import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from kalman_filter import KalmanFilter, estimate_initial_hedge_ratio


SYMBOL_Y = "WINJ26"  # dependent
SYMBOL_X = "WDOK26"  # independent
TIMEFRAME = mt5.TIMEFRAME_M5
BARS = 600

KALMAN_DELTA = 1e-4
KALMAN_VE = 1e-3
OLS_LOOKBACK = 60

# If True, fit GARCH on spread returns (diff). If False, fit on spread level.
USE_SPREAD_RETURNS = True
GARCH_SCALE = 100.0
GARCH_FALLBACK_WINDOW = 30


def _get_rates(symbol: str, timeframe: int, bars: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned for {symbol}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def _kalman_spread(log_y: np.ndarray, log_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    initial_beta = estimate_initial_hedge_ratio(log_y, log_x, lookback=OLS_LOOKBACK)
    kf = KalmanFilter(
        delta=KALMAN_DELTA,
        ve=KALMAN_VE,
        initial_state=initial_beta,
        initial_variance=1.0,
    )
    filter_results = kf.filter_batch(log_y, log_x)
    spread = filter_results["spread"].values
    beta = filter_results["kalman_hedge_ratio"].values
    return spread, beta


def _garch_sigma(series: pd.Series) -> Tuple[pd.Series, bool, list]:
    try:
        from arch import arch_model
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: install the 'arch' package to use GARCH(1,1)."
        ) from exc

    scaled = series * GARCH_SCALE
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = model.fit(disp="off")

    warning_texts = [str(w.message) for w in caught]
    converged = getattr(res, "convergence", 0) == 0
    bad_warning = any("Inequality constraints incompatible" in text for text in warning_texts)
    ok = converged and not bad_warning

    sigma = res.conditional_volatility / GARCH_SCALE
    return sigma, ok, warning_texts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MetaTrader 5")

    try:
        data_y = _get_rates(SYMBOL_Y, TIMEFRAME, BARS)
        data_x = _get_rates(SYMBOL_X, TIMEFRAME, BARS)

        min_len = min(len(data_y), len(data_x))
        data_y = data_y.iloc[-min_len:]
        data_x = data_x.iloc[-min_len:]

        log_y = np.log(data_y["close"].values)
        log_x = np.log(data_x["close"].values)

        spread, beta = _kalman_spread(log_y, log_x)
        spread_series = pd.Series(spread, index=data_y["time"].values)

        if USE_SPREAD_RETURNS:
            garch_input = spread_series.diff().dropna()
            spread_aligned = spread_series.loc[garch_input.index]
        else:
            garch_input = spread_series
            spread_aligned = spread_series

        sigma, garch_ok, garch_warnings = _garch_sigma(garch_input)
        sigma = sigma.reindex(spread_aligned.index)

        if (not garch_ok) or sigma.isna().all():
            logger.warning("GARCH fit issues detected; falling back to rolling std.")
            if garch_warnings:
                logger.warning("GARCH warnings: %s", garch_warnings)
            sigma = (
                spread_aligned.rolling(window=GARCH_FALLBACK_WINDOW, min_periods=5)
                .std()
                .replace(0, np.nan)
                .bfill()
                .ffill()
            )

        # Use a simple de-mean before standardizing
        spread_mean = spread_aligned.mean()
        z_score = (spread_aligned - spread_mean) / sigma

        latest_time = spread_aligned.index[-1]
        latest_z = z_score.iloc[-1]
        latest_beta = beta[-1]

        logger.info("Symbol Y: %s, Symbol X: %s", SYMBOL_Y, SYMBOL_X)
        logger.info("Latest time: %s", latest_time)
        logger.info("Latest Kalman beta: %.6f", latest_beta)
        logger.info("Latest GARCH z-score: %.4f", latest_z)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
