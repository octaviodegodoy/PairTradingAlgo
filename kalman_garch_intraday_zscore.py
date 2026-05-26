import logging
import time as _time
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from kalman_filter import KalmanFilter, estimate_initial_hedge_ratio


GROUP_Y = "WIN*"  # dependent futures group
GROUP_X = "WDO*"  # independent futures group
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


def _get_front_month(group: str) -> str:
    """Return the nearest-expiry active futures contract name for *group* (e.g. 'WIN*')."""
    symbols = mt5.symbols_get(group)
    if not symbols:
        raise RuntimeError(f"No symbols found for {group}")
    now = int(_time.time())
    candidates = {
        s.expiration_time: s.name
        for s in symbols
        if len(s.name) == 6 and s.expiration_time > now
    }
    if not candidates:
        raise RuntimeError(f"No active futures contract found for {group}")
    return candidates[min(candidates)]


def _get_rates_with_history(group: str, timeframe: int, bars: int) -> pd.DataFrame:
    """Fetch bars by concatenating the continuous alias (WIN$/WDO$) with the current
    front-month contract. The continuous series supplies historical depth; the active
    contract supplies the most recent prices. Front-month bars take priority on overlaps.
    """
    continuous_sym = group.replace("*", "$")
    front_month = _get_front_month(group)
    print(f"Front-month for {group}: {front_month}")

    # Continuous symbol — fetch 2× bars for historical depth
    df_hist = _get_rates(continuous_sym, timeframe, bars * 2) if True else pd.DataFrame()

    # Current front-month — recent bars
    df_cur = _get_rates(front_month, timeframe, bars)

    # Continuous first so front-month rows (appended last) win on drop_duplicates
    frames = []
    if not df_hist.empty:
        frames.append(df_hist)
    if not df_cur.empty:
        frames.append(df_cur)

    if not frames:
        raise RuntimeError(f"No rates returned for {group} (tried {continuous_sym} and {front_month})")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined.tail(bars).reset_index(drop=True)


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
        data_y = _get_rates_with_history(GROUP_Y, TIMEFRAME, BARS)
        data_x = _get_rates_with_history(GROUP_X, TIMEFRAME, BARS)

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

        logger.info("Symbol Group Y: %s, Symbol Group X: %s", GROUP_Y, GROUP_X)
        logger.info("Latest time: %s", latest_time)
        logger.info("Latest Kalman beta: %.6f", latest_beta)
        logger.info("Latest GARCH z-score: %.4f", latest_z)

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
