"""
options_analytics.py
--------------------
Utilities for retrieving the most-traded put options on a given underlying
and computing their Implied Volatility (IV) and Black-Scholes delta.

Usage example
-------------
    from mt5_connector import MT5Connector
    from options_analytics import get_puts_iv_delta

    broker = MT5Connector()
    broker.initialize()
    df = get_puts_iv_delta(broker, "PETR4", risk_free_rate=0.1075, n_top=10)
    print(df)
"""

import math
import logging
import time
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def black_scholes_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price of a European put option.

    Parameters
    ----------
    S : float
        Underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years (must be > 0).
    r : float
        Continuously-compounded risk-free rate (annualised).
    sigma : float
        Annualised implied volatility (must be > 0).

    Returns
    -------
    float
        Theoretical put price.
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes delta of a European put option.

    The delta of a long put is always in the range [-1, 0].

    Parameters
    ----------
    S : float
        Underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded risk-free rate (annualised).
    sigma : float
        Annualised implied volatility.

    Returns
    -------
    float
        Put delta in [-1, 0].
    """
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) - 1.0


def compute_implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-6,
) -> float:
    """
    Compute the implied volatility of a European put option using Brent's
    root-finding method.

    Parameters
    ----------
    option_price : float
        Observed market price of the put option.
    S : float
        Underlying spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously-compounded risk-free rate (annualised).
    tol : float
        Solver tolerance (default: 1e-6).

    Returns
    -------
    float
        Implied volatility as a decimal (e.g. 0.25 = 25 %), or ``float('nan')``
        if the solver fails.
    """
    if T <= 0 or option_price <= 0 or S <= 0 or K <= 0:
        return float("nan")

    # Lower bound: intrinsic value (discounted strike minus spot)
    intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    if option_price < intrinsic - tol:
        return float("nan")

    try:
        iv = brentq(
            lambda sigma: black_scholes_put_price(S, K, T, r, sigma) - option_price,
            1e-6,
            10.0,
            xtol=tol,
            maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError) as exc:
        logger.debug("IV solver failed for S=%.4f K=%.4f T=%.4f price=%.4f: %s", S, K, T, option_price, exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Main analytics function
# ---------------------------------------------------------------------------

def get_puts_iv_delta(
    mt5_conn,
    underlying_symbol: str,
    risk_free_rate: float = 0.1075,
    n_top: int = 10,
) -> pd.DataFrame:
    """
    Return the ``n_top`` most-traded put options for the given underlying,
    annotated with their Implied Volatility (IV) and Black-Scholes delta.

    The function queries the broker for all put options on *underlying_symbol*,
    ranks them by traded volume (descending), then for each option:

    * Fetches the latest tick to obtain the last-traded price.
    * Computes IV by inverting the Black-Scholes put-pricing formula.
    * Computes the Black-Scholes put delta.

    Parameters
    ----------
    mt5_conn : BrokerConnector
        An initialised broker connection instance.
    underlying_symbol : str
        Root symbol of the underlying instrument (e.g. ``"PETR4"``).
    risk_free_rate : float
        Annualised continuously-compounded risk-free rate.
        Defaults to the approximate Brazilian SELIC rate (10.75 %).
    n_top : int
        Maximum number of results to return (default: 10).

    Returns
    -------
    pd.DataFrame
        Columns:

        * ``symbol``          – option ticker
        * ``strike``          – strike price
        * ``expiry_days``     – calendar days to expiry
        * ``last_price``      – last-traded option price
        * ``volume``          – traded volume used for ranking
        * ``underlying_price``– underlying mid-price at query time
        * ``iv``              – implied volatility (decimal, e.g. 0.25 = 25 %)
        * ``delta``           – Black-Scholes put delta (range [-1, 0])

        Rows with no valid last price or failed IV are still returned; their
        ``iv`` and ``delta`` columns will contain ``NaN``.

    Raises
    ------
    RuntimeError
        If the underlying tick cannot be fetched.
    """
    # ------------------------------------------------------------------
    # 1. Fetch underlying spot price
    # ------------------------------------------------------------------
    underlying_tick = mt5_conn.get_symbol_tick(underlying_symbol)
    if underlying_tick is None:
        raise RuntimeError(f"Cannot fetch tick for underlying '{underlying_symbol}'")
    S = (underlying_tick.ask + underlying_tick.bid) / 2.0

    # ------------------------------------------------------------------
    # 2. Get all put options for the underlying
    # ------------------------------------------------------------------
    puts = mt5_conn.get_options_puts(underlying_symbol)
    if not puts:
        logger.warning("No put options found for '%s'", underlying_symbol)
        return pd.DataFrame(columns=[
            "symbol", "strike", "expiry_days", "last_price",
            "volume", "underlying_price", "iv", "delta",
        ])

    # ------------------------------------------------------------------
    # 3. Sort by volume descending, take top-N
    # ------------------------------------------------------------------
    puts_sorted = sorted(puts, key=lambda p: getattr(p, "volume_real", 0.0), reverse=True)
    top_puts = puts_sorted[:n_top]

    # ------------------------------------------------------------------
    # 4. Compute IV and delta for each option
    # ------------------------------------------------------------------
    now_ts = int(time.time())
    records = []

    for opt in top_puts:
        symbol = opt.name
        K = float(opt.option_strike)
        expiry_ts = opt.expiration_time
        expiry_days = max(0, (expiry_ts - now_ts) / SECONDS_PER_DAY)
        T = expiry_days / 365.0

        # Last-traded price from tick
        tick = mt5_conn.get_symbol_tick(symbol)
        if tick is not None and tick.last > 0:
            option_price = float(tick.last)
        else:
            option_price = float("nan")

        volume = float(getattr(opt, "volume_real", 0.0))

        if math.isnan(option_price) or option_price <= 0 or K <= 0:
            iv = float("nan")
            delta = float("nan")
        else:
            iv = compute_implied_volatility(option_price, S, K, T, risk_free_rate)
            if math.isnan(iv):
                delta = float("nan")
            else:
                delta = black_scholes_put_delta(S, K, T, risk_free_rate, iv)

        records.append({
            "symbol": symbol,
            "strike": K,
            "expiry_days": round(expiry_days, 1),
            "last_price": option_price,
            "volume": volume,
            "underlying_price": S,
            "iv": iv,
            "delta": delta,
        })

    return pd.DataFrame(records)
