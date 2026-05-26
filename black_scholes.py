"""Black-Scholes option pricing utilities with GARCH(1,1) volatility and earnings calendar.

Public API
----------
bs_price(S, K, T, r, sigma, option_type) -> float
    Compute the Black-Scholes theoretical price for a European option.

bs_delta(S, K, T, r, sigma, option_type) -> float
    Compute the Black-Scholes delta (∂price/∂S) for a European option.

bs_implied_volatility(market_price, S, K, T, r, option_type) -> float
    Invert the Black-Scholes formula to recover implied volatility (IV).

calculate_garch_volatility(ticker, lookback_days) -> float
    Fit GARCH(1,1) on the ticker's historical log-returns and return the
    latest conditional volatility, annualised.

get_next_earnings_dates(tickers, lookahead_days) -> dict[str, datetime | None]
    Return the next earnings announcement date for each ticker within the
    lookahead window, or None if none is found.

compare_garch_iv(garch_vol, iv, threshold) -> bool
    Return True when GARCH vol exceeds IV by at least *threshold*.

analyze_option(ticker, strike, expiry, market_price, option_type,
               risk_free_rate, threshold) -> dict
    Run the full pipeline: fetch GARCH vol, compute IV and delta, compare
    GARCH vs IV, and fetch next earnings date.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from constants import (
    BS_EARNINGS_LOOKAHEAD_DAYS,
    BS_GARCH_IV_THRESHOLD,
    BS_LOOKBACK_DAYS,
    BS_RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)

OptionType = Literal["call", "put"]

# ── Black-Scholes core ────────────────────────────────────────────────────────

def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Return (d1, d2) for the Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        raise ValueError(f"T and sigma must be positive; got T={T}, sigma={sigma}")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Compute the Black-Scholes theoretical price for a European option.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Option strike price.
    T : float
        Time to expiration in **years**.
    r : float
        Continuously compounded risk-free rate (annual).
    sigma : float
        Volatility of the underlying (annual, e.g. 0.20 for 20 %).
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Theoretical option price.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = "call",
) -> float:
    """Compute the Black-Scholes delta for a European option.

    Delta measures the sensitivity of the option price to a unit move in the
    underlying (∂price/∂S).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Same as :func:`bs_price`.
    option_type : {"call", "put"}

    Returns
    -------
    float
        Delta in the range [0, 1] for calls and [-1, 0] for puts.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1.0
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Invert the Black-Scholes formula to recover implied volatility (IV).

    Uses Brent's root-finding method on the interval [1e-6, 20.0].

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S, K, T, r : float
        Same as :func:`bs_price`.
    option_type : {"call", "put"}
    tol : float
        Tolerance passed to :func:`scipy.optimize.brentq`.
    max_iter : int
        Maximum iterations for the root finder.

    Returns
    -------
    float
        Implied volatility (annualised, e.g. 0.25 for 25 %).

    Raises
    ------
    ValueError
        If implied volatility cannot be found within [1e-6, 20.0].
    """
    if market_price <= 0:
        raise ValueError(f"market_price must be positive; got {market_price}")

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    lower, upper = 1e-6, 20.0
    try:
        iv = brentq(objective, lower, upper, xtol=tol, maxiter=max_iter)
    except ValueError as exc:
        raise ValueError(
            f"Could not find IV for market_price={market_price}, S={S}, K={K}, "
            f"T={T:.4f}, r={r}, option_type={option_type!r}. "
            "The market price may be outside the arbitrage bounds."
        ) from exc
    return iv


# ── GARCH(1,1) volatility ─────────────────────────────────────────────────────

def calculate_garch_volatility(
    ticker: str,
    lookback_days: int = BS_LOOKBACK_DAYS,
) -> float:
    """Fit GARCH(1,1) on recent log-returns and return annualised conditional volatility.

    Parameters
    ----------
    ticker : str
        Equity ticker understood by :mod:`yfinance` (e.g. ``"AAPL"``).
    lookback_days : int
        Number of *calendar* days of history to download.  The actual number
        of trading observations will be roughly ``lookback_days * 5/7``.

    Returns
    -------
    float
        Latest GARCH(1,1) conditional volatility, annualised (e.g. 0.25 for
        25 %).

    Raises
    ------
    RuntimeError
        If the 'arch' package is missing, no data is returned, or GARCH
        optimisation does not converge.
    """
    try:
        import yfinance as yf
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install 'yfinance' and 'arch' to use "
            "calculate_garch_volatility()."
        ) from exc

    hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for ticker {ticker!r}")

    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna() * 100.0

    if len(log_returns) < 30:
        raise RuntimeError(
            f"Too few observations ({len(log_returns)}) for ticker {ticker!r} "
            "to fit GARCH(1,1) reliably."
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # rescale=False keeps the model in the original % units so that
        # conditional_volatility can be divided directly by GARCH_SCALE (100)
        # to recover daily vol without an additional rescaling step.
        model = arch_model(log_returns, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
        res = model.fit(disp="off")

    warning_texts = [str(w.message) for w in caught]
    converged = getattr(res, "convergence", 0) == 0
    bad_warning = any("Inequality constraints incompatible" in t for t in warning_texts)

    if not converged or bad_warning:
        logger.warning(
            "GARCH(1,1) for %s did not converge cleanly; warnings: %s",
            ticker, warning_texts,
        )

    # conditional_volatility is in the same units as the input (% daily here)
    # annualise: daily vol → annual vol = daily_vol * sqrt(252)
    latest_daily_vol_pct = res.conditional_volatility.iloc[-1]
    annual_vol = (latest_daily_vol_pct / 100.0) * np.sqrt(252)
    logger.debug("GARCH(1,1) for %s: daily_vol=%.4f%%, annualised=%.4f", ticker, latest_daily_vol_pct, annual_vol)
    return annual_vol


# ── Earnings calendar ─────────────────────────────────────────────────────────

def get_next_earnings_dates(
    tickers: List[str],
    lookahead_days: int = BS_EARNINGS_LOOKAHEAD_DAYS,
) -> Dict[str, Optional[datetime]]:
    """Return the next earnings date for each ticker within *lookahead_days*.

    Parameters
    ----------
    tickers : list[str]
        Equity tickers understood by :mod:`yfinance`.
    lookahead_days : int
        Only return earnings dates within this many days from today.

    Returns
    -------
    dict[str, datetime | None]
        Maps each ticker to its next earnings ``datetime`` (UTC) or ``None``
        if no earnings are scheduled within the window.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install 'yfinance' to use get_next_earnings_dates()."
        ) from exc

    today = datetime.now(tz=timezone.utc).date()
    cutoff = today + timedelta(days=lookahead_days)
    results: Dict[str, Optional[datetime]] = {}

    for ticker in tickers:
        next_date: Optional[datetime] = None
        try:
            info = yf.Ticker(ticker).calendar
            # yfinance returns a dict or a DataFrame depending on the version
            if isinstance(info, dict):
                raw = info.get("Earnings Date")
                if raw:
                    # May be a list or a single value
                    dates = raw if isinstance(raw, list) else [raw]
                else:
                    dates = []
            else:
                # DataFrame with an "Earnings Date" row/column
                try:
                    raw = info.loc["Earnings Date"] if "Earnings Date" in info.index else None
                    dates = list(raw) if raw is not None else []
                except Exception:
                    dates = []

            upcoming = []
            for d in dates:
                # Normalise to date object for comparison
                if isinstance(d, datetime):
                    d_date = d.date()
                    d_dt = d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
                elif isinstance(d, date):
                    d_date = d
                    d_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                else:
                    continue
                if today <= d_date <= cutoff:
                    upcoming.append(d_dt)

            if upcoming:
                next_date = min(upcoming)
                logger.info("Next earnings for %s: %s", ticker, next_date.date())
            else:
                logger.info("No earnings for %s within %d days.", ticker, lookahead_days)

        except Exception as exc:
            logger.warning("Could not fetch earnings calendar for %s: %s", ticker, exc)

        results[ticker] = next_date

    return results


# ── GARCH vs IV comparison ────────────────────────────────────────────────────

def compare_garch_iv(
    garch_vol: float,
    iv: float,
    threshold: float = BS_GARCH_IV_THRESHOLD,
) -> bool:
    """Return True when GARCH volatility exceeds implied volatility by at least *threshold*.

    A positive signal (``True``) suggests that the option market is pricing
    volatility *below* what the GARCH model forecasts — a potential long-
    volatility (buy options) opportunity.

    Parameters
    ----------
    garch_vol : float
        Annualised GARCH(1,1) conditional volatility (e.g. 0.30 for 30 %).
    iv : float
        Annualised implied volatility extracted from the option price.
    threshold : float
        Minimum excess of GARCH vol over IV required to trigger a signal.

    Returns
    -------
    bool
    """
    return (garch_vol - iv) >= threshold


# ── Full pipeline ─────────────────────────────────────────────────────────────

def analyze_option(
    ticker: str,
    strike: float,
    expiry: date,
    market_price: float,
    option_type: OptionType = "call",
    risk_free_rate: float = BS_RISK_FREE_RATE,
    threshold: float = BS_GARCH_IV_THRESHOLD,
) -> dict:
    """Run the full Black-Scholes / GARCH / earnings pipeline for one option.

    Steps
    -----
    1. Fetch the current underlying price via :mod:`yfinance`.
    2. Compute IV and delta from the Black-Scholes model.
    3. Fit GARCH(1,1) on historical log-returns to get annualised vol.
    4. Compare GARCH vol vs IV using *threshold*.
    5. Fetch the next earnings date within :data:`~constants.BS_EARNINGS_LOOKAHEAD_DAYS`.

    Parameters
    ----------
    ticker : str
        Equity ticker (e.g. ``"AAPL"``).
    strike : float
        Option strike price.
    expiry : date
        Option expiration date.
    market_price : float
        Observed market price of the option.
    option_type : {"call", "put"}
    risk_free_rate : float
        Annually compounded risk-free rate.
    threshold : float
        GARCH-vs-IV comparison threshold (see :func:`compare_garch_iv`).

    Returns
    -------
    dict
        Keys: ``ticker``, ``underlying_price``, ``strike``, ``expiry``,
        ``option_type``, ``market_price``, ``implied_volatility``, ``delta``,
        ``garch_volatility``, ``garch_exceeds_iv``, ``next_earnings_date``.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: install 'yfinance' to use analyze_option()."
        ) from exc

    # 1. Fetch underlying price
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no price data for {ticker!r}")
    S = float(hist["Close"].iloc[-1])

    # 2. Time to expiry (in years)
    today = date.today()
    T = max((expiry - today).days, 1) / 365.0

    # 3. IV and Delta
    iv = bs_implied_volatility(market_price, S, strike, T, risk_free_rate, option_type)
    delta = bs_delta(S, strike, T, risk_free_rate, iv, option_type)

    # 4. GARCH(1,1) volatility
    garch_vol = calculate_garch_volatility(ticker)

    # 5. GARCH vs IV signal
    signal = compare_garch_iv(garch_vol, iv, threshold)

    # 6. Next earnings date
    earnings = get_next_earnings_dates([ticker])
    next_earnings = earnings.get(ticker)

    result = {
        "ticker": ticker,
        "underlying_price": S,
        "strike": strike,
        "expiry": expiry,
        "option_type": option_type,
        "market_price": market_price,
        "implied_volatility": iv,
        "delta": delta,
        "garch_volatility": garch_vol,
        "garch_exceeds_iv": signal,
        "next_earnings_date": next_earnings,
    }

    logger.info(
        "[%s] S=%.2f K=%.2f T=%.4f IV=%.4f delta=%.4f "
        "GARCH=%.4f signal=%s earnings=%s",
        ticker, S, strike, T, iv, delta, garch_vol, signal,
        next_earnings.date() if next_earnings else "N/A",
    )
    return result
