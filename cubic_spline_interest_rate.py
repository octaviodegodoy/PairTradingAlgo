"""
cubic_spline_interest_rate.py
------------------------------
Estimate the implied interest-rate term structure from futures prices using
cubic spline interpolation.

The cost-of-carry relationship for a continuously-compounded rate r is:
    F = S * exp(r * T)
    => r = ln(F / S) / T

Given N futures contracts at maturities T_1 < T_2 < … < T_N (in years), we:
  1. Compute the implied rate r_i for every contract.
  2. Fit a natural cubic spline through the (T_i, r_i) knot points.
  3. Expose evaluate() so callers can query the rate at any maturity.

Usage example (with MT5 live data)
------------------------------------
from cubic_spline_interest_rate import RateCurve

curve = RateCurve.from_mt5(mt5_conn, group="WIN*", spot_price=130_000)
rate_3m = curve.evaluate(0.25)   # implied rate at 3 months
print(f"Implied 3-month rate: {rate_3m:.4%}")

Usage example (synthetic / back-test data)
-------------------------------------------
import numpy as np
from cubic_spline_interest_rate import RateCurve

maturities = np.array([0.083, 0.25, 0.50, 0.75])  # months 1, 3, 6, 9
rates      = np.array([0.105, 0.108, 0.112, 0.115])
curve = RateCurve(maturities, rates)
print(curve.evaluate(0.33))
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: natural cubic spline implemented with NumPy only
# ---------------------------------------------------------------------------

def _fit_natural_cubic_spline(x: np.ndarray, y: np.ndarray):
    """
    Build a natural cubic spline through the knots (x[i], y[i]).

    A *natural* spline sets the second derivative to zero at both endpoints
    (M[0] = M[n-1] = 0), giving the smoothest possible interpolation.

    Returns a callable ``spline(t)`` that evaluates the piecewise cubic at t.

    Falls back to linear interpolation when len(x) < 3.
    """
    n = len(x)
    if n < 2:
        raise ValueError("At least 2 data points are required to build a spline.")

    if n == 2:
        # Degenerate case: single linear segment
        def _linear(t):
            return np.interp(t, x, y)
        return _linear

    # ------------------------------------------------------------------ #
    # Solve the tri-diagonal system for the second derivatives M[i].       #
    # Natural spline conditions: M[0] = M[n-1] = 0.                       #
    # ------------------------------------------------------------------ #
    h = np.diff(x).astype(float)          # h[i] = x[i+1] - x[i]
    dy = np.diff(y).astype(float)         # dy[i] = y[i+1] - y[i]

    # Build the right-hand side of the tri-diagonal system
    rhs = 6.0 * (dy[1:] / h[1:] - dy[:-1] / h[:-1])

    # Sub-, main, and super-diagonals (interior knots only: indices 1..n-2)
    main_diag = 2.0 * (h[:-1] + h[1:])
    off_diag = h[1:-1]

    # Thomas algorithm (forward sweep)
    size = n - 2
    c = off_diag.copy()
    d = main_diag.copy()
    r = rhs.copy()

    for i in range(1, size):
        factor = c[i - 1] / d[i - 1]
        d[i] -= factor * c[i - 1]
        r[i] -= factor * r[i - 1]

    # Back substitution
    M_interior = np.zeros(size)
    M_interior[-1] = r[-1] / d[-1]
    for i in range(size - 2, -1, -1):
        M_interior[i] = (r[i] - c[i] * M_interior[i + 1]) / d[i]

    # Full second-derivative array (natural BCs)
    M = np.concatenate([[0.0], M_interior, [0.0]])

    def _spline(t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the natural cubic spline at scalar or array *t*."""
        scalar_input = np.ndim(t) == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty_like(t_arr)

        for k, tk in enumerate(t_arr):
            # Clamp to the knot range (flat extrapolation outside)
            if tk <= x[0]:
                out[k] = y[0]
                continue
            if tk >= x[-1]:
                out[k] = y[-1]
                continue

            # Find the segment index i such that x[i] <= tk < x[i+1]
            i = int(np.searchsorted(x, tk, side="right")) - 1
            i = max(0, min(i, n - 2))

            hi = float(h[i])
            a = (x[i + 1] - tk) / hi
            b = (tk - x[i]) / hi

            out[k] = (
                a * y[i]
                + b * y[i + 1]
                + ((a**3 - a) * M[i] + (b**3 - b) * M[i + 1]) * hi**2 / 6.0
            )

        return float(out[0]) if scalar_input else out

    return _spline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RateCurve:
    """
    Cubic-spline interest-rate term structure estimated from futures prices.

    Attributes
    ----------
    maturities : np.ndarray
        Sorted array of maturities in years at which the rate was observed.
    rates : np.ndarray
        Implied continuously-compounded annualised interest rates at each
        maturity knot.
    """

    def __init__(self, maturities: Sequence[float], rates: Sequence[float]) -> None:
        maturities = np.asarray(maturities, dtype=float)
        rates = np.asarray(rates, dtype=float)

        if len(maturities) != len(rates):
            raise ValueError("maturities and rates must have the same length.")
        if len(maturities) < 2:
            raise ValueError("At least 2 (maturity, rate) pairs are required.")

        # Sort by maturity (ascending)
        order = np.argsort(maturities)
        self.maturities = maturities[order]
        self.rates = rates[order]
        self._spline = _fit_natural_cubic_spline(self.maturities, self.rates)

    # ------------------------------------------------------------------ #
    # Factory method: build directly from MT5                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_mt5(
        cls,
        mt5_conn,
        group: str,
        spot_price: Optional[float] = None,
        valuation_date: Optional[datetime] = None,
    ) -> "RateCurve":
        """
        Build a RateCurve from all active futures contracts in *group*.

        Parameters
        ----------
        mt5_conn : MT5Connector
            An initialised broker connector (MT5Connector instance).
        group : str
            Futures group wildcard, e.g. ``"WIN*"`` or ``"WDO*"``.
        spot_price : float, optional
            Underlying spot price.  When *None* the continuous contract price
            (WIN$ / WDO$) is used as a proxy for the spot.
        valuation_date : datetime, optional
            Pricing date for maturity calculation.  Defaults to *now* (UTC).

        Returns
        -------
        RateCurve
        """
        import MetaTrader5 as mt5  # only needed when called in a live session

        if valuation_date is None:
            valuation_date = datetime.now(timezone.utc)

        t_now = int(time.time())

        # Gather all active futures contracts for the group
        futures_symbols = mt5.symbols_get(group)
        if not futures_symbols:
            raise RuntimeError(f"No symbols found for group '{group}'")

        contracts = {
            s.expiration_time: s.name
            for s in futures_symbols
            # MT5 futures contract tickers are always exactly 6 characters long
            # (e.g. WINM25, WDOG25); longer names are continuous/rollover aliases.
            if len(s.name) == 6 and s.expiration_time > t_now
        }

        if len(contracts) < 2:
            raise RuntimeError(
                f"Need at least 2 active contracts for group '{group}', "
                f"found {len(contracts)}."
            )

        # Use continuous symbol as spot proxy when no explicit spot is given
        if spot_price is None:
            continuous_sym = group.replace("*", "$")
            tick_spot = mt5.symbol_info_tick(continuous_sym)
            if tick_spot is None:
                raise RuntimeError(
                    f"Cannot fetch spot price for '{continuous_sym}'. "
                    "Provide spot_price explicitly."
                )
            spot_price = (tick_spot.ask + tick_spot.bid) / 2.0
            logger.info("Spot proxy %s = %.4f", continuous_sym, spot_price)

        maturities: list[float] = []
        rates: list[float] = []

        for expiry_ts, name in sorted(contracts.items()):
            tick = mt5.symbol_info_tick(name)
            if tick is None:
                logger.warning("No tick data for %s — skipped.", name)
                continue

            futures_price = (tick.ask + tick.bid) / 2.0
            if futures_price <= 0 or spot_price <= 0:
                logger.warning("Non-positive price for %s — skipped.", name)
                continue

            # Time to expiry in years (act/365.25, accounting for leap years)
            expiry_dt = datetime.fromtimestamp(expiry_ts, tz=timezone.utc)
            T = (expiry_dt - valuation_date).total_seconds() / (365.25 * 86_400)
            if T <= 0:
                continue

            r = np.log(futures_price / spot_price) / T
            maturities.append(T)
            rates.append(r)
            logger.info(
                "Contract %s: F=%.4f, T=%.4f yrs, implied r=%.4f%%",
                name, futures_price, T, r * 100,
            )

        if len(maturities) < 2:
            raise RuntimeError(
                "Not enough valid contracts to build a rate curve "
                f"(need ≥ 2, got {len(maturities)})."
            )

        return cls(maturities, rates)

    # ------------------------------------------------------------------ #
    # Core evaluation                                                       #
    # ------------------------------------------------------------------ #

    def evaluate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        """
        Return the cubic-spline interpolated interest rate at *maturity* (years).

        For maturities outside the observed range the rate is held flat at
        the nearest end value (constant extrapolation).
        """
        return self._spline(maturity)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Compute the continuously-compounded forward rate between *t1* and *t2*
        (both in years, t2 > t1).
        """
        if t2 <= t1:
            raise ValueError("t2 must be strictly greater than t1.")
        r1 = float(self.evaluate(t1))
        r2 = float(self.evaluate(t2))
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def __repr__(self) -> str:  # pragma: no cover
        knots = len(self.maturities)
        t_min = self.maturities[0]
        t_max = self.maturities[-1]
        return (
            f"RateCurve(knots={knots}, "
            f"maturity_range=[{t_min:.3f}, {t_max:.3f}] yrs)"
        )
