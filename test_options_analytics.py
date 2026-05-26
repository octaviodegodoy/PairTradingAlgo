"""
test_options_analytics.py
--------------------------
Unit tests for the Black-Scholes helpers and get_puts_iv_delta function
in options_analytics.py.  All tests are pure-Python (no MT5 connection
required); broker interaction is covered by a lightweight mock.
"""

import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from options_analytics import (
    black_scholes_put_price,
    black_scholes_put_delta,
    compute_implied_volatility,
    get_puts_iv_delta,
)

# Bid/ask spread multipliers used in mock ticks
UNDERLYING_ASK_SPREAD = 1.001   # 0.1 % above mid
UNDERLYING_BID_SPREAD = 0.999   # 0.1 % below mid
OPTION_ASK_SPREAD = 1.01        # 1 % above last
OPTION_BID_SPREAD = 0.99        # 1 % below last


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

class TestBlackScholesPutPrice(unittest.TestCase):

    def test_atm_put_positive(self):
        """ATM put with positive T and sigma must have a positive price."""
        price = black_scholes_put_price(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        self.assertGreater(price, 0)

    def test_deep_itm_put(self):
        """Deep ITM put price must be close to the discounted intrinsic value."""
        S, K, T, r, sigma = 50, 100, 0.5, 0.05, 0.20
        intrinsic = K * math.exp(-r * T) - S
        price = black_scholes_put_price(S, K, T, r, sigma)
        self.assertAlmostEqual(price, intrinsic, delta=1.0)

    def test_deep_otm_put_near_zero(self):
        """Deep OTM put price must be very close to zero."""
        price = black_scholes_put_price(S=200, K=50, T=0.25, r=0.05, sigma=0.20)
        self.assertAlmostEqual(price, 0, delta=0.01)

    def test_zero_time_returns_intrinsic(self):
        """At expiry (T=0), price must equal max(K-S, 0)."""
        self.assertAlmostEqual(black_scholes_put_price(90, 100, 0, 0.05, 0.20), 10.0)
        self.assertAlmostEqual(black_scholes_put_price(110, 100, 0, 0.05, 0.20), 0.0)

    def test_known_value(self):
        """Cross-check against a textbook example."""
        # S=100, K=100, T=1, r=0.05, sigma=0.2 → put ≈ 5.5735
        price = black_scholes_put_price(100, 100, 1.0, 0.05, 0.20)
        self.assertAlmostEqual(price, 5.5735, delta=0.01)


class TestBlackScholesPutDelta(unittest.TestCase):

    def test_delta_range(self):
        """Put delta must always be in [-1, 0]."""
        for S in [80, 100, 120]:
            delta = black_scholes_put_delta(S=S, K=100, T=0.5, r=0.05, sigma=0.20)
            self.assertGreaterEqual(delta, -1.0)
            self.assertLessEqual(delta, 0.0)

    def test_atm_put_delta_near_minus_half(self):
        """ATM put delta is approximately -0.5 for long maturities."""
        delta = black_scholes_put_delta(S=100, K=100, T=1.0, r=0.0, sigma=0.20)
        self.assertAlmostEqual(delta, -0.5, delta=0.05)

    def test_deep_itm_delta_near_minus_one(self):
        """Deep ITM put delta must be close to -1."""
        delta = black_scholes_put_delta(S=50, K=100, T=0.5, r=0.05, sigma=0.20)
        self.assertAlmostEqual(delta, -1.0, delta=0.05)

    def test_deep_otm_delta_near_zero(self):
        """Deep OTM put delta must be close to 0."""
        delta = black_scholes_put_delta(S=200, K=100, T=0.5, r=0.05, sigma=0.20)
        self.assertAlmostEqual(delta, 0.0, delta=0.05)

    def test_zero_time_itm(self):
        """At expiry, ITM put delta must be -1."""
        self.assertEqual(black_scholes_put_delta(S=90, K=100, T=0, r=0.05, sigma=0.20), -1.0)

    def test_zero_time_otm(self):
        """At expiry, OTM put delta must be 0."""
        self.assertEqual(black_scholes_put_delta(S=110, K=100, T=0, r=0.05, sigma=0.20), 0.0)


class TestComputeImpliedVolatility(unittest.TestCase):

    def _roundtrip(self, S, K, T, r, sigma):
        """Price → IV must recover the original sigma to within 0.001."""
        price = black_scholes_put_price(S, K, T, r, sigma)
        recovered = compute_implied_volatility(price, S, K, T, r)
        self.assertAlmostEqual(recovered, sigma, delta=0.001,
                               msg=f"Roundtrip failed for S={S} K={K} T={T} r={r} sigma={sigma}")

    def test_roundtrip_atm(self):
        self._roundtrip(100, 100, 0.5, 0.05, 0.20)

    def test_roundtrip_itm(self):
        self._roundtrip(90, 100, 0.25, 0.05, 0.30)

    def test_roundtrip_otm(self):
        self._roundtrip(110, 100, 1.0, 0.03, 0.25)

    def test_roundtrip_high_vol(self):
        self._roundtrip(100, 100, 0.5, 0.10, 0.80)

    def test_zero_time_returns_nan(self):
        result = compute_implied_volatility(5.0, 95, 100, 0, 0.05)
        self.assertTrue(math.isnan(result))

    def test_negative_price_returns_nan(self):
        result = compute_implied_volatility(-1.0, 100, 100, 0.5, 0.05)
        self.assertTrue(math.isnan(result))

    def test_below_intrinsic_returns_nan(self):
        """Option price below intrinsic value has no real IV solution."""
        intrinsic = 100 * math.exp(-0.05 * 0.5) - 50  # ≈ 47.5
        result = compute_implied_volatility(intrinsic - 1, 50, 100, 0.5, 0.05)
        self.assertTrue(math.isnan(result))


# ---------------------------------------------------------------------------
# get_puts_iv_delta — integration with mock broker
# ---------------------------------------------------------------------------

class TestGetPutsIvDelta(unittest.TestCase):

    def _make_option(self, name, strike, expiry_offset_days, volume):
        """Build a minimal mock option symbol-info object."""
        opt = SimpleNamespace(
            name=name,
            option_strike=float(strike),
            expiration_time=int(time.time()) + int(expiry_offset_days * 86_400),
            volume_real=float(volume),
            visible=True,
        )
        return opt

    def _make_broker(self, underlying_price, options, option_last_prices):
        """Build a mock broker that returns the given options and tick prices."""
        broker = MagicMock()

        # Underlying tick
        underlying_tick = SimpleNamespace(ask=underlying_price * UNDERLYING_ASK_SPREAD,
                                          bid=underlying_price * UNDERLYING_BID_SPREAD,
                                          last=underlying_price)
        broker.get_symbol_tick.side_effect = lambda sym: (
            underlying_tick if sym == "UNDERLYING"
            else SimpleNamespace(ask=option_last_prices.get(sym, 0) * OPTION_ASK_SPREAD,
                                 bid=option_last_prices.get(sym, 0) * OPTION_BID_SPREAD,
                                 last=option_last_prices.get(sym, 0))
        )
        broker.get_options_puts.return_value = options
        return broker

    def test_returns_dataframe_with_expected_columns(self):
        opts = [
            self._make_option("OPT1", 95, 30, 1000),
            self._make_option("OPT2", 90, 30, 500),
        ]
        prices = {"OPT1": 3.5, "OPT2": 1.2}
        broker = self._make_broker(100.0, opts, prices)

        df = get_puts_iv_delta(broker, "UNDERLYING", risk_free_rate=0.05, n_top=5)

        expected_cols = {"symbol", "strike", "expiry_days", "last_price",
                         "volume", "underlying_price", "iv", "delta"}
        self.assertEqual(set(df.columns), expected_cols)

    def test_sorted_by_volume_descending(self):
        opts = [
            self._make_option("LOW_VOL",  95, 30, 100),
            self._make_option("HIGH_VOL", 95, 30, 9999),
            self._make_option("MED_VOL",  90, 30, 500),
        ]
        prices = {"LOW_VOL": 3.0, "HIGH_VOL": 3.0, "MED_VOL": 1.5}
        broker = self._make_broker(100.0, opts, prices)

        df = get_puts_iv_delta(broker, "UNDERLYING", risk_free_rate=0.05, n_top=3)

        self.assertEqual(df.iloc[0]["symbol"], "HIGH_VOL")
        self.assertEqual(df.iloc[1]["symbol"], "MED_VOL")
        self.assertEqual(df.iloc[2]["symbol"], "LOW_VOL")

    def test_n_top_respected(self):
        opts = [self._make_option(f"OPT{i}", 100 - i, 30, 1000 - i * 10) for i in range(10)]
        prices = {f"OPT{i}": 2.0 for i in range(10)}
        broker = self._make_broker(100.0, opts, prices)

        df = get_puts_iv_delta(broker, "UNDERLYING", n_top=3)

        self.assertEqual(len(df), 3)

    def test_iv_and_delta_are_finite_for_valid_options(self):
        opts = [self._make_option("OPT1", 95, 45, 500)]
        # Price 3.5 is a realistic ATM/OTM put for S=100, K=95, T≈0.12, r=0.05
        prices = {"OPT1": 3.5}
        broker = self._make_broker(100.0, opts, prices)

        df = get_puts_iv_delta(broker, "UNDERLYING", risk_free_rate=0.05)

        self.assertEqual(len(df), 1)
        self.assertFalse(math.isnan(df.iloc[0]["iv"]))
        self.assertFalse(math.isnan(df.iloc[0]["delta"]))
        # Delta must be in (-1, 0)
        self.assertGreater(df.iloc[0]["delta"], -1.0)
        self.assertLess(df.iloc[0]["delta"], 0.0)

    def test_zero_last_price_yields_nan_iv_delta(self):
        opts = [self._make_option("OPT1", 95, 30, 500)]
        prices = {"OPT1": 0}  # no last price
        broker = self._make_broker(100.0, opts, prices)

        df = get_puts_iv_delta(broker, "UNDERLYING", risk_free_rate=0.05)

        self.assertTrue(math.isnan(df.iloc[0]["iv"]))
        self.assertTrue(math.isnan(df.iloc[0]["delta"]))

    def test_empty_options_returns_empty_dataframe(self):
        broker = MagicMock()
        broker.get_symbol_tick.return_value = SimpleNamespace(ask=101, bid=99, last=100)
        broker.get_options_puts.return_value = []

        df = get_puts_iv_delta(broker, "UNDERLYING")

        self.assertTrue(df.empty)

    def test_underlying_tick_none_raises_runtime_error(self):
        broker = MagicMock()
        broker.get_symbol_tick.return_value = None

        with self.assertRaises(RuntimeError):
            get_puts_iv_delta(broker, "UNDERLYING")


if __name__ == "__main__":
    unittest.main()
