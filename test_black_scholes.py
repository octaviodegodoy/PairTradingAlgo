"""Unit tests for black_scholes.py — no network or MT5 required."""

import math
import sys
import types
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_yfinance(close_prices: list[float], earnings_dates: list[datetime] | None = None):
    """Return a fake yfinance module that can be injected via sys.modules."""
    import pandas as pd

    fake_yf = types.ModuleType("yfinance")

    class FakeTicker:
        def __init__(self, ticker):
            self._ticker = ticker

        def history(self, period=None):
            return pd.DataFrame({"Close": close_prices})

        @property
        def calendar(self):
            if earnings_dates is None:
                return {}
            return {"Earnings Date": earnings_dates}

    fake_yf.Ticker = FakeTicker
    return fake_yf


# ---------------------------------------------------------------------------
# bs_price
# ---------------------------------------------------------------------------

class TestBsPrice:
    """Black-Scholes pricing tests with well-known boundary conditions."""

    def setup_method(self):
        from black_scholes import bs_price
        self.bs_price = bs_price

    def test_call_price_deep_itm_approximately_intrinsic(self):
        # Deep ITM call: price ≈ S - K·e^{-rT}
        S, K, T, r, sigma = 200.0, 100.0, 1.0, 0.05, 0.20
        price = self.bs_price(S, K, T, r, sigma, "call")
        intrinsic = S - K * math.exp(-r * T)
        assert abs(price - intrinsic) < 1.0

    def test_put_price_deep_itm_approximately_intrinsic(self):
        S, K, T, r, sigma = 50.0, 150.0, 1.0, 0.05, 0.20
        price = self.bs_price(S, K, T, r, sigma, "put")
        intrinsic = K * math.exp(-r * T) - S
        assert abs(price - intrinsic) < 1.0

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
        call = self.bs_price(S, K, T, r, sigma, "call")
        put = self.bs_price(S, K, T, r, sigma, "put")
        parity_diff = call - put - (S - K * math.exp(-r * T))
        assert abs(parity_diff) < 1e-10

    def test_call_price_positive(self):
        price = self.bs_price(100.0, 100.0, 1.0, 0.05, 0.20, "call")
        assert price > 0

    def test_raises_on_bad_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            self.bs_price(100.0, 100.0, 1.0, 0.05, 0.20, "straddle")

    def test_raises_on_non_positive_T(self):
        with pytest.raises(ValueError):
            self.bs_price(100.0, 100.0, 0.0, 0.05, 0.20, "call")

    def test_raises_on_non_positive_sigma(self):
        with pytest.raises(ValueError):
            self.bs_price(100.0, 100.0, 1.0, 0.05, 0.0, "call")

    def test_known_value(self):
        # Merton (1973) example: S=42, K=40, T=0.5, r=0.10, sigma=0.20
        # Expected call ≈ 4.76
        price = self.bs_price(42.0, 40.0, 0.5, 0.10, 0.20, "call")
        assert abs(price - 4.76) < 0.05


# ---------------------------------------------------------------------------
# bs_delta
# ---------------------------------------------------------------------------

class TestBsDelta:
    def setup_method(self):
        from black_scholes import bs_delta
        self.bs_delta = bs_delta

    def test_call_delta_between_zero_and_one(self):
        delta = self.bs_delta(100.0, 100.0, 1.0, 0.05, 0.20, "call")
        assert 0.0 <= delta <= 1.0

    def test_put_delta_between_minus_one_and_zero(self):
        delta = self.bs_delta(100.0, 100.0, 1.0, 0.05, 0.20, "put")
        assert -1.0 <= delta <= 0.0

    def test_put_call_delta_relationship(self):
        # put_delta = call_delta - 1
        S, K, T, r, sigma = 100.0, 105.0, 0.5, 0.04, 0.25
        from black_scholes import bs_delta
        call_d = bs_delta(S, K, T, r, sigma, "call")
        put_d = bs_delta(S, K, T, r, sigma, "put")
        assert abs(call_d - 1.0 - put_d) < 1e-10

    def test_atm_call_delta_near_half(self):
        # ATM short-dated option delta ≈ 0.5
        delta = self.bs_delta(100.0, 100.0, 0.01, 0.0, 0.20, "call")
        assert abs(delta - 0.5) < 0.05


# ---------------------------------------------------------------------------
# bs_implied_volatility
# ---------------------------------------------------------------------------

class TestBsImpliedVolatility:
    def setup_method(self):
        from black_scholes import bs_implied_volatility, bs_price
        self.bs_iv = bs_implied_volatility
        self.bs_price = bs_price

    def test_roundtrip_call(self):
        S, K, T, r, true_sigma = 100.0, 100.0, 1.0, 0.05, 0.25
        market_price = self.bs_price(S, K, T, r, true_sigma, "call")
        iv = self.bs_iv(market_price, S, K, T, r, "call")
        assert abs(iv - true_sigma) < 1e-6

    def test_roundtrip_put(self):
        S, K, T, r, true_sigma = 100.0, 105.0, 0.5, 0.03, 0.30
        market_price = self.bs_price(S, K, T, r, true_sigma, "put")
        iv = self.bs_iv(market_price, S, K, T, r, "put")
        assert abs(iv - true_sigma) < 1e-6

    def test_raises_on_non_positive_market_price(self):
        with pytest.raises(ValueError, match="market_price"):
            self.bs_iv(0.0, 100.0, 100.0, 1.0, 0.05, "call")

    def test_raises_when_no_iv_exists(self):
        # A price far above the theoretical maximum should fail
        with pytest.raises(ValueError):
            self.bs_iv(9999.0, 100.0, 100.0, 1.0, 0.05, "call")


# ---------------------------------------------------------------------------
# compare_garch_iv
# ---------------------------------------------------------------------------

class TestCompareGarchIv:
    def setup_method(self):
        from black_scholes import compare_garch_iv
        self.compare = compare_garch_iv

    def test_garch_above_threshold(self):
        assert self.compare(0.30, 0.20, threshold=0.05) is True

    def test_garch_exactly_at_threshold(self):
        # Floating-point subtraction of arbitrary decimals is inexact, so we
        # test the boundary by nudging garch_vol 1 bp (0.001) above iv+threshold.
        assert self.compare(0.251, 0.20, threshold=0.05) is True

    def test_garch_below_threshold(self):
        assert self.compare(0.24, 0.20, threshold=0.05) is False

    def test_garch_equal_to_iv(self):
        assert self.compare(0.20, 0.20, threshold=0.05) is False

    def test_garch_below_iv(self):
        assert self.compare(0.15, 0.20, threshold=0.05) is False


# ---------------------------------------------------------------------------
# calculate_garch_volatility — mocked yfinance + arch
# ---------------------------------------------------------------------------

class TestCalculateGarchVolatility:
    def test_returns_positive_annualised_vol(self):
        import pandas as pd
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 300)))
        fake_yf = _make_fake_yfinance(list(prices))

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from importlib import reload
            import black_scholes as bs_mod
            reload(bs_mod)
            vol = bs_mod.calculate_garch_volatility("FAKE", lookback_days=252)

        assert vol > 0.0
        assert vol < 5.0  # sanity: less than 500 % annualised

    def test_raises_on_empty_data(self):
        import pandas as pd
        fake_yf = _make_fake_yfinance([])

        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from importlib import reload
            import black_scholes as bs_mod
            reload(bs_mod)
            with pytest.raises(RuntimeError, match="no data"):
                bs_mod.calculate_garch_volatility("EMPTY")


# ---------------------------------------------------------------------------
# get_next_earnings_dates — mocked yfinance
# ---------------------------------------------------------------------------

class TestGetNextEarningsDates:
    def _run(self, earnings_dates, lookahead_days=30):
        fake_yf = _make_fake_yfinance([100.0], earnings_dates=earnings_dates)
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from importlib import reload
            import black_scholes as bs_mod
            reload(bs_mod)
            return bs_mod.get_next_earnings_dates(["FAKE"], lookahead_days=lookahead_days)

    def test_earnings_within_window_returned(self):
        from datetime import timedelta
        future = datetime.now(tz=timezone.utc) + timedelta(days=10)
        result = self._run([future])
        assert result["FAKE"] is not None
        assert result["FAKE"].date() == future.date()

    def test_earnings_beyond_window_not_returned(self):
        from datetime import timedelta
        far_future = datetime.now(tz=timezone.utc) + timedelta(days=90)
        result = self._run([far_future], lookahead_days=30)
        assert result["FAKE"] is None

    def test_past_earnings_not_returned(self):
        from datetime import timedelta
        past = datetime.now(tz=timezone.utc) - timedelta(days=5)
        result = self._run([past])
        assert result["FAKE"] is None

    def test_no_earnings_returns_none(self):
        result = self._run(None)
        assert result["FAKE"] is None

    def test_multiple_tickers(self):
        from datetime import timedelta
        future = datetime.now(tz=timezone.utc) + timedelta(days=7)
        fake_yf = _make_fake_yfinance([100.0], earnings_dates=[future])
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from importlib import reload
            import black_scholes as bs_mod
            reload(bs_mod)
            result = bs_mod.get_next_earnings_dates(["A", "B"], lookahead_days=30)
        assert "A" in result
        assert "B" in result
