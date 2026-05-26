PERIODS = 60
ROLLING_PERIODS = 30
SHIFT_PERIODS = 0
MARGIN_PERCENT = 0.7        # Reduced from 0.40 — deploy less margin per session
MAX_POSITIONS = 10
MAX_RISK = 0.60              # Reduced from 0.50 — max daily loss = equity * 6.25% (was 20%)
MAX_HALF_LIFE = 1500.0
MAGIC_NUMBER = 345789
TRADING_PAIR_Y = ["WIN*"]
TRADING_PAIR_X = ["WDO*"]
START_TIME_HOUR = 12
START_TIME_MINUTE = 5
TRADE_WINDOW_TIME_HOURS = 4
TRADE_WINDOW_TIME_MINUTES = 30
Z_SCORE_ENTRY_THRESHOLD = 1.25  # Raised from 1.35 — higher quality signals only (~96th pct)
MARGIN_Y = 100
MARGIN_X = 150
VOLUME_FACTOR = 12
FIBO_VOLUME_FACTORS = [1,1,2,3,5,8]  # Flat sizing for first 5 grids (was [1,1,2,3,5,8,13] Martingale)
PROFIT_THRESHOLD = 0.40      # trailing activates at 40% of max_loss in profit
TRAILING_DISTANCE_POINTS = 25  # Raised from 5 — wider trailing to survive normal market noise
GRID_RANGE = 0.40            # Reduced from 0.60 — tighter z-score spacing between grids
MAX_GRIDS = 3                # Reduced from 5 — limit position buildup on diverging spreads
ADDITIONAL_GRID = GRID_RANGE / MAX_GRIDS
NOISE_VARIANCE = 0.004
KALMAN_FILTER_METHOD = True
COINTEGRATION_METHOD = "johansen"  # "johansen", "adf", or "engle"
JOHANSEN_CRIT_LEVEL = "90%"        # "90%", "95%", or "99%"
JOHANSEN_DET_ORDER = -1            # -1: no const (max power, intraday), 0: const in coint. vector, 1: unrestricted const (daily with drift)
JOHANSEN_MAX_LAGS = 3              # Max VAR lag candidates for AIC selection; keep small to protect degrees of freedom on limited samples
JOHANSEN_PERIODS = 60             # Bars fetched specifically for Johansen test (separate from PERIODS used for z-score/Kalman)
ADF_PVALUE_THRESHOLD = 0.10
ADF_CRIT_LEVEL = "10%"              # "1%", "5%", or "10%"
EG_PVALUE_THRESHOLD = 0.05
EG_CRIT_LEVEL = "10%"               # "1%", "5%", or "10%"
SCAN_COINTEGRATION_METHOD = "johansen"  # Applies only to scan_pairs_arbitrage
SCAN_JOHANSEN_CRIT_LEVEL = "90%"        # "90%", "95%", or "99%"
OU_LAMBDA_MIN = 0.01       # Minimum OU mean-reversion speed λ; below this the spread is near a random walk
VECM_ECT_THRESHOLD = 0.5   # Minimum |VECM ECT z-score| required to open orders
HURST_THRESHOLD = 0.5      # Spread Hurst exponent must be below this (mean-reverting)
WAVELET_LEVEL = 1          # DWT decomposition levels for spread denoising (higher = more smoothing)
KALMAN_ORDER = 1           # Kalman filter order for hedge-ratio estimation: 1 (standard) or 2 (tracks beta velocity/acceleration)

# ── Black-Scholes / GARCH / Earnings ─────────────────────────────────────────
BS_RISK_FREE_RATE = 0.05        # Annual risk-free rate used in Black-Scholes pricing
BS_GARCH_IV_THRESHOLD = 0.05    # Minimum excess of GARCH vol over IV to trigger a signal (5 pp)
BS_LOOKBACK_DAYS = 252          # Historical trading days used to fit GARCH(1,1)
BS_EARNINGS_LOOKAHEAD_DAYS = 30 # Upcoming days to scan for earnings announcements
