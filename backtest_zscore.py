"""
Z-Score Entry Threshold Sweep Backtest
=======================================
Simulates pair-trading entries/exits on the live WIN*/WDO* Kalman z-score series
for a grid of Z_SCORE_ENTRY_THRESHOLD values and prints a ranked performance table.

Run:
    python backtest_zscore.py

Requires MT5 to be running (data fetched via MT5Connector).
"""

import numpy as np
import pandas as pd
import logging
from mt5_connector import MT5Connector
from utils import get_dynamic_spread_zscores
from constants import TRADING_PAIR_Y, TRADING_PAIR_X

logging.basicConfig(level=logging.WARNING)

# ── Backtest-specific settings (independent of live PERIODS) ─────────────────
BACKTEST_BARS = 500     # daily bars to fetch; more bars → more trades → better stats

# ── Sweep grid ────────────────────────────────────────────────────────────────
THRESHOLDS = [0.75, 1.00, 1.25, 1.45, 1.50, 1.75, 2.00, 2.25, 2.50]
EXIT_ZSCORE = 0.0   # z-score level at which the trade is closed


def backtest_threshold(z: np.ndarray, spread: np.ndarray, threshold: float) -> dict:
    """
    Vectorised single-threshold backtest on a z-score series.

    Entry rules
    -----------
    - LONG  spread (buy Y / sell X) when z crosses below  -threshold
    - SHORT spread (sell Y / buy X) when z crosses above  +threshold

    Exit rule
    ---------
    - Close when z reverts back through EXIT_ZSCORE (sign flip toward zero).

    P&L
    ---
    Measured in spread units (log-price spread), which is proportional to
    real P&L for a fixed hedge ratio.  No transaction costs are modelled.
    """
    trades = []
    position = 0   # +1 = long spread, -1 = short spread, 0 = flat
    entry_spread = 0.0
    entry_z = 0.0

    for i in range(1, len(z)):
        if np.isnan(z[i]) or np.isnan(z[i - 1]):
            continue

        if position == 0:
            # Enter LONG spread
            if z[i - 1] >= -threshold and z[i] < -threshold:
                position = 1
                entry_spread = spread[i]
                entry_z = z[i]
            # Enter SHORT spread
            elif z[i - 1] <= threshold and z[i] > threshold:
                position = -1
                entry_spread = spread[i]
                entry_z = z[i]
        else:
            # Exit when z reverts through EXIT_ZSCORE
            crossed_zero = (position == 1 and z[i] >= EXIT_ZSCORE) or \
                           (position == -1 and z[i] <= EXIT_ZSCORE)
            if crossed_zero:
                pnl = position * (spread[i] - entry_spread)
                trades.append({
                    'entry_z': entry_z,
                    'exit_z': z[i],
                    'pnl': pnl,
                })
                position = 0

    if not trades:
        return {
            'threshold': threshold,
            'n_trades': 0,
            'win_rate': float('nan'),
            'total_pnl': 0.0,
            'avg_pnl': float('nan'),
            'sharpe': float('nan'),
            'max_drawdown': float('nan'),
            'calmar': float('nan'),
        }

    df = pd.DataFrame(trades)
    pnl_series = df['pnl']
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min()

    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series)) \
        if pnl_series.std() > 0 else float('nan')
    calmar = (cum_pnl.iloc[-1] / abs(max_dd)) if max_dd != 0 else float('nan')

    return {
        'threshold': threshold,
        'n_trades': len(trades),
        'win_rate': (pnl_series > 0).mean(),
        'total_pnl': cum_pnl.iloc[-1],
        'avg_pnl': pnl_series.mean(),
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
    }


def main():
    mt5 = MT5Connector()

    symbol_y = TRADING_PAIR_Y[0]
    symbol_x = TRADING_PAIR_X[0]

    print(f"Fetching data for {symbol_y} / {symbol_x} ({BACKTEST_BARS} bars)…")
    data_y = mt5.get_data_futures_btg(symbol_y, n_bars=BACKTEST_BARS)
    data_x = mt5.get_data_futures_btg(symbol_x, n_bars=BACKTEST_BARS)

    if data_y.empty or data_x.empty:
        print("ERROR: could not fetch data from MT5.")
        return

    print(f"Data loaded: {len(data_y)} bars Y, {len(data_x)} bars X")
    print("Computing Kalman z-scores…")
    result = get_dynamic_spread_zscores(data_y, data_x)

    z = result['z_scores'].values.astype(float)
    spread = result['spread'].values.astype(float)

    rows = []
    for thr in THRESHOLDS:
        row = backtest_threshold(z, spread, thr)
        rows.append(row)
        print(f"  threshold={thr:.2f}  trades={row['n_trades']:>3}  "
              f"win={row['win_rate']:.0%}  sharpe={row['sharpe']:>6.2f}  "
              f"calmar={row['calmar']:>6.2f}  total_pnl={row['total_pnl']:>8.5f}  "
              f"max_dd={row['max_drawdown']:>8.5f}")

    table = pd.DataFrame(rows)

    # ── Recommendation: highest Sharpe among thresholds with ≥5 trades ───────
    eligible = table[(table['n_trades'] >= 5) & table['sharpe'].notna()]
    if eligible.empty:
        print("\nNot enough trades to recommend a threshold — try more bars (increase BACKTEST_BARS).")
        return

    best_sharpe_row = eligible.loc[eligible['sharpe'].idxmax()]

    # Calmar may be nan when max_dd==0 (no losing trades); fall back to Sharpe
    calmar_eligible = eligible[eligible['calmar'].notna()]
    if not calmar_eligible.empty:
        best_calmar_row = calmar_eligible.loc[calmar_eligible['calmar'].idxmax()]
        calmar_note = f"Calmar={best_calmar_row['calmar']:.2f}, MaxDD={best_calmar_row['max_drawdown']:.5f}"
    else:
        # All max_dd == 0: no drawdown observed → rank by Sharpe × n_trades
        eligible = eligible.copy()
        eligible['score'] = eligible['sharpe'] * eligible['n_trades']
        best_calmar_row = eligible.loc[eligible['score'].idxmax()]
        calmar_note = f"(no drawdown observed — ranked by Sharpe×Trades, score={best_calmar_row['score']:.1f})"

    print("\n" + "═" * 70)
    print(f"  Best SHARPE  → threshold = {best_sharpe_row['threshold']:.2f}  "
          f"(Sharpe={best_sharpe_row['sharpe']:.2f}, "
          f"WinRate={best_sharpe_row['win_rate']:.0%}, "
          f"Trades={int(best_sharpe_row['n_trades'])})")
    print(f"  Best CALMAR  → threshold = {best_calmar_row['threshold']:.2f}  "
          f"({calmar_note}, "
          f"Trades={int(best_calmar_row['n_trades'])})")
    print("═" * 70)

    print("\nFull table (sorted by Sharpe):")
    print(table.sort_values('sharpe', ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
