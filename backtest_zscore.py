"""
Z-Score Entry Threshold Sweep Backtest
=======================================
Simulates pair-trading entries/exits on the live WIN*/WDO* Kalman z-score series
for a grid of Z_SCORE_ENTRY_THRESHOLD values and prints a ranked performance table.

Two simulation modes are available and both are run by default:

1. **Flat mode** (original) — a single-unit entry/exit with no grid.
2. **Grid mode** (new) — Fibonacci-scaled grid that adds up to MAX_GRIDS layers
   as the spread diverges further, then closes all layers on mean-reversion.
   Commission per round-turn is configurable via COMMISSION_PER_RT.

Run:
    python backtest_zscore.py

Requires MT5 to be running (data fetched via MT5Connector).
"""

import numpy as np
import pandas as pd
import logging
from mt5_connector import MT5Connector
from utils import get_dynamic_spread_zscores
from constants import (
    TRADING_PAIR_Y, TRADING_PAIR_X,
    FIBO_VOLUME_FACTORS, GRID_RANGE, MAX_GRIDS,
)

logging.basicConfig(level=logging.WARNING)

# ── Backtest-specific settings (independent of live PERIODS) ─────────────────
BACKTEST_BARS = 500     # daily bars to fetch; more bars → more trades → better stats

# ── Sweep grid ────────────────────────────────────────────────────────────────
THRESHOLDS = [0.75, 1.00, 1.25, 1.45, 1.50, 1.75, 2.00, 2.25, 2.50]
EXIT_ZSCORE = 0.0        # z-score level at which the trade is closed
COMMISSION_PER_RT = 0.0  # round-turn commission per unit of spread (set > 0 to model costs)


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
    trade_pnls = []
    position = 0   # +1 = long spread, -1 = short spread, 0 = flat
    entry_spread = 0.0

    for i in range(1, len(z)):
        if np.isnan(z[i]) or np.isnan(z[i - 1]):
            continue

        if position == 0:
            # Enter LONG spread
            if z[i - 1] >= -threshold and z[i] < -threshold:
                position = 1
                entry_spread = spread[i]
            # Enter SHORT spread
            elif z[i - 1] <= threshold and z[i] > threshold:
                position = -1
                entry_spread = spread[i]
        else:
            # Exit when z reverts through EXIT_ZSCORE
            crossed_zero = (position == 1 and z[i] >= EXIT_ZSCORE) or \
                           (position == -1 and z[i] <= EXIT_ZSCORE)
            if crossed_zero:
                trade_pnls.append(position * (spread[i] - entry_spread))
                position = 0

    return _summary_stats(pd.Series(trade_pnls), threshold)


def _summary_stats(pnl_series: pd.Series, threshold: float) -> dict:
    """Compute common performance metrics from a series of per-trade P&Ls."""
    if pnl_series.empty:
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
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min()

    sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series)) \
        if pnl_series.std() > 0 else float('nan')
    calmar = (cum_pnl.iloc[-1] / abs(max_dd)) if max_dd != 0 else float('nan')

    return {
        'threshold': threshold,
        'n_trades': len(pnl_series),
        'win_rate': (pnl_series > 0).mean(),
        'total_pnl': cum_pnl.iloc[-1],
        'avg_pnl': pnl_series.mean(),
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
    }


def backtest_grid(z: np.ndarray, spread: np.ndarray, threshold: float) -> dict:
    """
    Grid simulation backtest.

    Mimics the live trading logic:
    - Opens the first layer when |z| crosses the entry threshold.
    - Adds up to MAX_GRIDS layers spaced GRID_RANGE / MAX_GRIDS apart.
    - Each layer's size is scaled by the Fibonacci factor for that grid level.
    - All layers are closed together when z reverts through EXIT_ZSCORE.
    - COMMISSION_PER_RT is applied per unit of spread per round-turn.

    P&L per round-turn is measured in spread units, scaled by Fibonacci lot size.
    """
    GRID_STEP = GRID_RANGE / MAX_GRIDS  # additional z-score spacing between layers

    trade_pnls = []
    layers: list[dict] = []   # each dict: {'direction': +1/-1, 'entry_spread': float, 'fibo': float}
    next_entry_threshold = threshold  # threshold for the next grid layer

    for i in range(1, len(z)):
        if np.isnan(z[i]) or np.isnan(z[i - 1]):
            continue

        # ── Entry / grid addition ───────────────────────────────────────────
        if len(layers) < MAX_GRIDS:
            # Determine direction from first entry; subsequent layers must match
            if len(layers) == 0:
                if z[i - 1] >= -next_entry_threshold and z[i] < -next_entry_threshold:
                    direction = 1   # LONG spread
                elif z[i - 1] <= next_entry_threshold and z[i] > next_entry_threshold:
                    direction = -1  # SHORT spread
                else:
                    direction = 0
                if direction != 0:
                    grid_idx = 0
                    fibo = FIBO_VOLUME_FACTORS[min(grid_idx, len(FIBO_VOLUME_FACTORS) - 1)]
                    layers.append({'direction': direction, 'entry_spread': spread[i], 'fibo': fibo})
                    next_entry_threshold = threshold + (grid_idx + 1) * GRID_STEP
            else:
                direction = layers[0]['direction']
                grid_idx = len(layers)
                # Add a layer if z moves further away from equilibrium
                if direction == 1 and z[i] < -next_entry_threshold:
                    fibo = FIBO_VOLUME_FACTORS[min(grid_idx, len(FIBO_VOLUME_FACTORS) - 1)]
                    layers.append({'direction': direction, 'entry_spread': spread[i], 'fibo': fibo})
                    next_entry_threshold = threshold + (grid_idx + 1) * GRID_STEP
                elif direction == -1 and z[i] > next_entry_threshold:
                    fibo = FIBO_VOLUME_FACTORS[min(grid_idx, len(FIBO_VOLUME_FACTORS) - 1)]
                    layers.append({'direction': direction, 'entry_spread': spread[i], 'fibo': fibo})
                    next_entry_threshold = threshold + (grid_idx + 1) * GRID_STEP

        # ── Exit: mean-reversion through EXIT_ZSCORE ────────────────────────
        if layers:
            direction = layers[0]['direction']
            crossed_exit = (direction == 1 and z[i] >= EXIT_ZSCORE) or \
                           (direction == -1 and z[i] <= EXIT_ZSCORE)
            if crossed_exit:
                total_pnl = 0.0
                for layer in layers:
                    raw_pnl = layer['direction'] * (spread[i] - layer['entry_spread']) * layer['fibo']
                    commission = COMMISSION_PER_RT * layer['fibo']
                    total_pnl += raw_pnl - commission
                trade_pnls.append(total_pnl)
                layers = []
                next_entry_threshold = threshold

    return _summary_stats(pd.Series(trade_pnls), threshold)


def _print_recommendation(table: pd.DataFrame, label: str) -> None:
    eligible = table[(table['n_trades'] >= 5) & table['sharpe'].notna()]
    if eligible.empty:
        print(f"\n[{label}] Not enough trades to recommend a threshold — try more bars.")
        return

    best_sharpe_row = eligible.loc[eligible['sharpe'].idxmax()]

    calmar_eligible = eligible[eligible['calmar'].notna()]
    if not calmar_eligible.empty:
        best_calmar_row = calmar_eligible.loc[calmar_eligible['calmar'].idxmax()]
        calmar_note = f"Calmar={best_calmar_row['calmar']:.2f}, MaxDD={best_calmar_row['max_drawdown']:.5f}"
    else:
        eligible = eligible.copy()
        eligible['score'] = eligible['sharpe'] * eligible['n_trades']
        best_calmar_row = eligible.loc[eligible['score'].idxmax()]
        calmar_note = f"(no drawdown observed — ranked by Sharpe×Trades, score={best_calmar_row['score']:.1f})"

    print(f"\n[{label}] " + "═" * 60)
    print(f"  Best SHARPE  → threshold = {best_sharpe_row['threshold']:.2f}  "
          f"(Sharpe={best_sharpe_row['sharpe']:.2f}, "
          f"WinRate={best_sharpe_row['win_rate']:.0%}, "
          f"Trades={int(best_sharpe_row['n_trades'])})")
    print(f"  Best CALMAR  → threshold = {best_calmar_row['threshold']:.2f}  "
          f"({calmar_note}, "
          f"Trades={int(best_calmar_row['n_trades'])})")
    print("═" * 68)
    print(f"\n[{label}] Full table (sorted by Sharpe):")
    print(table.sort_values('sharpe', ascending=False).to_string(index=False))


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

    # ── Flat (single-entry) sweep ──────────────────────────────────────────────
    print("\n── Flat (single-unit) backtest ──")
    flat_rows = []
    for thr in THRESHOLDS:
        row = backtest_threshold(z, spread, thr)
        flat_rows.append(row)
        print(f"  threshold={thr:.2f}  trades={row['n_trades']:>3}  "
              f"win={row['win_rate']:.0%}  sharpe={row['sharpe']:>6.2f}  "
              f"calmar={row['calmar']:>6.2f}  total_pnl={row['total_pnl']:>8.5f}  "
              f"max_dd={row['max_drawdown']:>8.5f}")

    flat_table = pd.DataFrame(flat_rows)
    _print_recommendation(flat_table, "FLAT")

    # ── Grid sweep ────────────────────────────────────────────────────────────
    print(f"\n── Grid backtest (MAX_GRIDS={MAX_GRIDS}, GRID_RANGE={GRID_RANGE}, "
          f"commission={COMMISSION_PER_RT}) ──")
    grid_rows = []
    for thr in THRESHOLDS:
        row = backtest_grid(z, spread, thr)
        grid_rows.append(row)
        print(f"  threshold={thr:.2f}  trades={row['n_trades']:>3}  "
              f"win={row['win_rate']:.0%}  sharpe={row['sharpe']:>6.2f}  "
              f"calmar={row['calmar']:>6.2f}  total_pnl={row['total_pnl']:>8.5f}  "
              f"max_dd={row['max_drawdown']:>8.5f}")

    grid_table = pd.DataFrame(grid_rows)
    _print_recommendation(grid_table, "GRID")


if __name__ == "__main__":
    main()
