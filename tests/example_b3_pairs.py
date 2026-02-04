# example_b3_pairs.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kalman_pairs_strategy import KalmanPairsTradingStrategy
import matplotlib.pyplot as plt

# Download B3 data
def download_b3_data(tickers: list, period: str = '1y'):
    """Download B3 stock data"""
    data = {}
    for ticker in tickers:
        df = yf.download(f"{ticker}.SA", period=period, progress=False)
        data[ticker] = df['Close']
    
    return pd.DataFrame(data).dropna()

# Main execution
if __name__ == "__main__":
    
    print("🇧🇷 KALMAN FILTER PAIRS TRADING - B3 MARKET")
    print("="*60)
    
    # Example: PETR4 vs PETR3 (same company, different share classes)
    tickers = ['PETR4', 'PETR3']
    
    print(f"\n📊 Downloading data for {tickers}...")
    prices = download_b3_data(tickers, period='1y')
    
    print(f"✅ Downloaded {len(prices)} days of data")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Initialize strategy
    strategy = KalmanPairsTradingStrategy(
        delta=1e-4,              # Process variance
        ve=1e-3,                 # Observation variance
        zscore_window=30,        # 30-day rolling window
        entry_threshold=2.0,     # Enter at 2 std deviations
        exit_threshold=0.5,      # Exit at 0.5 std deviations
        stop_loss=3.0           # Stop loss at 3 std deviations
    )
    
    # Fit the model
    print(f"\n🔧 Fitting Kalman Filter...")
    results = strategy.fit(prices['PETR4'], prices['PETR3'])
    
    # Backtest
    print(f"📈 Running backtest...")
    metrics = strategy.backtest(
        transaction_cost=0.001,  # 0.1% transaction cost (B3 typical)
        initial_capital=100000   # R$ 100,000
    )
    
    # Print results
    strategy.print_metrics(metrics)
    
    # Current status
    print("📊 CURRENT STATUS")
    print("="*60)
    print(f"Current Hedge Ratio: {results['hedge_ratio'].iloc[-1]:.4f}")
    print(f"Current Spread:      {results['spread'].iloc[-1]:.2f}")
    print(f"Current Z-Score:     {results['zscore'].iloc[-1]:.2f}")
    print(f"Current Position:    {results['position'].iloc[-1]}")
    
    if results['position'].iloc[-1] == 1:
        print("   → LONG SPREAD (Long PETR3, Short PETR4)")
    elif results['position'].iloc[-1] == -1:
        print("   → SHORT SPREAD (Short PETR3, Long PETR4)")
    else:
        print("   → NO POSITION")
    
    print("="*60)
    
    # Plot results
    print("\n📊 Generating plots...")
    strategy.plot_results()
    
    print("✅ Analysis complete!")