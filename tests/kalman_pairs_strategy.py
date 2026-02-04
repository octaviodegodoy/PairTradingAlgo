# kalman_pairs_strategy.py

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class KalmanPairsTradingStrategy:
    """
    Complete pairs trading strategy using Kalman Filter.
    """
    
    def __init__(self,
                 delta: float = 1e-4,
                 ve: float = 1e-3,
                 zscore_window: int = 30,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss: float = 3.0):
        """
        Parameters:
        -----------
        delta : float
            Kalman Filter process variance
        ve : float
            Kalman Filter observation variance
        zscore_window : int
            Rolling window for Z-score calculation (30 days recommended)
        entry_threshold : float
            Z-score threshold to enter trade (default: 2.0)
        exit_threshold : float
            Z-score threshold to exit trade (default: 0.5)
        stop_loss : float
            Z-score stop-loss threshold (default: 3.0)
        """
        self.delta = delta
        self.ve = ve
        self.zscore_window = zscore_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        
        self.kf = None
        self.results = None
        
    def fit(self, prices_a: pd.Series, prices_b: pd.Series) -> pd.DataFrame:
        """
        Fit Kalman Filter to price data and calculate trading signals.
        
        Parameters:
        -----------
        prices_a : pd.Series
            Prices of stock A (independent variable)
        prices_b : pd.Series
            Prices of stock B (dependent variable)
            
        Returns:
        --------
        results : pd.DataFrame
            Complete results with signals and positions
        """
        # Ensure data is aligned
        assert len(prices_a) == len(prices_b), "Price series must have same length"
        
        # Convert to numpy arrays
        x = prices_a.values
        y = prices_b.values
        
        # Estimate initial hedge ratio
        initial_beta = estimate_initial_hedge_ratio(y, x, lookback=60)
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter(
            delta=self.delta,
            ve=self.ve,
            initial_state=initial_beta,
            initial_variance=1.0
        )
        
        # Run filter
        results = self.kf.filter_batch(y, x)
        results.index = prices_a.index
        
        # Add stock names for reference
        results['stock_a'] = x
        results['stock_b'] = y
        
        # Generate trading signals
        results = self._generate_signals(results)
        
        self.results = results
        return results
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Z-score.
        
        Signal logic:
        - zscore > entry_threshold: SHORT spread (short B, long A)
        - zscore < -entry_threshold: LONG spread (long B, short A)
        - |zscore| < exit_threshold: EXIT position
        - |zscore| > stop_loss: STOP LOSS
        """
        df = df.copy()
        
        # Initialize signals
        df['signal'] = 0
        df['position'] = 0
        
        # Current position tracker
        current_position = 0
        
        for i in range(len(df)):
            if pd.isna(df['zscore'].iloc[i]):
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'position'] = current_position
                continue
            
            zscore = df['zscore'].iloc[i]
            
            # Stop loss check
            if abs(zscore) > self.stop_loss:
                current_position = 0  # Exit on stop loss
                df.loc[df.index[i], 'signal'] = 0
            
            # Entry signals (if not in position)
            elif current_position == 0:
                if zscore > self.entry_threshold:
                    current_position = -1  # Short spread
                    df.loc[df.index[i], 'signal'] = -1
                elif zscore < -self.entry_threshold:
                    current_position = 1   # Long spread
                    df.loc[df.index[i], 'signal'] = 1
            
            # Exit signals (if in position)
            elif current_position != 0:
                if abs(zscore) < self.exit_threshold:
                    df.loc[df.index[i], 'signal'] = 0
                    current_position = 0  # Exit position
            
            df.loc[df.index[i], 'position'] = current_position
        
        return df
    
    def backtest(self, 
                 transaction_cost: float = 0.001,
                 initial_capital: float = 100000) -> Dict:
        """
        Backtest the strategy.
        
        Parameters:
        -----------
        transaction_cost : float
            Transaction cost as fraction (0.001 = 0.1%)
        initial_capital : float
            Initial capital in base currency
            
        Returns:
        --------
        metrics : dict
            Performance metrics
        """
        if self.results is None:
            raise ValueError("Must call fit() before backtest()")
        
        df = self.results.copy()
        
        # Calculate spread returns
        df['spread_return'] = df['spread'].pct_change()
        
        # Position changes (for transaction costs)
        df['position_change'] = df['position'].diff().abs()
        
        # Strategy returns
        df['strategy_return'] = df['position'].shift(1) * df['spread_return']
        
        # Apply transaction costs
        df['transaction_costs'] = df['position_change'] * transaction_cost
        df['strategy_return_net'] = df['strategy_return'] - df['transaction_costs']
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['strategy_return_net']).cumprod()
        df['equity_curve'] = initial_capital * df['cumulative_returns']
        
        # Calculate metrics
        metrics = self._calculate_metrics(df, initial_capital)
        
        # Update results
        self.results = df
        
        return metrics
    
    def _calculate_metrics(self, df: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate performance metrics"""
        
        # Basic metrics
        total_return = df['cumulative_returns'].iloc[-1] - 1
        
        # Annualized return (assuming 252 trading days)
        n_days = len(df)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        daily_vol = df['strategy_return_net'].std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum Drawdown
        running_max = df['equity_curve'].cummax()
        drawdown = (df['equity_curve'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (df['strategy_return_net'] > 0).sum()
        total_trading_days = (df['strategy_return_net'] != 0).sum()
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        # Number of trades
        num_trades = (df['position'].diff() != 0).sum()
        
        # Average holding period
        positions = df[df['position'] != 0].copy()
        if len(positions) > 0:
            position_changes = positions['position'].diff()
            trade_starts = positions[position_changes != 0].index
            if len(trade_starts) > 1:
                avg_holding_period = np.mean(np.diff(trade_starts)).days
            else:
                avg_holding_period = 0
        else:
            avg_holding_period = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'avg_holding_period_days': avg_holding_period,
            'final_equity': df['equity_curve'].iloc[-1]
        }
    
    def plot_results(self, figsize=(16, 12)):
        """
        Plot comprehensive results.
        """
        if self.results is None:
            raise ValueError("Must call fit() and backtest() first")
        
        df = self.results
        
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # 1. Price series and hedge ratio
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(df.index, df['stock_a'], label='Stock A', color='blue', alpha=0.7)
        ax1.plot(df.index, df['stock_b'], label='Stock B', color='red', alpha=0.7)
        ax1_twin.plot(df.index, df['hedge_ratio'], label='Hedge Ratio (β)', 
                     color='green', linestyle='--', linewidth=2)
        
        ax1.set_ylabel('Price', fontsize=12)
        ax1_twin.set_ylabel('Hedge Ratio', fontsize=12, color='green')
        ax1.set_title('Stock Prices and Dynamic Hedge Ratio', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spread and Z-score
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(df.index, df['spread'], label='Spread', color='purple', alpha=0.6)
        ax2.axhline(y=df['spread'].mean(), color='black', linestyle='--', 
                   alpha=0.5, label='Mean')
        
        ax2_twin.plot(df.index, df['zscore'], label='Z-Score', color='orange', linewidth=2)
        ax2_twin.axhline(y=self.entry_threshold, color='r', linestyle='--', 
                        alpha=0.5, label=f'Entry ±{self.entry_threshold}')
        ax2_twin.axhline(y=-self.entry_threshold, color='r', linestyle='--', alpha=0.5)
        ax2_twin.axhline(y=self.exit_threshold, color='g', linestyle='--', 
                        alpha=0.5, label=f'Exit ±{self.exit_threshold}')
        ax2_twin.axhline(y=-self.exit_threshold, color='g', linestyle='--', alpha=0.5)
        ax2_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_ylabel('Spread', fontsize=12)
        ax2_twin.set_ylabel('Z-Score', fontsize=12, color='orange')
        ax2.set_title('Spread and Z-Score', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Positions
        ax3 = axes[2]
        ax3.fill_between(df.index, 0, df['position'], where=(df['position'] > 0),
                        color='green', alpha=0.3, label='Long Spread')
        ax3.fill_between(df.index, 0, df['position'], where=(df['position'] < 0),
                        color='red', alpha=0.3, label='Short Spread')
        ax3.set_ylabel('Position', fontsize=12)
        ax3.set_title('Trading Positions', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Equity curve
        ax4 = axes[3]
        ax4.plot(df.index, df['equity_curve'], label='Strategy Equity', 
                color='darkgreen', linewidth=2)
        ax4.axhline(y=df['equity_curve'].iloc[0], color='black', 
                   linestyle='--', alpha=0.5, label='Initial Capital')
        ax4.fill_between(df.index, df['equity_curve'].iloc[0], df['equity_curve'],
                        where=(df['equity_curve'] >= df['equity_curve'].iloc[0]),
                        color='green', alpha=0.2)
        ax4.fill_between(df.index, df['equity_curve'].iloc[0], df['equity_curve'],
                        where=(df['equity_curve'] < df['equity_curve'].iloc[0]),
                        color='red', alpha=0.2)
        
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Equity ($)', fontsize=12)
        ax4.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kalman_pairs_trading_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_metrics(self, metrics: Dict):
        """Print performance metrics in formatted table"""
        print("\n" + "="*60)
        print("KALMAN FILTER PAIRS TRADING - PERFORMANCE METRICS")
        print("="*60)
        print(f"Total Return:              {metrics['total_return']:>12.2%}")
        print(f"Annualized Return:         {metrics['annualized_return']:>12.2%}")
        print(f"Annualized Volatility:     {metrics['annualized_volatility']:>12.2%}")
        print(f"Sharpe Ratio:              {metrics['sharpe_ratio']:>12.2f}")
        print(f"Maximum Drawdown:          {metrics['max_drawdown']:>12.2%}")
        print(f"Win Rate:                  {metrics['win_rate']:>12.2%}")
        print(f"Number of Trades:          {metrics['num_trades']:>12.0f}")
        print(f"Avg Holding Period (days): {metrics['avg_holding_period_days']:>12.1f}")
        print(f"Final Equity:              ${metrics['final_equity']:>11,.2f}")
        print("="*60 + "\n")