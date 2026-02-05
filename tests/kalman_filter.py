# kalman_filter.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class KalmanFilter:
    """
    Kalman Filter for dynamic hedge ratio estimation in pairs trading.
    
    Based on state-space model:
    - Observation: y_t = β_t * x_t + ε_t
    - State: β_t = β_{t-1} + w_t
    """
    
    def __init__(self, 
                 delta: float = 1e-4,      # Process variance (how fast β can change)
                 ve: float = 1e-3,         # Observation variance (measurement noise)
                 initial_state: float = 0.0,
                 initial_variance: float = 1.0):
        """
        Parameters:
        -----------
        delta : float
            Process variance (V_w). Controls adaptation speed.
            - Higher delta (1e-3): Fast adaptation, more noise
            - Lower delta (1e-5): Slow adaptation, more stable
            Recommended: 1e-4 for daily data
            
        ve : float
            Observation variance (V_e). Measurement noise.
            Recommended: 1e-3
            
        initial_state : float
            Initial hedge ratio estimate (β_0)
            
        initial_variance : float
            Initial state variance (P_0)
        """
        self.delta = delta
        self.ve = ve
        
        # State variables
        self.state_mean = initial_state        # β_t
        self.state_variance = initial_variance # P_t
        
        # History for analysis
        self.state_means = []
        self.state_variances = []
        self.observations = []
        
    def update(self, observation: float, x: float) -> Tuple[float, float]:
        """
        Kalman Filter update step.
        
        Parameters:
        -----------
        observation : float
            Observed value of y_t (price of stock B)
        x : float
            Predictor value (price of stock A)
            
        Returns:
        --------
        state_mean : float
            Updated hedge ratio estimate (β_t)
        state_variance : float
            Updated state variance (P_t)
        """
        # Prediction step
        # β_t|t-1 = β_{t-1}
        predicted_state = self.state_mean
        
        # P_t|t-1 = P_{t-1} + V_w
        predicted_variance = self.state_variance + self.delta
        
        # Update step
        # Innovation (prediction error): e_t = y_t - β_t|t-1 * x_t
        innovation = observation - predicted_state * x
        
        # Innovation variance: S_t = x_t² * P_t|t-1 + V_e
        innovation_variance = x**2 * predicted_variance + self.ve
        
        # Kalman Gain: K_t = P_t|t-1 * x_t / S_t
        kalman_gain = predicted_variance * x / innovation_variance
        
        # Updated state estimate: β_t = β_t|t-1 + K_t * e_t
        self.state_mean = predicted_state + kalman_gain * innovation
        
        # Updated state variance: P_t = P_t|t-1 - K_t * x_t * P_t|t-1
        self.state_variance = predicted_variance - kalman_gain * x * predicted_variance
        
        # Store history
        self.state_means.append(self.state_mean)
        self.state_variances.append(self.state_variance)
        self.observations.append(observation)
        
        return self.state_mean, self.state_variance
    
    def filter_batch(self, y: np.ndarray, x: np.ndarray) -> pd.DataFrame:
        """
        Run Kalman Filter on batch of data.
        
        Parameters:
        -----------
        y : np.ndarray
            Prices of stock B (dependent variable)
        x : np.ndarray
            Prices of stock A (independent variable)
            
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with hedge ratios, spreads, and statistics
        """
        assert len(y) == len(x), "y and x must have same length"
        
        # Reset history
        self.state_means = []
        self.state_variances = []
        self.observations = []
        
        # Run filter
        for i in range(len(y)):
            self.update(y[i], x[i])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'y': y,
            'x': x,
            'hedge_ratio': self.state_means,
            'hedge_ratio_variance': self.state_variances,
            'spread': y - np.array(self.state_means) * x,
        })
        
        # Add spread statistics
        results['spread_mean'] = results['spread'].rolling(window=30).mean()
        results['spread_std'] = results['spread'].rolling(window=30).std()
        results['zscore'] = (results['spread'] - results['spread_mean']) / results['spread_std']
        
        return results
    
    def reset(self, initial_state: float = 0.0, initial_variance: float = 1.0):
        """Reset filter to initial conditions"""
        self.state_mean = initial_state
        self.state_variance = initial_variance
        self.state_means = []
        self.state_variances = []
        self.observations = []


def estimate_initial_hedge_ratio(y: np.ndarray, x: np.ndarray, 
                                  lookback: int = 60) -> float:
    """
    Estimate initial hedge ratio using OLS on recent data.
    
    Parameters:
    -----------
    y, x : np.ndarray
        Price series
    lookback : int
        Number of recent observations to use
        
    Returns:
    --------
    beta : float
        Initial hedge ratio estimate
    """
    recent_y = y[-lookback:]
    recent_x = x[-lookback:]
    
    # OLS: β = (X'X)^-1 X'Y
    beta = np.polyfit(recent_x, recent_y, 1)[0]
    
    return beta