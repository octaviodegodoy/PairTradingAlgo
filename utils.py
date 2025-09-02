from datetime import datetime, timezone, timedelta
from config import START_TIME_HOUR,START_TIME_MINUTE,TRADE_WINDOW_TIME_HOURS,TRADE_WINDOW_TIME_MINUTES
from filterpy.kalman import KalmanFilter
import numpy as np
import logging
from mt5_connector import MT5Connector

class Utils:
    def __init__(self):
        mt5_conn = MT5Connector()
        self.logger = logging.getLogger(__name__)

    def check_trading_time():
        now = datetime.now(timezone.utc)
        trading_time_start = now.replace(hour=START_TIME_HOUR, minute=START_TIME_MINUTE, second=0, microsecond=0)
        trading_time_end = trading_time_start + timedelta(hours=TRADE_WINDOW_TIME_HOURS, minutes=TRADE_WINDOW_TIME_MINUTES)
        return trading_time_start <= now < trading_time_end

    def run_kalman_filter_momentum(y, x, NOISE_VARIANCE=0.001):
        kf = KalmanFilter(dim_x=2, dim_z=1)  # State: [slope, intercept], Measurement: y
        # Define state transition matrix (random walk for slope and intercept)
        kf.F = np.array([[1, 0],  # Slope stays constant (random walk)
                     [0, 1]]) # Intercept stays constant (random walk)
    
        # Initial state and covariance
        kf.x = np.array([0.5, 0])  # Initial guess: slope = 0.5, intercept = 0
        kf.P *= 0.01  # Initial uncertainty
        kf.R = NOISE_VARIANCE   # Measurement noise variance
        kf.Q = np.eye(2) * 1e-5  # Process noise (small for slow evolution)


    def get_dynamic_spread(self,symbolY,symbolX):
        asset1_prices = self.mt5_conn.get_data(symbolY)
        asset2_prices = self.mt5_conn.get_data(symbolX)

        # Get data series in the period and sort by date

        asset_x = get_symbols_futures("WDO*")
        asset_y = get_symbols_futures("WIN*")

        last_y_price = get_data(asset_y, mt5.TIMEFRAME_D1, 1, start).dropna()
        last_x_price = get_data(asset_x, mt5.TIMEFRAME_D1, 1, start).dropna()

        asset1_prices = pd.concat([asset1_prices, last_y_price])
        asset2_prices = pd.concat([asset2_prices, last_x_price])
        # Drop duplicate indices (keep the last occurrence)
        asset1_prices = asset1_prices[~asset1_prices.index.duplicated(keep='last')]
        asset2_prices = asset2_prices[~asset2_prices.index.duplicated(keep='last')]

        asset1_prices = asset1_prices.sort_index()
        asset2_prices = asset2_prices.sort_index()  
   
        # Calculate the Pearson correlation coefficient between the two assets
        correlation = asset1_prices['close'].corr(asset2_prices['close'])
        dates = asset1_prices['time']
        asset1_prices = np.array(asset1_prices['close'])
        asset2_prices = np.array(asset2_prices['close'])
         
        # Log-transform the prices
        log_asset1 = np.log(asset1_prices)
        log_asset2 = np.log(asset2_prices)

