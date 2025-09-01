import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from config import PERIODS, SHIFT_PERIODS

class MT5Connector:
    
    POSITION_TYPE_BUY = mt5.POSITION_TYPE_BUY
    POSITION_TYPE_SELL = mt5.POSITION_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, SHIFT_PERIODS, PERIODS)
        if rates is None:
            self.logger.error(f"Could not get rates for {symbol}")
            return None
        else:
            df = pd.DataFrame(rates)
            df = df.dropna()
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Filter out weekends (keep only weekdays)
            df = df[df['time'].dt.weekday < 5]  # 0=Monday, ..., 4=Friday
            return df
    
    def positions_get(self):
        positions = mt5.positions_get()
        return len(positions) if positions else 0

    def get_open_positions_count(self):
        positions = mt5.positions_get()
        return len(positions) if positions else 0
    
    def last_error():
        last_error = mt5.last_error()
        return last_error

    def sleep(self, seconds):
        time.sleep(seconds)

    def initialize(self):
        return mt5.initialize()

    def shutdown(self):
        mt5.shutdown()