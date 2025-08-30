import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta, timezone
import time
import pandas as pd

class MT5Connector:
    
    POSITION_TYPE_BUY = mt5.POSITION_TYPE_BUY
    POSITION_TYPE_SELL = mt5.POSITION_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol, timeframe, n, start):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, start, n)
        if rates is None:
            self.logger.error(f"Could not get rates for {symbol}")
            return None
        else:
            self.logger.info(f"Retrieved {len(rates)} rates for {symbol}")
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
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