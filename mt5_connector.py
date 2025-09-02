import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from config import PERIODS, SHIFT_PERIODS
import math

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
            print(f"Data fetched for {symbol}: {df.head()}")
            self.logger.info(f"Retrieved {len(df)} data points for {symbol} and {PERIODS} periods ")
            return df
    
    def get_data_futures(self, symbol):
        futures_symbols = mt5.symbols_get(symbol)
        time_now = int(time.time())
        next_symbols_fut = {}
        past_symbols_fut = {}
        for s in futures_symbols:
            if s.expiration_time > time_now:
               next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time < time_now:
               past_symbols_fut[s.expiration_time] = s.name
        
        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        sorted_past_futures = dict(sorted(past_symbols_fut.items()))

        # Find the index of the current key
        last_item = list(sorted_past_futures.items())[-1]
        current_item = list(sorted_next_futures.items())[0]
        data_prices = self.get_data(current_item[1])
        prices_len = len(data_prices)
        count_prices_range = prices_len
        k = 0
        while count_prices_range < PERIODS:
            self.logger.info(f"Not enough data points ({prices_len}) for {current_item[1]}, need {PERIODS}. Fetching more from past futures...")
            k=k-1
            needed_symbol = list(sorted_past_futures.items())[k]
            data_prices_needed = self.get_data(needed_symbol[1])
            count_prices_range += len(data_prices_needed)

            self.logger.info(f"Fetching data from past future: {needed_symbol[1]}")    

        ranges_factor = PERIODS/prices_len
        
        total_ranges = int(math.ceil(ranges_factor))

        symbol_concat = [current_item[1]]
        for i in range(-1,-total_ranges,-1):
            needed_symbol = list(sorted_past_futures.items())[i]
            range_prices = self.get_data(needed_symbol[1])
            prices_len += len(range_prices)
            print(f"Needed symbol {needed_symbol[1]} with prices len {prices_len}")
            symbol_concat.append(needed_symbol[1]) 
            if prices_len >= PERIODS:
                break                 
        
        self.logger.info(f"Current future symbol: {symbol_concat}")        
            
        
    def get_symbols_futures(self,group_name):
        # get symbols containing RU in their names 
        futures_symbols = mt5.symbols_get(group_name)
        future_symbol = None
        for s in futures_symbols:
            if "Vencimento" in s.description:  #[5:3] 
                symbol = s.description.split('-')
                future_symbol = symbol[1][17:23]
                break 
        return future_symbol 
    
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