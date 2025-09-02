import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
from config import PERIODS, SHIFT_PERIODS, UNIX_DAY

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
        current_symbol = list(sorted_next_futures.items())[0]
        from_time = time_now - UNIX_DAY * PERIODS
        fut_history_symbols = [current_symbol[1]]

        for t in range(len(sorted_past_futures)):
            past_symbol = list(sorted_past_futures.items())[-(t+1)]
            fut_history_symbols.append(past_symbol[1])
            if past_symbol[0] < from_time:
                break          

        dataframes = []

        for fut_symbol in fut_history_symbols:
            rates = mt5.copy_rates_from_pos(fut_symbol, mt5.TIMEFRAME_D1, SHIFT_PERIODS, PERIODS)
            if rates is not None and len(rates) > 0:
               df = pd.DataFrame(rates)
               df['symbol'] = symbol  # Optionally keep track of the symbol
               dataframes.append(df)

        # Concatenate all DataFrames
        if dataframes:
            all_data = pd.concat(dataframes, ignore_index=True)
            # Remove duplicates based on the index (in this case, you might want to remove by 'time' or another column)
            all_data = all_data.drop_duplicates(subset=['time', 'symbol'])  # Adjust subset as needed
        else:
            print("No data retrieved.")

        # If 'time' is not already datetime, convert it
        all_data['time'] = pd.to_datetime(all_data['time'], unit='s')  # or remove unit if already datetime
        # Sort by date (most recent last)
        all_data = all_data.sort_values('time')
        futrues_data = all_data[all_data['time'].dt.weekday < 5].tail(PERIODS)
        return futrues_data   
            
        
    def get_symbol_futures(self,group_name):
        futures_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        next_symbols_fut = {}
        past_symbols_fut = {}
        for s in futures_symbols:
            if s.expiration_time > time_now:
               next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time < time_now:
               past_symbols_fut[s.expiration_time] = s.name
        
        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        current_symbol = list(sorted_next_futures.items())[0]

        return current_symbol 
    
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