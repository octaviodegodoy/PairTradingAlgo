import logging
import numpy as np
import pandas as pd
from config import TRADING_PAIR_Y, TRADING_PAIR_X, PERIODS, SHIFT_PERIODS
from mt5_connector import MT5Connector
import time

class PairTradingStrategy:
    def __init__(self, max_half_life, min_zscore):
        self.max_half_life = max_half_life
        self.min_zscore = min_zscore
        self.logger = logging.getLogger(__name__)


    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        mt5_conn = MT5Connector()
        test_counter = 0
        arbitrage_found = False

        while True:
            test_counter += 1
            print(f"Selecting pairs with min z score {self.min_zscore} and half life {self.max_half_life} ")

            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  prices_data_y = mt5_conn.get_data(pair_y[i], mt5_conn.TIMEFRAME_D1, PERIODS, SHIFT_PERIODS).dropna()
                  prices_data_x = mt5_conn.get_data(pair_x[j], mt5_conn.TIMEFRAME_D1, PERIODS, SHIFT_PERIODS).dropna()
                  self.logger.info(f"Retrieved {len(prices_data_y)} data points for {pair_y[i]} and {len(prices_data_x)} for {pair_x[j]}")
                  self.logger.info(f"Arbitrage not yet found, continuing scan...")
                  if not arbitrage_found:
                     continue
                  self.logger.info(f"Loop counter {test_counter}")
                  if test_counter > 5:
                    self.logger.info("Arbitrage found leaving loop.")
                    arbitrage_found = True
                    break
              #self.logger.info(f"Last price y: {prices_data_y[-1]} and last price x {prices_data_x}")

            return arbitrage_found    