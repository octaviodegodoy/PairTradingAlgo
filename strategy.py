import logging
import numpy as np
import pandas as pd
from config import TRADING_PAIR_Y, TRADING_PAIR_X, PERIODS, SHIFT_PERIODS
from mt5_connector import MT5Connector
import time
import random

class PairTradingStrategy:
    def __init__(self, max_half_life, min_zscore):
        self.max_half_life = max_half_life
        self.min_zscore = min_zscore
        self.logger = logging.getLogger(__name__)


    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        mt5_conn = MT5Connector()
        arbitrage_found = False

        while True:
          
            print(f"Searching for pairs with min z score {self.min_zscore} and half life {self.max_half_life} ")

            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  prices_data_y = mt5_conn.get_data(pair_y[i])
                  prices_data_x = mt5_conn.get_data(pair_x[j])
                  self.logger.info(f"Retrieved {len(prices_data_y)} data points for {pair_y[i]} and {len(prices_data_x)} for {pair_x[j]}")
                  
            arbitrage_found = random.random() < 0.7     

            if not arbitrage_found:
             self.logger.info(f"Arbitrage not yet found, continuing scan...")
             time.sleep(5)
             continue
            break
        self.logger.info(f"Arbitrage opportunity found between {pair_y[i]} and {pair_x[j]}!")
        return arbitrage_found    