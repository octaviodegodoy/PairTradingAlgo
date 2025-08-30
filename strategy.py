import logging
import numpy as np
import pandas as pd
from config import TRADING_PAIR_Y, TRADING_PAIR_X, PERIODS, SHIFT_PERIODS
from mt5_connector import MT5Connector

class PairTradingStrategy:
    def __init__(self, max_half_life, min_zscore):
        self.max_half_life = max_half_life
        self.min_zscore = min_zscore
        self.logger = logging.getLogger(__name__)


    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        mt5_conn = MT5Connector()

        print(f"Selecting pairs with min z score {self.min_zscore} and half life {self.max_half_life} ")

        for i in range(len(pair_y)):
          for j in range(len(pair_x)):
            self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
            prices_data_y = mt5_conn.get_data(pair_y[i], mt5_conn.TIMEFRAME_D1, PERIODS, SHIFT_PERIODS).dropna()
            prices_data_x = mt5_conn.get_data(pair_x[j], mt5_conn.TIMEFRAME_D1, PERIODS, SHIFT_PERIODS).dropna()
            self.logger.info(f"Last price y: {prices_data_y} and last price x {prices_data_x}")