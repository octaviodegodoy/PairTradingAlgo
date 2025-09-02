import logging
import numpy as np
import pandas as pd
from config import TRADING_PAIR_Y, TRADING_PAIR_X, Z_SCORE_ENTRY_THRESHOLD, MAX_HALF_LIFE
import time
from utils import Utils
import random

class PairTradingStrategy:
    def __init__(self, max_half_life, min_zscore):
        self.max_half_life = max_half_life
        self.utils = Utils()
        self.logger = logging.getLogger(__name__)


    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        arbitrage_found = False

        while True:
          
            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  rolling_z_scores, spreads, hedge_ratio, correlation = self.utils.get_dynamic_spread_zscores(pair_y[i], pair_x[j])
                  zscore_condition = abs(rolling_z_scores[-1]) > Z_SCORE_ENTRY_THRESHOLD
                  self.logger.info(f"Current Z-Score: {rolling_z_scores[-1]}, Z-Score Condition Met: {zscore_condition}") 
                  half_life = self.utils.get_half_life(spreads)
                  half_life_condition = half_life < MAX_HALF_LIFE
                  self.logger.info(f"Calculated Half-Life: {half_life}, Half-Life Condition Met: {half_life_condition}")
                  cointegration_condition = self.utils.check_cointegration(spreads)
                  self.logger.info(f"Cointegration Condition Met: {cointegration_condition}")
                  self.logger.info(f"Correlation between {pair_y[i]} and {pair_x[j]}: {correlation}")
                  arbitrage_found = zscore_condition and half_life_condition and cointegration_condition
                  
                  

            if not arbitrage_found:
             self.logger.info(f"Arbitrage not yet found, continuing scan...")
             time.sleep(5)
             continue
            break
        self.logger.info(f"Arbitrage opportunity found between {pair_y[i]} and {pair_x[j]}!")
        return hedge_ratio, spreads, rolling_z_scores, arbitrage_found    