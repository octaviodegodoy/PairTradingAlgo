import logging
import numpy as np
import pandas as pd
from constants import TRADING_PAIR_Y, TRADING_PAIR_X, MAX_HALF_LIFE, Z_SCORE_ENTRY_THRESHOLD
import time
import random
from mt5_connector import MT5Connector
from utils import check_cointegration, get_dynamic_spread_zscores, get_half_life

class PairTradingStrategy:
    def __init__(self):
        self.mt5_conn = MT5Connector()
        self.logger = logging.getLogger(__name__)

    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD
        arbitrage_found = False
        pair = []
        total_positions = self.mt5_conn.get_total_positions() #self.mt5_conn.positions_total()
        self.logger.info(f"Total current positions: {total_positions}")
        if total_positions > 0:
            self.logger.info("Existing positions detected, skipping new pair scanning.")
            return None, None, None, None, None, arbitrage_found
        
        ## Get daily profit and highest z score period
        day_profit,highest_zscore_period,total_profit = self.mt5_conn.total_daily_risk()
        if (abs(highest_zscore_period) > Z_SCORE_ENTRY_THRESHOLD):
              updated_zscore_entry = float(highest_zscore_period)
        self.logger.info(f"Updated Z score entry : {updated_zscore_entry}")

     
        while True:
            
            self.logger.info(f"Day profit: {day_profit}, Highest Z-Score Period: {highest_zscore_period}, Total Profit: {total_profit}")            

            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  assets_y = self.mt5_conn.get_data_futures(pair_y[i])
                  assets_x = self.mt5_conn.get_data_futures(pair_x[j])
                  rolling_z_scores, spreads, hedge_ratio, correlation = get_dynamic_spread_zscores(assets_y, assets_x)
                  zscore_condition = abs(rolling_z_scores[-1]) > updated_zscore_entry
                  self.logger.info(f"Current Z-Score: {rolling_z_scores[-1]}, for minimum {updated_zscore_entry} Z-Score Condition Met: {zscore_condition}") 
                  half_life = get_half_life(spreads)
                  half_life_condition = half_life < MAX_HALF_LIFE
                  self.logger.info(f"Calculated Half-Life: {half_life}, Half-Life Condition Met: {half_life_condition}")
                  cointegration_condition = check_cointegration(spreads)
                  self.logger.info(f"Cointegration Condition Met: {cointegration_condition}")
                  self.logger.info(f"Correlation between {pair_y[i]} and {pair_x[j]}: {correlation}")
                  arbitrage_found = zscore_condition and half_life_condition and cointegration_condition
                  if arbitrage_found:
                     y = self.mt5_conn.get_symbol_futures(pair_y[i])
                     x = self.mt5_conn.get_symbol_futures(pair_x[j])
                     pair = (y[1], x[1])
                     self.logger.info(f"Arbitrage conditions met for pair: {pair}")
                     break         
                  

            if not arbitrage_found:
             self.logger.info(f"Arbitrage not yet found, continuing scan...")
             time.sleep(5)
             continue
            break

        self.logger.info(f"Arbitrage opportunity found between {pair[0]} and {pair[1]}!")
        return hedge_ratio, spreads, rolling_z_scores, pair, correlation,arbitrage_found    