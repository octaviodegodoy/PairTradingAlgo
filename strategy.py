import logging
import math
import numpy as np
import pandas as pd
from constants import MARGIN_PERCENT, MAX_RISK, PROFIT_THRESHOLD, TRADING_PAIR_Y, TRADING_PAIR_X, MAX_HALF_LIFE, Z_SCORE_ENTRY_THRESHOLD
import time
import random
from mt5_connector import MT5Connector
from utils import check_cointegration, get_dynamic_spread_zscores, get_half_life, check_trading_time, get_linear_regression_spread_zscores

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
        highest_zscore_period,total_profit,total_volume = self.mt5_conn.total_daily_risk()
        if (abs(highest_zscore_period) > Z_SCORE_ENTRY_THRESHOLD):
              updated_zscore_entry = float(highest_zscore_period)
        self.logger.info(f"Updated Z score entry : {updated_zscore_entry} total volumes {total_volume} ")
     
        while True:

            self.logger.info(f"Highest Z-Score Period: {highest_zscore_period}, Total Profit: {total_profit}, Total Volume traded : {total_volume}")
            total_volume_risk = self.mt5_conn.get_max_lots()
            self.logger.info(f"Total volume risk allowed : {total_volume_risk}")
            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  assets_y = self.mt5_conn.get_data_futures(pair_y[i])
                  assets_x = self.mt5_conn.get_data_futures(pair_x[j])
                  #rolling_z_scores, spreads, hedge_ratio, correlation = get_dynamic_spread_zscores(assets_y, assets_x)
                  rolling_z_scores, spreads, hedge_ratio, correlation = get_linear_regression_spread_zscores(assets_y, assets_x)
                  zscore_condition = abs(rolling_z_scores.iloc[-1]) > updated_zscore_entry
                  ratio = abs(hedge_ratio)
                  # Calculate volumes based on hedge ratio
                  investment_asset_y = math.floor(total_volume_risk/(1 + ratio))
                  investment_asset_x = math.floor(total_volume_risk - investment_asset_y)
                  self.logger.info(f"Current Z-Score: {rolling_z_scores.iloc[-1]} hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x} for minimum {updated_zscore_entry} Z-Score Condition Met: {zscore_condition}")
                  current_equity = self.mt5_conn.get_account_info().equity
                  # Calculate risk parameters
                  total_margin = current_equity*MARGIN_PERCENT
                  max_loss = total_margin*MAX_RISK
                  trailing_start = max_loss*PROFIT_THRESHOLD
                  self.logger.info(f"Max loss: {max_loss}, trailing start profit: {trailing_start}")

                  half_life = get_half_life(spreads)
                  half_life_condition = half_life < MAX_HALF_LIFE
                  self.logger.info(f"Calculated Half-Life: {half_life}, Half-Life Condition Met: {half_life_condition}")
                  cointegration_condition = True #check_cointegration(spreads)
                  self.logger.info(f"Cointegration Condition Met: {cointegration_condition}")
                  self.logger.info(f"Correlation between {pair_y[i]} and {pair_x[j]}: {correlation}")
                  arbitrage_found = zscore_condition and half_life_condition and cointegration_condition
                  print(f"Arbitrage Found: {arbitrage_found}")
                  time.sleep(15)
                  if arbitrage_found:
                     y = self.mt5_conn.get_symbol_futures(pair_y[i])
                     x = self.mt5_conn.get_symbol_futures(pair_x[j])
                     pair = (y[1], x[1])
                     self.logger.info(f"Arbitrage conditions met for pair: {pair}")
                     return hedge_ratio, spreads, rolling_z_scores, pair, correlation, arbitrage_found         
                  
            if not check_trading_time():
             self.logger.info(f"Outside trading hours, stopping scan.")
             break
            elif not arbitrage_found:
             time.sleep(15)
             continue

        return hedge_ratio, spreads, rolling_z_scores, pair, correlation, arbitrage_found