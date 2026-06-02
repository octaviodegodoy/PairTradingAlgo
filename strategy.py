import logging
import pandas as pd
from constants import KALMAN_FILTER_METHOD, TRADING_PAIR_Y, TRADING_PAIR_X, MAX_HALF_LIFE, Z_SCORE_ENTRY_THRESHOLD, VECM_ECT_THRESHOLD, HURST_THRESHOLD, SCAN_COINTEGRATION_METHOD, SCAN_JOHANSEN_CRIT_LEVEL, OU_LAMBDA_MIN, JOHANSEN_PERIODS, MAGIC_NUMBER
import time
from broker_connector import BrokerConnector
from utils import check_cointegration, get_correlation, get_half_life, check_trading_time, get_linear_regression_spread_zscores, updates_zscore_entry, get_dynamic_spread_zscores, get_vecm_ect_zscore, get_hurst_exponent, get_ou_params

class PairTradingStrategy:
    def __init__(self, connector: BrokerConnector):
        self.mt5_conn = connector
        self.logger = logging.getLogger(__name__)

    def scan_pairs_arbitrage(self):
        pair_y = TRADING_PAIR_Y
        pair_x = TRADING_PAIR_X
        updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD
        arbitrage_found = False
        pair = []
        # Initialise scan_results so the variable is always bound even when no
        # pair has been evaluated yet (avoids NameError in the while-loop guard).
        scan_results = {'arbitrage_found': False}
        total_positions = self.mt5_conn.get_total_positions() #self.mt5_conn.positions_total()
        self.logger.info(f"Total current positions: {total_positions}")
        if total_positions > 0:
            self.logger.info("Existing positions detected, skipping new pair scanning.")
            return None, None, None, None, None, arbitrage_found
        
        ## Get daily profit and highest z score period
        highest_zscore_period,total_profit,total_volume,grid_history = self.mt5_conn.total_daily_risk()
        grids_count = total_positions/2
        updated_zscore_entry = updates_zscore_entry(highest_zscore_period,total_profit,total_volume,grid_history,grids_count,MAGIC_NUMBER)
        self.logger.info(f"Updated Z score entry : {updated_zscore_entry} total volumes {total_volume} grids history is {grid_history} how is grids count {grids_count} and total profit {total_profit}")
     
        while True:

            self.logger.info(f"Highest Z-Score Period: {highest_zscore_period}, Total Profit: {total_profit}, Total Volume traded : {total_volume}")
            for i in range(len(pair_y)):
              for j in range(len(pair_x)):
                  self.logger.info(f"Scanning pairs: {pair_y[i]} and {pair_x[j]}")
                  assets_y = self.mt5_conn.get_data_futures_btg(pair_y[i])
                  assets_x = self.mt5_conn.get_data_futures_btg(pair_x[j])
                  correlation = get_correlation(assets_y,assets_x)
                  self.logger.info(f"Correlation between {pair_y[i]} and {pair_x[j]} is {correlation}")
                  if KALMAN_FILTER_METHOD:
                     results = get_dynamic_spread_zscores(assets_y, assets_x)
                  else:
                     results = get_linear_regression_spread_zscores(assets_y, assets_x)
                  
                  self.logger.info(f"Z score {results['z_scores'].iloc[-1]} for pair {pair_y[i]} and {pair_x[j]}")
                  zscore_condition = abs(results['z_scores'].iloc[-1]) > updated_zscore_entry
                  ratio = results['hedge_ratio'].iloc[-1]
                  print(f"Z-Score Condition Met: {zscore_condition} and hedge ratio is {ratio}")

                  half_life = get_half_life(results['spread'])
                  half_life_condition = half_life < MAX_HALF_LIFE
                  self.logger.info(f"Calculated Half-Life: {half_life}, Half-Life Condition Met: {half_life_condition}")
                  # Johansen needs more observations than the Kalman/z-score window;
                  # fetch a longer series (JOHANSEN_PERIODS daily bars) specifically for the test.
                  if SCAN_COINTEGRATION_METHOD.lower() == "johansen":
                      coint_y = self.mt5_conn.get_data_futures_btg(pair_y[i], n_bars=JOHANSEN_PERIODS)
                      coint_x = self.mt5_conn.get_data_futures_btg(pair_x[j], n_bars=JOHANSEN_PERIODS)
                  else:
                      coint_y = assets_y
                      coint_x = assets_x
                  cointegration_condition = check_cointegration(
                      coint_y,
                      coint_x,
                      method_override=SCAN_COINTEGRATION_METHOD,
                      johansen_crit_level_override=SCAN_JOHANSEN_CRIT_LEVEL,
                  )

                  vecm_result = get_vecm_ect_zscore(assets_y, assets_x)
                  vecm_ect_zscore = vecm_result["ect_zscore"]
                  vecm_alpha      = vecm_result["alpha"]
                  vecm_alpha_valid = vecm_result["alpha_valid"]
                  vecm_condition = abs(vecm_ect_zscore) >= VECM_ECT_THRESHOLD

                  hurst = get_hurst_exponent(results['spread'].values)
                  hurst_condition = hurst < HURST_THRESHOLD

                  ou = get_ou_params(results['spread'].values)
                  ou_condition = ou['is_mean_reverting']

                  self.logger.info(f"Cointegration Condition Met: {cointegration_condition}")
                  self.logger.info(
                      f"VECM ECT Z-Score: {vecm_ect_zscore:.4f}, VECM Condition Met: {vecm_condition} | "
                      f"alpha_y={vecm_alpha[0]:.6f}, alpha_x={vecm_alpha[1]:.6f}, alpha_valid={vecm_alpha_valid}"
                  )
                  self.logger.info(f"Hurst Exponent: {hurst:.4f}, Hurst Condition Met: {hurst_condition}")
                  self.logger.info(f"OU λ={ou['lambda_']:.6f}, μ={ou['mu']:.6f}, σ={ou['sigma']:.6f}, OU Condition Met: {ou_condition}")
                  self.logger.info(f"Hedge ratio between {pair_y[i]} and {pair_x[j]}: {results['hedge_ratio'].iloc[-1]} and z score is {results['z_scores'].iloc[-1]} and spread is {results['spread'].iloc[-1]} for threshold {updated_zscore_entry}")
                  arbitrage_found = zscore_condition and half_life_condition and cointegration_condition and vecm_condition and hurst_condition and ou_condition
                  print(f"Arbitrage Found: {arbitrage_found}")
                  scan_results = {
                      'pair_y': pair_y[i],
                      'pair_x': pair_x[j],
                      'correlation': correlation,
                      'hedge_ratio': results['hedge_ratio'].iloc[-1],
                      'spread': results['spread'].iloc[-1],
                      'z_score': results['z_scores'].iloc[-1],
                      'arbitrage_found': arbitrage_found
                    }
                  time.sleep(15)

                  if scan_results['arbitrage_found']:
                     y = self.mt5_conn.get_symbol_futures(pair_y[i])
                     x = self.mt5_conn.get_symbol_futures(pair_x[j])
                     pair = (y[1], x[1])
                     self.logger.info(f"Arbitrage conditions met for pair: {pair}")
                     return correlation, results['hedge_ratio'].iloc[-1], results['spread'].iloc[-1], results['z_scores'].iloc[-1], pair, scan_results['arbitrage_found']         
                  
            if not check_trading_time():
             self.logger.info(f"Outside trading hours, stopping scan.")
             break
            elif not scan_results['arbitrage_found']:
             time.sleep(15)
             continue

        # Return scalars (consistent with the early-return success path above).
        # If no pair was ever evaluated, results is unbound; return safe defaults.
        if 'results' not in dir() or results is None:
            return None, None, None, None, pair, False
        return correlation, results['hedge_ratio'].iloc[-1], results['spread'].iloc[-1], results['z_scores'].iloc[-1], pair, scan_results['arbitrage_found']