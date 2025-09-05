import logging
from mt5_connector import MT5Connector
from config import (
    ADDITIONAL_GRID,
    MAX_POSITIONS,
    Z_SCORE_ENTRY_THRESHOLD
)
import time

class TradeExecution:
    def __init__(self, magic_number):
        self.magic_number = magic_number
        self.mt5_conn = MT5Connector()
        self.logger = logging.getLogger(__name__)
    
    def execute_trade(self, symbolY, symbolX, slope, z_score, correlation):
        # Implement actual trade logic; replace with MetaTrader 5 API calls
        total_positions = self.mt5_conn.get_total_positions()
        grid_count = (total_positions/2) # Example grid size, adjust as needed
        updated_zscore_entry = 0.0

        day_profit,highest_zscore_period,total_profit = self.mt5_conn.total_daily_risk()
        if (abs(highest_zscore_period) > Z_SCORE_ENTRY_THRESHOLD):
              updated_zscore_entry = float(highest_zscore_period) + (grid_count)*ADDITIONAL_GRID
        elif highest_zscore_period == 0:
              updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + (grid_count)*ADDITIONAL_GRID
        
        self.logger.info(f"Updated grid z score entry : {updated_zscore_entry}")

         # Trading logic based on z-score and correlation
        if (total_positions < MAX_POSITIONS):
            if (correlation > 0):
                if (z_score < -updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY,symbolX,orders_type,slope,z_score)

                elif (z_score > updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY,symbolX,orders_type,slope,z_score)

            elif (correlation < 0):
                if (z_score < -updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY,symbolX,orders_type,slope,z_score)

                elif (z_score > updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY,symbolX,orders_type,slope,z_score)