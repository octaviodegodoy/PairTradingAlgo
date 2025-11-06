import logging
from mt5_connector import MT5Connector
from utils import calculate_volumes
from constants import (
    ADDITIONAL_GRID,
    MARGIN_PERCENT,
    MARGIN_X,
    MARGIN_Y,
    MAX_GRIDS,
    Z_SCORE_ENTRY_THRESHOLD
)
import time

class TradeExecution:
    def __init__(self, magic_number):
        self.magic_number = magic_number
        self.mt5_conn = MT5Connector()
        self.logger = logging.getLogger(__name__)
    
    def execute_trade(self, symbolY, symbolX, slope, z_score):
        # Implement actual trade logic; replace with MetaTrader 5 API calls
        total_positions = self.mt5_conn.get_total_positions()
        grid_count = (total_positions/2) # Example grid size, adjust as needed
        updated_zscore_entry = 0.0
        total_max_lots = self.mt5_conn.get_max_lots()
                       
         ## Get daily profit and highest z score period

        highest_zscore_period,total_profit,total_traded_volumes = self.mt5_conn.total_daily_risk()
        self.logger.info(f"Highest Z-Score Period: {highest_zscore_period} total volumes {total_traded_volumes} and max lots {total_max_lots}")
        if (abs(highest_zscore_period) > Z_SCORE_ENTRY_THRESHOLD):
              updated_zscore_entry = float(highest_zscore_period) + (grid_count)*ADDITIONAL_GRID
        elif highest_zscore_period == 0:
              updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + (grid_count)*ADDITIONAL_GRID
        
        self.logger.info(f"Max volume : {total_max_lots} and open positions volume {total_traded_volumes} current zscore {z_score} updated zscore entry {updated_zscore_entry}  ")
        min_lot_Y = self.mt5_conn.get_symbol_info(symbolY).volume_min
        min_lot_X = self.mt5_conn.get_symbol_info(symbolX).volume_min
        volumeY, volume_X = calculate_volumes(symbolY,symbolX,slope,min_lot_Y,min_lot_X,total_max_lots,total_positions)
        self.logger.info(f"Calculated volumes - {symbolY}: {volumeY}, {symbolX}: {volume_X}")
        
        self.logger.info(f"Total lots volume after calculation: {total_traded_volumes} and max lots {total_max_lots}")
        
         # Trading logic based on z-score and correlation
        if (total_traded_volumes < total_max_lots) and grid_count < MAX_GRIDS:
            self.logger.info(f"Sending order with: current z-score {z_score} and updated zscore entry {updated_zscore_entry} and correlation {correlation}")
            if (slope > 0):
                if (z_score < -updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY,symbolX,volumeY,volume_X,orders_type,z_score)

                elif (z_score > updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY,symbolX,volumeY,volume_X,orders_type,z_score)

            elif (slope < 0):
                if (z_score < -updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY,symbolX,volumeY,volume_X,orders_type,z_score)

                elif (z_score > updated_zscore_entry):
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY,symbolX,volumeY,volume_X,orders_type,z_score)