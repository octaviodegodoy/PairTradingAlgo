import logging
from broker_connector import BrokerConnector
from risk_manager import RiskManager
from utils import calculate_volumes, updates_zscore_entry
from constants import (
    MAX_GRIDS,
    Z_SCORE_ENTRY_THRESHOLD
)
import time

class TradeExecution:
    def __init__(self, magic_number, connector: BrokerConnector):
        self.magic_number = magic_number
        self.mt5_conn = connector
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)
    
    def execute_trade(self, symbolY, symbolX, slope, hedge_ratio, z_score):
        total_positions = self.mt5_conn.get_total_positions()
        grid_count = (total_positions / 2)
        equity = self.mt5_conn.get_account_info().equity
        total_max_lots = self.risk_manager.max_lots(equity)

        highest_zscore_period, total_profit, total_traded_volumes, grid_count_history = self.mt5_conn.total_daily_risk()
        self.logger.info(f"Highest Z-Score Period: {highest_zscore_period} total volumes {total_traded_volumes} and max lots {total_max_lots}")
        updated_zscore_entry = updates_zscore_entry(highest_zscore_period, total_profit, total_traded_volumes, grid_count_history, grid_count)

        self.logger.info(f"Max volume : {total_max_lots} open positions volume {total_traded_volumes} current zscore {z_score} updated zscore entry {updated_zscore_entry}")
        min_lot_Y = self.mt5_conn.get_symbol_info(symbolY).volume_min
        min_lot_X = self.mt5_conn.get_symbol_info(symbolX).volume_min
        volumeY, volume_X = calculate_volumes(symbolY, symbolX, hedge_ratio, min_lot_Y, min_lot_X, total_max_lots, total_positions)
        self.logger.info(f"Calculated volumes - {symbolY}: {volumeY}, {symbolX}: {volume_X}")

        # Risk per individual leg: split session budget evenly across all grids and both legs
        risk_per_leg = self.risk_manager.max_loss(equity) / (MAX_GRIDS * 2)
        self.logger.info(f"Risk per leg (proportional SL budget): {risk_per_leg:.2f}")

        if (total_traded_volumes < total_max_lots) and grid_count < MAX_GRIDS:
            self.logger.info(f"Sending order: z_score={z_score}, threshold={updated_zscore_entry}, slope={slope}")
            if slope > 0:
                if z_score < -updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_SELL]
                    sl_y, sl_x = self._compute_leg_sls(symbolY, symbolX, orders_type, volumeY, volume_X, risk_per_leg)
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score, sl_y, sl_x)
                elif z_score > updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_BUY]
                    sl_y, sl_x = self._compute_leg_sls(symbolY, symbolX, orders_type, volumeY, volume_X, risk_per_leg)
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score, sl_y, sl_x)
            elif slope < 0:
                if z_score < -updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_BUY]
                    sl_y, sl_x = self._compute_leg_sls(symbolY, symbolX, orders_type, volumeY, volume_X, risk_per_leg)
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score, sl_y, sl_x)
                elif z_score > updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_SELL]
                    sl_y, sl_x = self._compute_leg_sls(symbolY, symbolX, orders_type, volumeY, volume_X, risk_per_leg)
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score, sl_y, sl_x)

    def _compute_leg_sls(self, symbolY, symbolX, orders_type, volumeY, volumeX, risk_per_leg):
        """Fetch current market prices and compute a proportional SL price for each leg."""
        tick_y = self.mt5_conn.get_symbol_tick(symbolY)
        info_y = self.mt5_conn.get_symbol_info(symbolY)
        price_y = tick_y.bid if orders_type[0] == self.mt5_conn.ORDER_TYPE_SELL else tick_y.ask
        sl_y = self.risk_manager.calc_sl_price(
            orders_type[0], price_y, volumeY, risk_per_leg,
            info_y.trade_tick_value, info_y.trade_tick_size, info_y.digits,
            order_type_buy=self.mt5_conn.ORDER_TYPE_BUY,
            stops_level=info_y.trade_stops_level,
        )

        tick_x = self.mt5_conn.get_symbol_tick(symbolX)
        info_x = self.mt5_conn.get_symbol_info(symbolX)
        price_x = tick_x.ask if orders_type[1] == self.mt5_conn.ORDER_TYPE_BUY else tick_x.bid
        sl_x = self.risk_manager.calc_sl_price(
            orders_type[1], price_x, volumeX, risk_per_leg,
            info_x.trade_tick_value, info_x.trade_tick_size, info_x.digits,
            order_type_buy=self.mt5_conn.ORDER_TYPE_BUY,
            stops_level=info_x.trade_stops_level,
        )
        return sl_y, sl_x