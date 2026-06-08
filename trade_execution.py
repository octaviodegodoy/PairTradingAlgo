import logging
from broker_connector import BrokerConnector
from risk_manager import RiskManager
from utils import calculate_volumes, updates_zscore_entry, get_spread_trade_direction
from constants import (
    MAX_GRIDS,
    Z_SCORE_ENTRY_THRESHOLD,
    MAGIC_NUMBER,
)
import time

class TradeExecution:
    def __init__(self, magic_number, connector: BrokerConnector):
        self.magic_number = magic_number
        self.mt5_conn = connector
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)
    
    def execute_trade(self, symbolY, symbolX, slope, hedge_ratio, z_score, trend_signal: int = 0):
        total_positions = self.mt5_conn.get_total_positions()
        grid_count = (total_positions / 2)
        equity = self.mt5_conn.get_account_info().equity
        total_max_lots = self.risk_manager.max_lots(equity)

        highest_zscore_period, total_profit, total_traded_volumes, grid_count_history = self.mt5_conn.total_daily_risk()
        self.logger.info(f"Highest Z-Score Period: {highest_zscore_period} total volumes {total_traded_volumes} and max lots {total_max_lots}")
        updated_zscore_entry = updates_zscore_entry(highest_zscore_period, total_profit, total_traded_volumes, grid_count_history, grid_count, MAGIC_NUMBER)

        # Resolve direction and skew alignment before placing orders
        direction = get_spread_trade_direction(z_score, slope, updated_zscore_entry, trend_signal)
        skew_label = (
            "CONFIRMS" if direction['skew_confirms'] else
            "WARNS"    if direction['skew_warns']    else
            "NEUTRAL"
        )
        self.logger.info(
            f"[Direction] spread={direction['spread_direction']} | "
            f"Y={direction['action_y']}, X={direction['action_x']} | "
            f"heteroscedasticity_trend={skew_label} (trend_signal={trend_signal:+d})"
        )
        if direction['skew_warns']:
            self.logger.warning(
                f"Residual variance trending AGAINST {direction['spread_direction']} — "
                f"heteroscedasticity opposes the z-score signal. Skipping trade."
            )

        self.logger.info(f"Max volume : {total_max_lots} open positions volume {total_traded_volumes} current zscore {z_score} updated zscore entry {updated_zscore_entry}")
        min_lot_Y = self.mt5_conn.get_symbol_info(symbolY).volume_min
        min_lot_X = self.mt5_conn.get_symbol_info(symbolX).volume_min
        volumeY, volume_X = calculate_volumes(symbolY, symbolX, hedge_ratio, min_lot_Y, min_lot_X, total_max_lots, total_positions)
        self.logger.info(f"Calculated volumes - {symbolY}: {volumeY}, {symbolX}: {volume_X}")

        if (total_traded_volumes < total_max_lots) and grid_count < MAX_GRIDS:
            # Block the trade when volatility skewness contradicts the z-score direction
            if direction['skew_warns']:
                self.logger.warning(
                    f"[BLOCKED] Residual variance trending against {direction['spread_direction']} "
                    f"(trend_signal={trend_signal:+d}) — heteroscedasticity opposes z-score, no order placed."
                )
                return

            self.logger.info(f"Sending order: z_score={z_score}, threshold={updated_zscore_entry}, slope={slope}")
            if slope > 0:
                if z_score < -updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score)
                elif z_score > updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score)
            elif slope < 0:
                if z_score < -updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_BUY, self.mt5_conn.ORDER_TYPE_BUY]
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score)
                elif z_score > updated_zscore_entry:
                    orders_type = [self.mt5_conn.ORDER_TYPE_SELL, self.mt5_conn.ORDER_TYPE_SELL]
                    self.mt5_conn.place_order(symbolY, symbolX, volumeY, volume_X, orders_type, z_score)