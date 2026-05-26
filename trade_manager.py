import logging
from broker_connector import BrokerConnector
from trade_execution import TradeExecution
from risk_manager import RiskManager
from constants import (
    KALMAN_FILTER_METHOD, PROFIT_THRESHOLD, MAGIC_NUMBER, TRAILING_DISTANCE_POINTS
)
import time
from utils import check_trading_time, get_correlation, get_dynamic_spread_zscores, get_group_name, get_linear_regression_spread_zscores

class TradeManager:
    def __init__(self, connector: BrokerConnector):
        self.mt5_conn = connector
        self.trade_execution = TradeExecution(MAGIC_NUMBER, connector)
        self.risk_manager = RiskManager()
        self.logger = logging.getLogger(__name__)

    def manage_trades(self):        
        
        stop_active = False
        open_position_y = None
        open_position_x = None
        # Pre-initialise so these names are always bound before the logging
        # statements outside the for-loop (fixes potential UnboundLocalError).
        ticket_y = None
        ticket_x = None
        stop_loss_y = None
        stop_loss_x = None
        type_position_y = None
        type_position_x = None
        
        while True:
            # Recalculate risk limits each cycle so shrinking equity tightens the stop
            equity = self.mt5_conn.get_account_info().equity
            max_loss = self.risk_manager.max_loss(equity)
            trailing_start = max_loss * PROFIT_THRESHOLD
            # Implement position management logic
            #          

            total_positions = self.mt5_conn.get_total_positions() #self.mt5_conn.positions_total() 
            if total_positions > 0:
                self.logger.info(f"Total open positions: {total_positions}")
                profit = self.mt5_conn.get_profit()
                self.logger.info(f"Current profit: {profit}, Trailing start: {trailing_start}, Max loss: {max_loss} ")

                if profit >= trailing_start:
                    self._set_initial_stop_losses()
                    self.logger.info("Initial stop-losses activated for all positions.")
                elif (profit <= -max_loss) or not check_trading_time():
                    self.logger.info("Profit below max loss or outside trading time, closing all positions.")
                    self.mt5_conn.close_all_positions()
                    break


                positions = self.mt5_conn.get_open_positions()  #self.mt5_conn.positions_get()

                for position in positions:
                    position_name = position.comment.split(",")
                    
                    if position.sl != 0.0:
                        stop_active = True
                    
                    if position_name[0] == 'x':
                        # Extract position details
                        ticket_x = position.ticket
                        open_position_x = position.symbol
                        stop_loss_x = position.sl
                        type_position_x = position.type  # 0 = BUY, 1 = SELL
                        self._update_trailing_stop(open_position_x, type_position_x, stop_loss_x, ticket_x, position)

                    elif position_name[0] == 'y':
                        ticket_y = position.ticket
                        open_position_y = position.symbol
                        stop_loss_y = position.sl
                        type_position_y = position.type  # 0 = BUY, 1 = SELL
                        self._update_trailing_stop(open_position_y, type_position_y, stop_loss_y, ticket_y, position)

                if (stop_active):
                    time.sleep(15)
                    continue
   
                if open_position_y:
                    self.logger.info(f"Position Y: {open_position_y}, Type: {type_position_y}, Stop Loss: {stop_loss_y}, Ticket: {ticket_y}")
                if open_position_x:
                    self.logger.info(f"Position X: {open_position_x}, Type: {type_position_x}, Stop Loss: {stop_loss_x}, Ticket: {ticket_x}")

                if not open_position_y or not open_position_x:
                    self.logger.warning("One of the open positions is missing, skipping trade execution.")
                    time.sleep(15)
                    continue

                 # Get group names and data for z-score calculation
                asset_group_y = get_group_name(open_position_y)
                asset_group_x = get_group_name(open_position_x)
                assets_y = self.mt5_conn.get_data_futures_btg(asset_group_y)
                assets_x = self.mt5_conn.get_data_futures_btg(asset_group_x)
                correlation = get_correlation(assets_y,assets_x)
                result = None
                if KALMAN_FILTER_METHOD:
                     result = get_dynamic_spread_zscores(assets_y, assets_x)
                else:
                     result = get_linear_regression_spread_zscores(assets_y, assets_x)

                current_z = result['z_scores'].iloc[-1]
                self.logger.info(f"Current z_score={current_z:.4f}, hedge_ratio={result['hedge_ratio'].iloc[-1]:.4f}, correlation={correlation:.4f}")

                self.trade_execution.execute_trade(open_position_y, open_position_x, correlation, result['hedge_ratio'].iloc[-1], current_z)
                time.sleep(15)
                
                
            
            elif total_positions == 0:
                self.logger.info("No open positions to manage")
                break
            
            
            self.logger.info(f"Managing positions")
            time.sleep(15)  # Sleep for a while before next management cycle

    def _set_initial_stop_losses(self):
        """
        Set an initial stop-loss on every open position that does not yet have one.
        Called once when portfolio profit reaches the trailing_start threshold.
        Also fixes the loop-forever bug in the old all_positions_stop_loss.
        """
        positions = self.mt5_conn.get_open_positions()
        if not positions:
            return
        for position in positions:
            if position.sl != 0.0:
                continue  # already protected
            symbol = position.symbol
            info = self.mt5_conn.get_symbol_info(symbol)
            tick = self.mt5_conn.get_symbol_tick(symbol)
            tick_size = info.trade_tick_size
            if position.type == self.mt5_conn.POSITION_TYPE_BUY:
                sl_price = tick.bid - TRAILING_DISTANCE_POINTS * tick_size
            else:
                sl_price = tick.ask + TRAILING_DISTANCE_POINTS * tick_size
            success = self.mt5_conn.modify_position_sl(position.ticket, symbol, sl_price, position.tp)
            if success:
                self.logger.info(f"Initial SL set for {symbol} ticket {position.ticket}: {sl_price:.{info.digits}f}")
            else:
                self.logger.warning(f"Failed to set initial SL for {symbol} ticket {position.ticket}")

    def _update_trailing_stop(self, symbol, position_type, stop_loss, ticket, position):
        """
        Ratchet the stop-loss for a single open position towards the current price.
        Only modifies the SL when the new level is more favourable than the existing one.
        Does nothing when stop_loss == 0.0 (SL not yet placed).
        """
        if stop_loss == 0.0:
            return
        info = self.mt5_conn.get_symbol_info(symbol)
        tick = self.mt5_conn.get_symbol_tick(symbol)
        tick_size = info.trade_tick_size
        digits = info.digits
        if position_type == self.mt5_conn.ORDER_TYPE_BUY:
            new_sl = round(tick.bid - TRAILING_DISTANCE_POINTS * tick_size, digits)
            if new_sl > stop_loss:
                success = self.mt5_conn.modify_position_sl(ticket, symbol, new_sl, position.tp)
                if success:
                    self.logger.info(f"Trailing stop advanced BUY {symbol}: {stop_loss:.{digits}f} → {new_sl:.{digits}f}")
        elif position_type == self.mt5_conn.ORDER_TYPE_SELL:
            new_sl = round(tick.ask + TRAILING_DISTANCE_POINTS * tick_size, digits)
            if new_sl < stop_loss:
                success = self.mt5_conn.modify_position_sl(ticket, symbol, new_sl, position.tp)
                if success:
                    self.logger.info(f"Trailing stop advanced SELL {symbol}: {stop_loss:.{digits}f} → {new_sl:.{digits}f}")
