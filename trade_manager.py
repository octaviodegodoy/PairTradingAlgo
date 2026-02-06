import logging
from mt5_connector import MT5Connector
from trade_execution import TradeExecution  # Trade execution not implemented yet
from strategy import PairTradingStrategy
from constants import (
    ADDITIONAL_GRID, KALMAN_FILTER_METHOD, MARGIN_PERCENT, MAX_RISK, PROFIT_THRESHOLD, TRADING_PAIR_X, TRADING_PAIR_Y, TRAILING_DISTANCE_POINTS, MAGIC_NUMBER
)
import time
from utils import check_trading_time, get_dynamic_spread_zscores,get_group_name, get_linear_regression_spread_zscores

class TradeManager:
    def __init__(self):
        self.mt5_conn = MT5Connector()
        self.trade_execution = TradeExecution(MAGIC_NUMBER)
        self.pair_trading_strategy = PairTradingStrategy()
        self.logger = logging.getLogger(__name__)

    def manage_trades(self):        
        
        stop_active = False
        open_position_y = None
        open_position_x = None
        current_equity = self.mt5_conn.get_account_info().equity
        total_margin = current_equity*MARGIN_PERCENT
        max_loss = total_margin*MAX_RISK
        trailing_start = max_loss*PROFIT_THRESHOLD
        
        while True:
            # Implement position management logic
            #          

            total_positions = self.mt5_conn.get_total_positions() #self.mt5_conn.positions_total() 
            if total_positions > 0:
                self.logger.info(f"Total open positions: {total_positions}")
                profit = self.mt5_conn.get_profit()
                self.logger.info(f"Current profit: {profit}, Trailing start: {trailing_start}, Max loss: {max_loss} ")

                if profit >= trailing_start:
                    self.mt5_conn.all_positions_stop_loss()
                    self.logger.info("Trailing stop activated for all positions.")
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
                        self.mt5_conn.trailing_stop(open_position_x,type_position_x,stop_loss_x,ticket_x,position)                        
                        
                    elif position_name[0] == 'y':
                        ticket_y = position.ticket
                        open_position_y = position.symbol
                        stop_loss_y = position.sl
                        type_position_y = position.type  # 0 = BUY, 1 = SELL
                        self.mt5_conn.trailing_stop(open_position_y,type_position_y,stop_loss_y,ticket_y,position)

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
                if KALMAN_FILTER_METHOD:
                     rolling_z_scores, spreads, hedge_ratio = get_dynamic_spread_zscores(assets_y, assets_x)
                else:
                     rolling_z_scores, spreads, hedge_ratio = get_linear_regression_spread_zscores(assets_y, assets_x)

                self.logger.info(f"Sending new order to trade execution with z score {rolling_z_scores.iloc[-1]} and hedge ratio {hedge_ratio} grid add {ADDITIONAL_GRID} and correlation {hedge_ratio}")
                self.trade_execution.execute_trade(open_position_y, open_position_x, hedge_ratio,rolling_z_scores.iloc[-1])
                time.sleep(15)
                
                
            
            elif total_positions == 0:
                self.logger.info("No open positions to manage")
                break
            
            
            self.logger.info(f"Managing positions")
            time.sleep(15)  # Sleep for a while before next management cycle

    def close_all_positions(self):
        # Implement closing logic
        self.logger.info("Closing all open positions")

    def all_positions_stop_loss(self):    
        while True:  
            # Get all open positions
            positions = self.mt5_conn.positions_get()
    
            if positions is None:
                self.logger.info(f"No positions found, error code = {self.mt5_conn.last_error()} ")
                print()
                return
            all_set = True

            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                position_type = position.type
                symbol_info = self.mt5_conn.symbol_info(symbol)
                tick_size = symbol_info.trade_tick_size        
                ask = symbol_info.ask
                bid = symbol_info.bid
        
                # Calculate the stop loss price based on the position type
                if position_type == self.mt5_conn.POSITION_TYPE_BUY:
                    stop_loss_price = bid - TRAILING_DISTANCE_POINTS * tick_size
                elif position_type == self.mt5_conn.POSITION_TYPE_SELL:
                    stop_loss_price = ask + TRAILING_DISTANCE_POINTS * tick_size
                print(f"Tentando setar stop com stop price {stop_loss_price} no ask {ask}")
                # Modify the position to include the stop loss
                request = {
                        "action": self.mt5_conn.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": ticket,
                        "sl": stop_loss_price,
                        "tp": 0.0,
                    }

                if position.sl == 0.0:
                    # Only set the stop loss if it is not already set
                    print(f"Setting stop loss for position {ticket} on {symbol} to {stop_loss_price}")
                    result = self.mt5_conn.order_send(request)
                    print(f"Result of order check for ticket {ticket}: ", result)
           
                if result.retcode != self.mt5_conn.TRADE_RETCODE_DONE:
                    print(f"Failed to set stop loss for position {ticket}, error code: {result.retcode}")
                all_set = False

            if all_set:
                print("All positions have non-zero stop loss. Exiting loop.")
                break

            time.sleep(1)  # Sleep for a short time before checking again 
