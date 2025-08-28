import logging
from mt5_connector import MT5Connector

class TradeManager:
    def __init__(self, magic_number):
        self.magic_number = magic_number
        self.mt5_conn = MT5Connector()
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, symbolY, symbolX, hedge_ratio, z_score):
        # Implement actual trade logic; replace with MetaTrader 5 API calls
        self.logger.info(f"Trading {symbolY} vs {symbolX} | Hedge: {hedge_ratio:.2f} | Z-score: {z_score:.2f}")

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
                    result = mt5.order_send(request)
                    print(f"Result of order check for ticket {ticket}: ", result)
           
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to set stop loss for position {ticket}, error code: {result.retcode}")
                all_set = False

            if all_set:
                print("All positions have non-zero stop loss. Exiting loop.")
                break

            time.sleep(1)  # Sleep for a short time before checking again 
