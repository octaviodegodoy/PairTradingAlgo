import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
import pandas as pd
from constants import MARGIN_PERCENT, MARGIN_X, MARGIN_Y, PERIODS, SHIFT_PERIODS, TRAILING_DISTANCE_POINTS, UNIX_DAY, MAGIC_NUMBER
from utils import calculate_volumes

class MT5Connector:
    
    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1

    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MetaTrader 5")
        self.logger = logging.getLogger(__name__)

    def get_data(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, SHIFT_PERIODS, PERIODS)
        if rates is None:
            self.logger.error(f"Could not get rates for {symbol}")
            return None
        else:
            df = pd.DataFrame(rates)
            df = df.dropna()
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Filter out weekends (keep only weekdays)
            df = df[df['time'].dt.weekday < 5]  # 0=Monday, ..., 4=Friday
            return df
        
    def get_data_futures_btg(self, symbol):
        futures_symbols = mt5.symbols_get(symbol)
        time_now = int(time.time())
        next_symbols_fut = {}
        for s in futures_symbols:
            if s.expiration_time > time_now and len(s.name) == 6:
               #print(f"Found future symbol {s.name} expiring at {datetime.fromtimestamp(s.expiration_time)}")
               next_symbols_fut[s.expiration_time] = s.name

        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        current_symbol = list(sorted_next_futures.items())[0]
        print(f"Current future symbol for {symbol} is {current_symbol[1]}")

        fut_history_symbols = [current_symbol[1]]
        fut_history_symbols.append(symbol.replace("*", "$"))  # Append the historical symbol

        dataframes = []

        for fut_symbol in fut_history_symbols:
            rates = mt5.copy_rates_from_pos(fut_symbol, mt5.TIMEFRAME_D1, SHIFT_PERIODS, PERIODS)
            print(f"Retrieved {len(rates) if rates is not None else 0} records for symbol {fut_symbol}")
            if rates is not None and len(rates) > 0:
               df = pd.DataFrame(rates)
               df['symbol'] = symbol  # Optionally keep track of the symbol
               dataframes.append(df)

        if dataframes:
            all_data = pd.concat(dataframes, ignore_index=True)
            # Remove duplicates based on the index (in this case, you might want to remove by 'time' or another column)
            all_data = all_data.drop_duplicates(subset=['time', 'symbol'])  # Adjust subset as needed
        else:
            print("No data retrieved.")

        # If 'time' is not already datetime, convert it
        all_data['time'] = pd.to_datetime(all_data['time'], unit='s')  # or remove unit if already datetime
        # Sort by date (most recent last)
        all_data = all_data.sort_values('time')
        futures_data = all_data[all_data['time'].dt.weekday < 5].tail(PERIODS)

        #print(f"Retrieved without weekends {len(futures_data)} with weekend records {len(all_data)} for futures data of {symbol}")

        return futures_data            
        
    def get_symbol_futures(self,group_name):
        futures_symbols = mt5.symbols_get(group_name)
        time_now = int(time.time())
        next_symbols_fut = {}
        past_symbols_fut = {}
        for s in futures_symbols:
            if s.expiration_time > time_now and len(s.name) == 6:
               next_symbols_fut[s.expiration_time] = s.name
            elif s.expiration_time < time_now and len(s.name) == 6:
               past_symbols_fut[s.expiration_time] = s.name
        
        sorted_next_futures = dict(sorted(next_symbols_fut.items()))
        current_symbol = list(sorted_next_futures.items())[0]

        return current_symbol
    
    def check_positions_type(self,symbol,position_type):
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            self.logger.error(f"Could not get positions for {symbol}")
            return False
        for pos in positions:
            if pos.type == position_type:
                return True
        return False
    
    def all_positions_stop_loss(self):

        while True:  
            # Get all open positions
            positions = mt5.positions_get()

            if positions is None:
                print("No positions found, error code =", mt5.last_error())
                return
            all_set = True

            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                position_type = position.type
                symbol_info = mt5.symbol_info(symbol)
                tick_size = symbol_info.trade_tick_size
            
                ask = symbol_info.ask
                bid = symbol_info.bid
            
                # Calculate the stop loss price based on the position type
                if position_type == mt5.POSITION_TYPE_BUY:
                    stop_loss_price = bid - TRAILING_DISTANCE_POINTS * tick_size
                elif position_type == mt5.POSITION_TYPE_SELL:
                    stop_loss_price = ask + TRAILING_DISTANCE_POINTS * tick_size
                print(f"Tentando setar stop com stop price {stop_loss_price} no ask {ask}")
                # Modify the position to include the stop loss
                request = {
                        "action": mt5.TRADE_ACTION_SLTP,
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

    def trailing_stop(self,symbol,position_type,stop_loss,ticket,position):
    # Get all open positions    
        profit = mt5.account_info().profit
        stop_active = stop_loss != 0.0
        symbol_info = mt5.symbol_info(symbol)
        tick_size = symbol_info.trade_tick_size
        digits = symbol_info.digits
        ask = symbol_info.ask
        bid = symbol_info.bid    
        
        # Check if profit exceeds the threshold
        if stop_active:
            
            # Calculate the new stop loss level
            if position_type == mt5.ORDER_TYPE_BUY:
                new_stop_loss = bid - (TRAILING_DISTANCE_POINTS * tick_size)
                
                # Only modify if the new SL is higher than the current one
                if (stop_loss == 0.0) or new_stop_loss > stop_loss:
                    print(f"Updating Trailing para {symbol} point {tick_size} com profit {profit} new stop {new_stop_loss} old stop {stop_loss} ")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": ticket,
                        "sl": new_stop_loss,
                        "tp": position.tp,
                    }
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Failed to update SL for BUY {symbol}, ticket {ticket}: {result.comment}")
                    else:
                        print(f"Trailing stop updated for BUY {symbol}, ticket {ticket}. New SL: {new_stop_loss:.{digits}f}")
            elif position_type == mt5.ORDER_TYPE_SELL:
                new_stop_loss = ask + (TRAILING_DISTANCE_POINTS * tick_size)
                # Only modify if the new SL is lower than the current one
                if (stop_loss == 0.0) or new_stop_loss < stop_loss:
                    print(f"Updating Trailing para {symbol} point {tick_size} com profit {profit} new stop {new_stop_loss} old stop {stop_loss} ")
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": ticket,
                        "sl": new_stop_loss,
                        "tp": position.tp,
                    }
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print(f"Failed to update SL for SELL {symbol}, ticket {ticket}: {result.comment}")
                    else:
                        print(f"Trailing stop updated for SELL {symbol}, ticket {ticket}. New SL: {new_stop_loss:.{digits}f}")


    def place_order(self,symbolY,symbolX,volumeY,volumeX,orders_type,zscore):
        """
        Place orders with retry logic for both symbols.
        Returns True if both orders succeeded, False otherwise.
        """
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        # Retry-able error codes
        retryable_codes = [
            mt5.TRADE_RETCODE_REQUOTE,      # 10004
            mt5.TRADE_RETCODE_PRICE_OFF,    # 10015
            mt5.TRADE_RETCODE_TIMEOUT,      # 10012
            mt5.TRADE_RETCODE_PRICE_CHANGED # 10016
        ]
        
        # Place Y order with retry logic
        result_y_order = None
        for attempt in range(max_retries):
            # Get fresh price for each attempt
            tick_y = mt5.symbol_info_tick(symbolY)
            price_y = tick_y.bid if orders_type[0] == mt5.ORDER_TYPE_SELL else tick_y.ask
            
            request_y = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbolY,
                "volume": volumeY,
                "type": orders_type[0],
                "price": price_y,
                "sl": 0.0,
                "tp": 0.0,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": "y,{:.2f}".format(zscore),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result_y_order = mt5.order_send(request_y)
            
            if result_y_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✓ Order Y ({symbolY}) executed successfully on attempt {attempt + 1}")
                break
            elif result_y_order.retcode in retryable_codes:
                print(f"⚠ Order Y attempt {attempt + 1} failed with retcode {result_y_order.retcode}: {result_y_order.comment}. Retrying...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Order Y failed with non-retryable error {result_y_order.retcode}: {result_y_order.comment}")
                return False
        
        if result_y_order.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"✗ Order Y failed after {max_retries} attempts")
            return False
        
        # Place X order with retry logic
        result_x_order = None
        for attempt in range(max_retries):
            # Get fresh price for each attempt
            tick_x = mt5.symbol_info_tick(symbolX)
            price_x = tick_x.ask if orders_type[1] == mt5.ORDER_TYPE_BUY else tick_x.bid
            
            request_x = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbolX,
                "volume": volumeX,
                "type": orders_type[1],
                "price": price_x,
                "sl": 0.0,
                "tp": 0.0,
                "deviation": 10,
                "magic": MAGIC_NUMBER,
                "comment": "x,{:.2f}".format(zscore),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result_x_order = mt5.order_send(request_x)
            
            if result_x_order.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✓ Order X ({symbolX}) executed successfully on attempt {attempt + 1}")
                break
            elif result_x_order.retcode in retryable_codes:
                print(f"⚠ Order X attempt {attempt + 1} failed with retcode {result_x_order.retcode}: {result_x_order.comment}. Retrying...")
                time.sleep(retry_delay)
            else:
                print(f"✗ Order X failed with non-retryable error {result_x_order.retcode}: {result_x_order.comment}")
                # Y order succeeded but X failed - consider closing Y position here
                return False
        
        if result_x_order.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"✗ Order X failed after {max_retries} attempts")
            # Y order succeeded but X failed - consider closing Y position here
            return False
        
        print(f"✓ Both orders executed successfully!")
        return True 
    
    def close_all_positions(self):
        # Get all open positions
        positions = mt5.positions_get()
        if positions is not None or len(positions) > 0:
            # Loop through each position and close it
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                volume = position.volume
                position_magic = position.magic
                position_type = position.type  # 0 for buy, 1 for sell

            # Determine the opposite order type to close the position
                if position_type == mt5.ORDER_TYPE_BUY:
                    order_type = mt5.ORDER_TYPE_SELL
                    zscore = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    zscore = mt5.symbol_info_tick(symbol).ask

            # Create a close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "position": ticket,
                    "zscore": zscore,
                    "deviation": 20,
                    "magic": MAGIC_NUMBER,
                    "comment": "Close position",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

            # Send the close request
                if (position_magic != MAGIC_NUMBER):
                    continue
                # Retry logic for closing position
                max_retries = 3
                retry_delay = 0.5
                result = None
                
                for attempt in range(max_retries):
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        break
                    elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF, 
                                           mt5.TRADE_RETCODE_TIMEOUT, mt5.TRADE_RETCODE_PRICE_CHANGED]:
                        print(f"⚠ Close attempt {attempt + 1} failed for {symbol}, retcode {result.retcode}. Retrying...")
                        # Update price for retry
                        if order_type == mt5.ORDER_TYPE_SELL:
                            request["zscore"] = mt5.symbol_info_tick(symbol).bid
                        else:
                            request["zscore"] = mt5.symbol_info_tick(symbol).ask
                        time.sleep(retry_delay)
                    else:
                        print(f"✗ Close failed with non-retryable error {result.retcode}: {result.comment}")
                        break

            # Check the result
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Failed to close position {ticket} on {symbol}, Error code: {result.retcode}")
                else:
                    print(f"Successfully closed position {ticket} on {symbol}")

    
    def total_daily_risk(self):
        from_date = datetime.now() - timedelta(hours=12,minutes=0)
        #get the number of deals in history
        to_date=datetime.now()
        print(f"From date {from_date} to date {to_date}")
        deals=mt5.history_deals_get(from_date, to_date) 
        total_profit = 0
        total_volume = 0.0
        highest_score = 0.0
        traded_zscore = 0.0
        total_valid_deals = 0
        grid_deals_count = 0
        if deals==None:   
                logging.error("No deals , error code={}".format(mt5.last_error()))   
        elif len(deals) > 0:        
            for deal in deals:
                if (len(deal.comment) > 1) and (deal.symbol != ''):
                    comment_deal = deal.comment.split(",")
                    total_valid_deals += 1
                    if (comment_deal[0] == 'y') or (comment_deal[0] == 'x'):
                        traded_zscore = abs(float(comment_deal[1]))
                    if (traded_zscore > highest_score):
                        highest_score = traded_zscore
                total_profit = total_profit + deal.commission + deal.profit
                total_volume = total_volume + deal.volume
        grid_deals_count = total_valid_deals/2
        
        return highest_score,total_profit,total_volume,grid_deals_count
    
    def get_symbol_info(self,symbol):
        symbol_info = mt5.symbol_info(symbol)
        return symbol_info

    def get_account_info(self):
        account_info = mt5.account_info()
        return account_info    

    def get_open_positions(self):
        positions = mt5.positions_get()
        return positions
    
    def get_total_volume(self):
        total_volume = 0.0
        positions = mt5.positions_get()
        if positions is not None:
            for pos in positions:
                total_volume += pos.volume
        return total_volume
    
    def get_total_positions(self):
        total_orders = mt5.orders_total()
        total_positions = mt5.positions_total()
        total_orders_positions = total_orders + total_positions
        return total_orders_positions
    
    def last_error():
        last_error = mt5.last_error()
        return last_error

    def sleep(self, seconds):
        time.sleep(seconds)

    def initialize(self):
        return mt5.initialize()

    def shutdown(self):
        mt5.shutdown()

    def get_profit(self):
        profit = mt5.account_info().profit
        return profit
    
    def get_max_lots(self):
        current_equity = mt5.account_info().equity
        total_margin = current_equity*MARGIN_PERCENT
        max_lots_y = total_margin/MARGIN_Y
        max_lots_x = total_margin/MARGIN_X
        total_max_lots = max_lots_y + max_lots_x
        return total_max_lots