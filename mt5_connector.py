import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta
import time
import pandas as pd
from broker_connector import BrokerConnector
from constants import MAGIC_NUMBER, PERIODS, SHIFT_PERIODS


class MT5Connector(BrokerConnector):
    """MetaTrader 5 implementation of BrokerConnector."""

    ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    TIMEFRAME_D1 = mt5.TIMEFRAME_D1
    POSITION_TYPE_BUY = mt5.POSITION_TYPE_BUY
    POSITION_TYPE_SELL = mt5.POSITION_TYPE_SELL

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_data_futures_btg(self, symbol, n_bars=None):
        if n_bars is None:
            n_bars = PERIODS

        # Resolve the current front-month contract
        futures_symbols = mt5.symbols_get(symbol)
        if not futures_symbols:
            print(f"No symbols found for {symbol}")
            return pd.DataFrame()

        time_now = int(time.time())
        next_symbols_fut = {
            s.expiration_time: s.name
            for s in futures_symbols
            if len(s.name) == 6 and s.expiration_time > time_now
        }

        if not next_symbols_fut:
            print(f"No active futures contract found for {symbol}")
            return pd.DataFrame()

        front_name = next_symbols_fut[min(next_symbols_fut)]
        print(f"Current future symbol for {symbol} is {front_name}")

        def _fetch(sym, count):
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, SHIFT_PERIODS, count)
            n_fetched = len(rates) if rates is not None else 0
            print(f"Retrieved {n_fetched} records for {sym}")
            if rates is None or n_fetched == 0:
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['symbol'] = symbol
            return df

        # Fetch continuous symbol for history, front-month for recent prices.
        # Continuous is appended first; front-month last so it wins on dedup.
        continuous_symbol = symbol.replace("*", "$")
        df_cont = _fetch(continuous_symbol, max(n_bars * 5, 300))
        df_front = _fetch(front_name, n_bars)

        frames = [df for df in (df_cont, df_front) if not df.empty]
        if not frames:
            print("No data retrieved.")
            return pd.DataFrame()

        all_data = pd.concat(frames, ignore_index=True)
        all_data = all_data.drop_duplicates(subset=['time'], keep='last')
        all_data = all_data.sort_values('time').reset_index(drop=True)
        futures_data = all_data[all_data['time'].dt.weekday < 5].tail(n_bars)

        print(f"Concatenated {len(futures_data)} bars for {symbol} ({continuous_symbol} + {front_name})")
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
            if pos.magic == MAGIC_NUMBER and pos.type == position_type:
                return True
        return False

    def place_order(self, symbolY, symbolX, volumeY, volumeX, orders_type, zscore, sl_y: float = 0.0, sl_x: float = 0.0):
        """
        Place paired orders with retry logic.
        sl_y / sl_x : pre-computed stop-loss prices for each leg (0.0 = no SL).
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
                "sl": sl_y,
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
                "sl": sl_x,
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
                # Y order succeeded but X failed — close Y immediately to avoid unhedged exposure
                self._emergency_close_leg(symbolY, orders_type[0], volumeY)
                return False
        
        if result_x_order.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"\u2717 Order X failed after {max_retries} attempts")
            # Y order succeeded but X failed — close Y immediately to avoid unhedged exposure
            self._emergency_close_leg(symbolY, orders_type[0], volumeY)
            return False
        
        print(f"✓ Both orders executed successfully!")
        return True

    def _emergency_close_leg(self, symbol: str, open_order_type: int, volume: float):
        """Close a single unhedged leg to eliminate naked exposure after a partial fill failure."""
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        own = [p for p in positions if p.magic == MAGIC_NUMBER]
        if not own:
            return
        pos = max(own, key=lambda p: p.time)
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Emergency close unhedged leg",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(req)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\u2713 Emergency close of unhedged {symbol} leg succeeded")
        else:
            print(f"\u2717 Emergency close of {symbol} failed: {result.retcode} {result.comment}")

    def close_all_positions(self):
        # Get all open positions
        positions = mt5.positions_get()
        if positions is not None and len(positions) > 0:  # Fixed: 'or' caused TypeError when positions is None
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
                    close_price = mt5.symbol_info_tick(symbol).bid
                else:
                    order_type = mt5.ORDER_TYPE_BUY
                    close_price = mt5.symbol_info_tick(symbol).ask

            # Create a close request
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "position": ticket,
                    "price": close_price,  # Fixed: was incorrectly keyed as "zscore"
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
                            request["price"] = mt5.symbol_info_tick(symbol).bid
                        else:
                            request["price"] = mt5.symbol_info_tick(symbol).ask
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
                if deal.magic != MAGIC_NUMBER:
                    continue
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
        if positions is None:
            return []
        return [p for p in positions if p.magic == MAGIC_NUMBER]
    
    def get_total_positions(self):
        positions = mt5.positions_get()
        orders = mt5.orders_get()
        pos_count = sum(1 for p in positions if p.magic == MAGIC_NUMBER) if positions else 0
        ord_count = sum(1 for o in orders if o.magic == MAGIC_NUMBER) if orders else 0
        return pos_count + ord_count
    
    def sleep(self, seconds):
        time.sleep(seconds)

    def initialize(self):
        return mt5.initialize()

    def shutdown(self):
        mt5.shutdown()

    def get_profit(self):
        profit = mt5.account_info().profit
        return profit
    
    def get_order_calc_margin(self, order_type, symbol, volume, price):
        margin = mt5.order_calc_margin(order_type, symbol, volume, price)
        return margin
    
    def get_symbol_tick(self, symbol):
        """Return the latest tick (ask/bid/last) for the given symbol."""
        return mt5.symbol_info_tick(symbol)

    def modify_position_sl(self, ticket: int, symbol: str, sl_price: float, tp_price: float = 0.0) -> bool:
        """Send a TRADE_ACTION_SLTP request to update a position's stop-loss."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": sl_price,
            "tp": tp_price,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE