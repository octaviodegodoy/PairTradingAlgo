import asyncio
from mt5_connector import MT5Connector
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import TRADING_PAIR_Y, TRADING_PAIR_X

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    try:
        if not mt5_conn.initialize():
            logger.error("MT5 initialization failed")
            return
        logger.info("MT5 initialized successfully")
        data_y = mt5_conn.get_data(TRADING_PAIR_Y[0])
        data_x = mt5_conn.get_data(TRADING_PAIR_X[0])
        dates = data_y['time']

        print(f"Data Y length: {len(data_y['close'])} and Data X length: {len(data_x)}")

        data_y = np.array(data_y['close'])
        log_asset_y = np.log(data_y)
        print(f"Data Y length after np array: {len(log_asset_y)}")

        data = pd.DataFrame({'price_y':log_asset_y}, index=dates)

         # Calculate log returns
        data['log_return'] = np.log(data['price_y'] / data['price_y'].shift(1))
        # Calculate cumulative log returns
        data['cum_log_return'] = data['log_return'].cumsum()
        # Convert cumulative log returns to cumulative returns (compounded)
        data['cum_return'] = np.exp(data['cum_log_return']) - 1    

        print(f"Data Y length: {data}")    

        data_win = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])    

        plt.figure(figsize=(12, 6))
        plt.plot(data_win['time'], data_win['close'], marker='o', linestyle='-')
        plt.title('Close Price - Last 252 Business Days')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        mt5_conn.shutdown()
        logger.info("MT5 shutdown completed")        
    await asyncio.sleep(1)
    print("Main function completed.")

async def test_get_data_prices():
    logging.basicConfig(level=logging.INFO)
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    
    data_win = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])    

    plt.figure(figsize=(12, 6))
    plt.plot(data_win['time'], data_win['close'], marker='o', linestyle='-')
    plt.title(f'Close Price - Last 252 Business Days for {data_win["symbol"].iloc[0]}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    #data_y = mt5_conn.get_data(symbol)

asyncio.run(test_get_data_prices())