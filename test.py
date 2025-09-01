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

        print(f"Data Y length: {data}")        

        plt.plot(data.index,data['price_y'], label='Prices WIN ')
        plt.legend()
        plt.show()

        # Calculate log returns
        data_x['log_return'] = np.log(data_x['close'] / data_x['close'].shift(1))
        # Calculate cumulative log returns
        data_x['cum_log_return'] = data_x['log_return'].cumsum()
        # Convert cumulative log returns to cumulative returns (compounded)
        data_x['cum_return'] = np.exp(data_x['cum_log_return']) - 1     

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        mt5_conn.shutdown()
        logger.info("MT5 shutdown completed")        
    await asyncio.sleep(1)
    print("Main function completed.")

asyncio.run(main())