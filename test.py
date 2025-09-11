import asyncio
from mt5_connector import MT5Connector
from utils import get_dynamic_spread_zscores
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import TRADING_PAIR_Y, TRADING_PAIR_X
import seaborn as sns


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()

    try:
        if not mt5_conn.initialize():
            logger.error("MT5 initialization failed")
            return
        logger.info("MT5 initialized successfully")
        data_y = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])
        data_x = mt5_conn.get_data_futures(TRADING_PAIR_X[0])
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
    
    assets_y = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])
    assets_x = mt5_conn.get_data_futures(TRADING_PAIR_X[0])
    rolling_z_scores, spreads, hedge_ratio, correlation = get_dynamic_spread_zscores(assets_y, assets_x)
    
    price1 = np.array(assets_y['close'])
    price2 = np.array(assets_x['close'])
    dates = np.array(assets_y['time'])

    log_asset1 = np.log(price1)
    log_asset2 = np.log(price2)

    print(f"Assets Y length: {len(assets_y)} and Assets X length: {len(assets_x)}")    

    data = pd.DataFrame({'Price1': price1, 'Price2': price2,'LogPrice1': log_asset1, 'LogPrice2': log_asset2,'Rolling Z': rolling_z_scores,'Hedge Ratio': hedge_ratio}, index=dates)
        
    # Calculate cumulative returns
    data['Return1'] = data['Price1'].pct_change().cumsum()
    data['Return2'] = data['Price2'].pct_change().cumsum()

    # Plot the cumulative returns
    plt.figure(figsize=(12, 8),layout='constrained')

    plt.subplot(2, 1, 1)        
    plt.plot(data.index, data['Return1'], label='Cumulative returns WDO', color='red')
    plt.plot(data.index, data['Return2'], label='Cumulative returns WIN', color='blue')
    plt.ylabel('Cumulative Return')
    plt.title(f'Pair Trade Cumulative Returns of {TRADING_PAIR_Y[0]} and {TRADING_PAIR_X[0]} correlation {correlation}')
    plt.axhline(0, color='black')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    #plt.plot(data.index, data['Z scores'], label='Z-scores', color='purple')
    plt.plot(data.index, data['Rolling Z'],label='Z-scores Rolling', color='green')
    plt.plot(data.index, data['Hedge Ratio'], label='Hedge Ratio', color='orange')
    plt.axhline(0, color='black')
    plt.axhline(1, color='blue',linestyle='--')
    plt.axhline(2, color='green', linestyle='--', label='+2 Std Dev')
    plt.axhline(-1, color='red', linestyle='--')
    plt.axhline(-2, color='green', linestyle='--', label='-2 Std Dev')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

asyncio.run(test_get_data_prices())