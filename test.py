import asyncio
from email import parser
import math
from statistics import correlation
from turtle import color
from mt5_connector import MT5Connector
from utils import get_dynamic_spread_zscores, calculate_volumes, get_linear_regression_spread_zscores
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import MARGIN_PERCENT, MARGIN_X, MARGIN_Y, NOISE_VARIANCE, PERIODS, TRADING_PAIR_Y, TRADING_PAIR_X
from sklearn.linear_model import LinearRegression


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

async def plot_data_prices():
    logging.basicConfig(level=logging.INFO)
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return
    
    for i in range(len(TRADING_PAIR_Y)):
        for j in range(len(TRADING_PAIR_X)):
            
            print("Testing Index Y:", TRADING_PAIR_Y[i], "Index X:", TRADING_PAIR_X[j])
    
            assets_y = mt5_conn.get_data_futures(TRADING_PAIR_Y[i])
            assets_x = mt5_conn.get_data_futures(TRADING_PAIR_X[j])
            print(f"y data {len(assets_y)} and x data {len(assets_x)}")
            if len(assets_y) >= PERIODS and len(assets_x) >= PERIODS:
                    rolling_z_scores, spreads, hedge_ratio, correlation = get_dynamic_spread_zscores(assets_y, assets_x)

                    ratio = hedge_ratio[-1]
                    investment_asset_y = (20/(1 + ratio))
                    investment_asset_x = (20 - investment_asset_y)
                    print(f"Current Z-Score: {rolling_z_scores[-1]} hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x} ")
                    
                    price1 = np.array(assets_y['close'])
                    price2 = np.array(assets_x['close'])
                    dates = np.array(assets_y['time'])  # Use the last 300 dates for better visibility

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
                    plt.title(f'Pair Trade Cumulative Returns of {TRADING_PAIR_Y[i]} and {TRADING_PAIR_X[j]} correlation {correlation} volume y {math.floor(investment_asset_y)} and x {math.floor(investment_asset_x)}')
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
                    plt.grid(True)
                    plt.show()

async def get_daily_data():
    logging.basicConfig(level=logging.INFO)
    mt5_conn = MT5Connector()
    total_day_risk,highest_score,total_profit,total_volume = mt5_conn.total_daily_risk()
    print(f"Total daily risk: {total_day_risk}, Highest score: {highest_score}, Total profit: {total_profit} and total volume {total_volume}")
    
async def get_group_name(symbol):
    logging.basicConfig(level=logging.INFO)
    group_name = symbol[:3]+'*'
    print(f"Getting group name for symbol: {group_name}")

async def check_volumes():
        mt5_conn = MT5Connector()
        current_equity = mt5_conn.get_account_info().equity
        total_margin = current_equity*MARGIN_PERCENT
        max_lots_y = total_margin/MARGIN_Y
        max_lots_x = total_margin/MARGIN_X
        total_max_lots = max_lots_y + max_lots_x
        print(f"Max lots for Y: {max_lots_y}, Max lots for X: {max_lots_x}, Total max lots: {total_max_lots}")
    

async def get_residuals_zscore_stdev():
        mt5_conn = MT5Connector()
        
        assets_y = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])
        assets_x = mt5_conn.get_data_futures(TRADING_PAIR_X[0])

        print(f"Data Y length: {len(assets_y)} and Data X length: {len(assets_x)}")

        # Log-transform the prices
        log_asset1 = np.log(assets_y['close'])
        log_asset2 = np.log(assets_x['close'])

        log_asset1 = pd.Series(log_asset1)
        log_asset2 = pd.Series(log_asset2)

        X = log_asset1.values.reshape(-1, 1)
        y = log_asset2.values

        model = LinearRegression()
        model.fit(X, y)

        # Predict log_price2 using the regression model
        log_price2_pred = model.predict(X)

        # Calculate residuals: actual - predicted
        residuals = log_asset2 - log_price2_pred

        residual_spreads = residuals.rolling(window=PERIODS, min_periods=1).std()
        noise_variance = residual_spreads.std()
        print(f"Residuals standard deviation: {residual_spreads.std()}")
        print(f"Residuals variance: {residual_spreads.var():.10f}")
        return noise_variance

async def print_linear_regression_spread_zscores():
    mt5_conn = MT5Connector()
    
    assets_y = mt5_conn.get_data_futures(TRADING_PAIR_Y[0])
    assets_x = mt5_conn.get_data_futures(TRADING_PAIR_X[0])

    log_asset1 = np.log(assets_y['close'])
    log_asset2 = np.log(assets_x['close'])

    cum_log_return_asset1 = log_asset1 - log_asset1.iloc[0]
    cum_log_return_asset2 = log_asset2 - log_asset2.iloc[0]

    # Calculate cumulative percentage returns using natural log
    # Cumulative percentage return = (exp(cumulative log return) - 1) * 100
    cum_pct_return_asset1 = (np.exp(cum_log_return_asset1) - 1) * 100
    cum_pct_return_asset2 = (np.exp(cum_log_return_asset2) - 1) * 100

    rolling_z_scores, spreads, hedge_ratio, correlation = get_linear_regression_spread_zscores(assets_y, assets_x)
    ratio = hedge_ratio
    investment_asset_y = (20/(1 + ratio))
    investment_asset_x = (20 - investment_asset_y)

    dates = assets_y['time']

    results = pd.DataFrame({
    'cum_pct_return_asset1': cum_pct_return_asset1.values,
    'cum_pct_return_asset2': cum_pct_return_asset2.values,
    'residuals': spreads,
    'zscores': rolling_z_scores
}, index=dates)

    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(20, 18))

    ax2.plot(results.index, rolling_z_scores, label='Z-Score')
    ax2.axhline(ratio, color='orange', linestyle='--', label='Hedge Ratio')
    ax2.axhline(1, color='green', linestyle='--', label='Upper Threshold (+2)')
    ax2.axhline(0, color='black', linestyle='--', label='Middle Threshold (0)')
    ax2.axhline(-1, color='green', linestyle='--', label='Lower Threshold (-2)')
    ax2.set_title(f'Z-Score of Residuals with ratio {ratio} correlation {correlation}')
    ax2.legend()

    ax3.plot(results.index, results['cum_pct_return_asset1'], color='blue', label='Cumulative Returns Asset Y')
    ax3.plot(results.index, results['cum_pct_return_asset2'], color='red', label='Cumulative Returns Asset X')
    ax3.set_title('Cumulative Returns')
    ax3.set_xlabel('Date')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    print(f"Current Z-Score: {rolling_z_scores[-1]} hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x} correlation {correlation} ")

asyncio.run(plot_data_prices())