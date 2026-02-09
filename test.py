import asyncio
from email import parser
import math
from turtle import color
from mt5_connector import MT5Connector
from utils import calculate_volumes, get_dynamic_spread_zscores, get_linear_regression_spread_zscores,check_cointegration,get_correlation
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import ADDITIONAL_GRID, MARGIN_PERCENT, MARGIN_X, MARGIN_Y, NOISE_VARIANCE, PERIODS, TRADING_PAIR_Y, TRADING_PAIR_X, Z_SCORE_ENTRY_THRESHOLD
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import coint



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
    account_info = mt5_conn.get_account_info()
    for i in range(len(TRADING_PAIR_Y)):
        for j in range(len(TRADING_PAIR_X)):
            
            print("Testing Index Y:", TRADING_PAIR_Y[i], "Index X:", TRADING_PAIR_X[j])
    
            assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[i])
            assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[j])
            print(f"y data {len(assets_y)} and x data {len(assets_x)}")
            if len(assets_y) >= PERIODS and len(assets_x) >= PERIODS:
                    result = get_linear_regression_spread_zscores(assets_y, assets_x)
                    # Kalman Filter implementation
                    result_kalman = get_dynamic_spread_zscores(assets_y, assets_x)
                    print(f"Hedge ratio: {result['hedge_ratio'].iloc[-1]}, Z-Score: {result['z_scores'].iloc[-1]}")
                    print(f"Kalman Hedge ratio: {result_kalman['hedge_ratio'].iloc[-1]}, Kalman Z-Score: {len(result_kalman['z_scores'])} kalman spread {len(result_kalman['spread'])}")
                    correlation = assets_y['close'].corr(assets_x['close'])
                    print(f"Correlation is {correlation} between {TRADING_PAIR_Y[i]} and {TRADING_PAIR_X[j]}")
                    cointegration_condition = check_cointegration(result['spread'])
                    log_asset1 = np.log(assets_y['close'])
                    log_asset2 = np.log(assets_x['close'])

                    t_stat, pval, crit = coint(log_asset1, log_asset2, trend='c', autolag='AIC')
                    print("\nEngle–Granger coint test:")
                    print(f"t-stat={t_stat:.3f}, p-value={pval:.4f}, crit={crit}")
                    
                    ratio = abs(result['hedge_ratio'].iloc[-1])
                    investment_asset_x = (20/(1 + ratio))
                    investment_asset_y = (20 - investment_asset_x)
                    print(f"Current Z-Score: {result['z_scores'].iloc[-1]} hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x} ")
                    
                    k_ratio = abs(result_kalman['hedge_ratio'].iloc[-1])
                    k_investment_asset_x = (20/(1 + k_ratio))
                    k_investment_asset_y = (20 - k_investment_asset_x)
                    
                    price1 = np.array(assets_y['close'])
                    price2 = np.array(assets_x['close'])
                    dates = np.array(assets_y['time'])  # Use the last 300 dates for better visibility

                    result.index = dates
                    result_kalman.index = dates

                    log_asset1 = np.log(price1)
                    log_asset2 = np.log(price2)

                    print(f"Assets Y length: {len(assets_y)} and Assets X length: {len(assets_x)}")    

                    data = pd.DataFrame({'Price1': price1, 
                                         'Price2': price2,
                                         'LogPrice1': log_asset1, 
                                         'LogPrice2': log_asset2,
                                         'Rolling Z': result['z_scores'],
                                         'Hedge Ratio': result['hedge_ratio'],
                                         'Kalman Hedge Ratio': result_kalman['hedge_ratio'],
                                         'Kalman Z': result_kalman['z_scores'],
                                         'Kalman Spread': result_kalman['spread']}, index=dates)
                    
                    print(f"Data head:\n{data['Rolling Z'].head()}\nData tail:\n{data['Rolling Z'].tail()} and rolling z {result['z_scores'].tail()}")    
                    # Calculate cumulative returns
                    data['Return1'] = data['Price1'].pct_change().cumsum()
                    data['Return2'] = data['Price2'].pct_change().cumsum()
                    print(f"Kalman data returns:\n{data[['Kalman Z']].tail()}")

                    # Plot the cumulative returns
                    plt.figure(figsize=(12, 8),layout='constrained')

                    plt.subplot(3, 1, 1)        
                    plt.plot(data.index, data['Return1'], label='Cumulative returns WIN', color='red')
                    plt.plot(data.index, data['Return2'], label='Cumulative returns WDO', color='blue')
                    plt.ylabel('Cumulative Return')
                    plt.title(f'Pair Trade Cumulative Returns of {TRADING_PAIR_Y[i]} and {TRADING_PAIR_X[j]} cointegrated ? {cointegration_condition} correlation {result["hedge_ratio"].iloc[-1]:.2f} volume y {math.floor(investment_asset_y)} and x {math.floor(investment_asset_x)}')
                    plt.axhline(0, color='black')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(3, 1, 2)
                    plt.title(f'Rolling Z-scores from server {account_info.server} Z SCORE {result["z_scores"].iloc[-1]:.2f}')
                    plt.plot(data.index, data['Rolling Z'],label='Z-scores Rolling', color='green')
                    plt.plot(data.index, data['Hedge Ratio'], label='Hedge Ratio', color='orange')
                    plt.axhline(0, color='black')
                    plt.axhline(1, color='blue',linestyle='--')
                    plt.axhline(2, color='green', linestyle='--', label='+2 Std Dev')
                    plt.axhline(-1, color='red', linestyle='--')
                    plt.axhline(-2, color='green', linestyle='--', label='-2 Std Dev')
                    plt.grid(True)


                    plt.subplot(3, 1, 3)
                    plt.title(f'Kalman Z-scores from server {account_info.server} Z SCORE {result_kalman["z_scores"].iloc[-1]:.2f} and Hedge Ratio {result_kalman["hedge_ratio"].iloc[-1]:.2f} and volume {TRADING_PAIR_Y[i]} is {math.floor(k_investment_asset_x)} and volume {TRADING_PAIR_X[j]} is {math.floor(k_investment_asset_y)}' )
                    plt.plot(data.index, data['Kalman Z'],label='Z-scores Kalman', color='purple')
                    plt.axhline(0, color='black')
                    plt.axhline(1, color='blue',linestyle='--')
                    plt.axhline(2, color='green', linestyle='--', label='+2 Std Dev')
                    plt.axhline(-1, color='red', linestyle='--')
                    plt.axhline(-2, color='green', linestyle='--', label='-2 Std Dev')
                    plt.grid(True)
                    plt.show()

async def get_daily_data():
    logging.basicConfig(level=logging.INFO)
    mt5_conn = MT5Connector()
    highest_score,total_profit,total_volume,grid_deals_count = mt5_conn.total_daily_risk()
    print(f" Highest score: {highest_score}, Total profit: {total_profit} and total volume {total_volume} with grid deals count {grid_deals_count}")
    
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

async def test_get_data_futures():
    mt5_conn = MT5Connector()
    assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0])
    assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0])
    print(f"Data Y: {assets_y.tail(2)}")
    print(f"Data X: {assets_x.tail(2)}")
    print(f"Data Y length: {len(assets_y)} and Data X length: {len(assets_x)}")
    rolling_z_scores, spreads, hedge_ratio = get_linear_regression_spread_zscores(assets_y, assets_x)
    print(f"Hedge ratio: {hedge_ratio}, Z-Score: {rolling_z_scores.iloc[-1]}")
    grid_lot_investment = 100
    ratio = abs(hedge_ratio)    
    investment_asset_x = (grid_lot_investment/(1 + ratio))
    investment_asset_y = (grid_lot_investment - investment_asset_x)
    print(f"hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x} correlation {hedge_ratio} ")


async def print_linear_regression_spread_zscores():
    mt5_conn = MT5Connector()
    account_info = mt5_conn.get_account_info()
    print(f"Account Info: {mt5_conn.get_account_info()}")
    
    assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0])
    assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0])

    log_asset1 = np.log(assets_y['close'])
    log_asset2 = np.log(assets_x['close'])

    print(f"Tail y : {assets_y.tail(2)} and Data X tail: {assets_x.tail(2)}")

    print(f"Assets Y length: {len(assets_y)} and Assets X length: {len(assets_x)}")

    cum_log_return_asset1 = log_asset1 - log_asset1.iloc[0]
    cum_log_return_asset2 = log_asset2 - log_asset2.iloc[0]

    # Calculate cumulative percentage returns using natural log
    # Cumulative percentage return = (exp(cumulative log return) - 1) * 100
    cum_pct_return_asset1 = (np.exp(cum_log_return_asset1) - 1) * 100
    cum_pct_return_asset2 = (np.exp(cum_log_return_asset2) - 1) * 100

    # Get minimum length and trim both series to equal length
    min_length = min(len(cum_pct_return_asset1), len(cum_pct_return_asset2))
    cum_pct_return_asset1 = cum_pct_return_asset1[:min_length]
    cum_pct_return_asset2 = cum_pct_return_asset2[:min_length]
    

    rolling_z_scores, spreads, hedge_ratio = get_linear_regression_spread_zscores(assets_y, assets_x)
    ratio = abs(hedge_ratio)
    investment_asset_y = (50/(1 + ratio))
    investment_asset_x = (50 - investment_asset_y)

    dates = assets_y['time'][:min_length]

    print(f"Lengths after trimming - Asset 1: {len(cum_pct_return_asset1)}, Asset 2: {len(cum_pct_return_asset2)}, Spreads: {len(spreads)}, Z-Scores: {len(rolling_z_scores)}")

    results = pd.DataFrame({
    'cum_pct_return_asset1': cum_pct_return_asset1.values,
    'cum_pct_return_asset2': cum_pct_return_asset2.values,
    'residuals': spreads,
    'zscores': rolling_z_scores
}, index=dates)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=results.index, y=rolling_z_scores, mode='lines', name='Z-Score',line=dict(color='green')))
    fig.add_hline(y=1, line_color='green', line_dash='dash', name='Upper Threshold (+2)')
    fig.add_hline(y=0, line_color='black', line_dash='dash', name='Middle Threshold (0)')
    fig.add_hline(y=-1, line_color='red', line_dash='dash', name='Lower Threshold (-2)')

    fig.add_trace(go.Scatter(x=results.index, y=results['cum_pct_return_asset1'], mode='lines', name='Cumulative Returns Asset Y', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=results.index, y=results['cum_pct_return_asset2'], mode='lines', name='Cumulative Returns Asset X', line=dict(color='red')))
    fig.update_layout(title=f'Server {account_info.server} Hedge Ratio {abs(ratio):.2f} asset y {investment_asset_y:.2f} asset x {investment_asset_x:.2f}', xaxis_title='Date')

    fig.show()

    print(f"Current Z-Score: {rolling_z_scores.iloc[-1]} hedge ratio is {ratio}, volume y is {investment_asset_y} and volume x {investment_asset_x}")

async def zscores_calculation_test():
    mt5_conn = MT5Connector()
    highest_zscore_period,total_profit,total_traded_volumes,total_deals = mt5_conn.total_daily_risk()
    positions = mt5_conn.get_open_positions()
    total_grids = len(positions)/2
    total_grids_history = total_deals/2
    updated_zscore_entry = 0.0
    if total_grids > 0.0 or total_grids_history > 0.0:
        print(f"Total open grids: {total_grids} and total grids history: {total_grids_history}")
        grids_total = total_grids + total_grids_history

    print(f"Total grids history: {total_grids_history}, Total traded volumes: {total_traded_volumes}, Total profit: {total_profit}, Highest zscore period: {highest_zscore_period}, Total grids: {total_grids}")
    if grids_total == 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
        updated_zscore_entry = float(highest_zscore_period) + ADDITIONAL_GRID
    elif grids_total == 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + ADDITIONAL_GRID
    elif grids_total > 0.0 and highest_zscore_period > Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = float(highest_zscore_period) + (ADDITIONAL_GRID * total_grids)
    elif grids_total > 0.0 and highest_zscore_period <= Z_SCORE_ENTRY_THRESHOLD:
         updated_zscore_entry = Z_SCORE_ENTRY_THRESHOLD + (ADDITIONAL_GRID * total_grids)

    print(f"Updated z score is {updated_zscore_entry} for highest z score period {highest_zscore_period} and total grids {total_grids}")

    return updated_zscore_entry 

async def get_correlation_test():
    mt5_conn = MT5Connector()
    assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0])
    assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0])
    correlation = get_correlation(assets_y,assets_x)
    print(f"Correlation between {TRADING_PAIR_Y[0]} and {TRADING_PAIR_X[0]} is {correlation}")

asyncio.run(plot_data_prices())