import asyncio
from email import parser
import math
from turtle import color
from mt5_connector import MT5Connector
from utils import calculate_volumes, get_dynamic_spread_zscores, get_linear_regression_spread_zscores, check_cointegration, get_correlation, get_half_life, get_vecm_ect_zscore
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import ADDITIONAL_GRID, FIBO_VOLUME_FACTORS, KALMAN_FILTER_METHOD, MARGIN_PERCENT, MARGIN_X, MARGIN_Y, MAX_HALF_LIFE, MAX_RISK, NOISE_VARIANCE, PERIODS, ROLLING_PERIODS, START_TIME_HOUR, START_TIME_MINUTE, TRADE_WINDOW_TIME_HOURS, TRADE_WINDOW_TIME_MINUTES, TRADING_PAIR_Y, TRADING_PAIR_X, VOLUME_FACTOR, Z_SCORE_ENTRY_THRESHOLD
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
                    cointegration_condition = check_cointegration(assets_y, assets_x)
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


def test_calculate_volumes_notional():
    symbolY,symbolX,hedge_ratio,total_positions = "WINJ26","WDOJ26",1.41,0
    """
    Calculate volumes using METHOD 3: Notional Value Balanced
    
    This ensures proper pairs trading by balancing notional exposure according to hedge ratio:
    Notional_X = hedge_ratio × Notional_Y
    
    Args:
        symbolY: Symbol name for asset Y
        symbolX: Symbol name for asset X
        hedge_ratio: Beta/slope from regression
        min_lot_Y: Minimum volume for asset Y
        min_lot_X: Minimum volume for asset X
        total_max_lots: Total available margin/lots budget
        total_positions: Current number of open positions
        
    Returns:
        volume_y, volume_x: Calculated volumes for each asset
    """
    import MetaTrader5 as mt5
    
    print(f"Volume calculation for {symbolY} and {symbolX} with hedge ratio {hedge_ratio}")

    # Get symbol information
    info_y = mt5.symbol_info(symbolY)
    info_x = mt5.symbol_info(symbolX)
    print(f"Symbol info {symbolY}: {info_y}")
    print(f"Symbol info {symbolX}: {info_x}")
    min_lot_X = float(info_x.volume_min) 
    min_lot_Y = float(info_y.volume_min)
    total_max_lots = mt5.account_info().margin_free / 0.5  # Example: use margin free as budget
    print(f"Min lot {symbolY}: {min_lot_Y}, Min lot {symbolX}: {min_lot_X}")    
    # Get current prices and contract sizes
    price_y = info_y.ask if info_y.ask > 0 else info_y.last
    price_x = info_x.ask if info_x.ask > 0 else info_x.last
    contract_y = info_y.trade_contract_size
    print(f"Contract size {symbolY}: {contract_y}")
    contract_x = info_x.trade_contract_size
    print(f"Contract size {symbolX}: {contract_x}")
    
    # Calculate margin per lot for each asset
    margin_y_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbolY, 1.0, price_y)
    print(f"Margin per lot {symbolY}: {margin_y_per_lot}")
    margin_x_per_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbolX, 1.0, price_x)
    print(f"Margin per lot {symbolX}: {margin_x_per_lot}")
    
    # Apply grid/position scaling
    grid_count = (total_positions/2)
    fibo_index = min(int(grid_count), len(FIBO_VOLUME_FACTORS) - 1)
    fibo_multiplier = FIBO_VOLUME_FACTORS[fibo_index]
    
    # Calculate available risk for this trade (scaled by grid position and volume factor)
    risk_per_trade = (total_max_lots / VOLUME_FACTOR) * fibo_multiplier
    
    # METHOD 3: Notional Value Balanced
    # We want: Notional_X = hedge_ratio × Notional_Y
    # Where: Notional = Price × Contract_Size × Volume
    # Therefore: (price_x × contract_x × volume_x) = hedge_ratio × (price_y × contract_y × volume_y)
    
    abs_hedge_ratio = abs(hedge_ratio)
    
    # Calculate the notional ratio
    # volume_x = notional_ratio × volume_y
    notional_ratio = abs_hedge_ratio * (price_y * contract_y) / (price_x * contract_x)
    
    # Solve for volume_y:
    # margin_y × volume_y + margin_x × notional_ratio × volume_y = risk_per_trade
    # volume_y × (margin_y + margin_x × notional_ratio) = risk_per_trade
    # volume_y = risk_per_trade / (margin_y + margin_x × notional_ratio)
    
    denominator = margin_y_per_lot + (margin_x_per_lot * notional_ratio)
    
    if denominator > 0:
        volume_y_calculated = risk_per_trade / denominator
        volume_x_calculated = volume_y_calculated * notional_ratio
    else:
        # Fallback to equal allocation if calculation fails
        print(f"Warning: Invalid denominator in METHOD 3 calculation, using equal allocation")
        volume_y_calculated = risk_per_trade / (2 * margin_y_per_lot)
        volume_x_calculated = risk_per_trade / (2 * margin_x_per_lot)
    
    # Round to volume steps
    volume_y = round(volume_y_calculated / info_y.volume_step) * info_y.volume_step
    volume_x = round(volume_x_calculated / info_x.volume_step) * info_x.volume_step
    
    # Enforce minimum volumes
    volume_y = max(volume_y, min_lot_Y)
    volume_x = max(volume_x, min_lot_X)
    
    # Enforce maximum volumes
    volume_y = min(volume_y, info_y.volume_max)
    volume_x = min(volume_x, info_x.volume_max)
    
    # Calculate actual notional values for verification
    notional_y = price_y * contract_y * volume_y
    notional_x = price_x * contract_x * volume_x
    notional_ratio_actual = notional_x / notional_y if notional_y > 0 else 0
    
    # Calculate actual margin usage
    margin_used_y = volume_y * margin_y_per_lot
    margin_used_x = volume_x * margin_x_per_lot
    total_margin_used = margin_used_y + margin_used_x
    
    print(f"Grid count: {grid_count}, Fibo index: {fibo_index}, Multiplier: {fibo_multiplier}")
    print(f"Risk per trade: ${risk_per_trade:.2f}")
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"Volume {symbolY}: {volume_y:.2f} lots (notional: ${notional_y:.2f}, margin: ${margin_used_y:.2f})")
    print(f"Volume {symbolX}: {volume_x:.2f} lots (notional: ${notional_x:.2f}, margin: ${margin_used_x:.2f})")
    print(f"Notional ratio: {notional_ratio_actual:.4f} (target: {abs_hedge_ratio:.4f})")
    print(f"Total margin used: ${total_margin_used:.2f}")
    
    return float(volume_y), float(volume_x)

async def test_calculate_volumes():

    """
    Calculate volumes using METHOD 3: Notional Value Balanced
    
    This ensures proper pairs trading by balancing notional exposure according to hedge ratio:
    Notional_X = hedge_ratio × Notional_Y
    
    Args:
        symbolY: Symbol name for asset Y
        symbolX: Symbol name for asset X
        hedge_ratio: Beta/slope from regression
        min_lot_Y: Minimum volume for asset Y
        min_lot_X: Minimum volume for asset X
        total_max_lots: Total available margin/lots budget
        total_positions: Current number of open positions
        
    Returns:
        volume_y, volume_x: Calculated volumes for each asset
    """
    mt5_conn = MT5Connector()
    symbolY = 'WINJ26'
    symbolX = 'WDOJ26'
    total_positions = 0
    hedge_ratio = 1.41  # Example hedge ratio

    
    print(f"Volume calculation for {symbolY} and {symbolX} with hedge ratio {hedge_ratio}")

    # Get symbol information
    info_y = mt5_conn.get_symbol_info(symbolY)
    info_x = mt5_conn.get_symbol_info(symbolX)
    min_lot_X = float(info_x.volume_min) 
    min_lot_Y = float(info_y.volume_min)
    total_max_lots = mt5_conn.get_account_info().margin_free / 0.5  # Example: use margin free as budget
    print(f"Min lot {symbolY}: {min_lot_Y}, Min lot {symbolX}: {min_lot_X}")    
    # Get current prices and contract sizes
    price_y = info_y.ask if info_y.ask > 0 else info_y.last
    price_x = info_x.ask if info_x.ask > 0 else info_x.last
    contract_y = info_y.trade_contract_size
    print(f"Contract size {symbolY}: {contract_y}")
    contract_x = info_x.trade_contract_size
    print(f"Contract size {symbolX}: {contract_x}")
    
    # Calculate margin per lot for each asset
    margin_y_per_lot = mt5_conn.get_order_calc_margin(mt5_conn.ORDER_TYPE_BUY, symbolY, 1.0, price_y)
    print(f"Margin per lot {symbolY}: {margin_y_per_lot}")
    margin_x_per_lot = mt5_conn.get_order_calc_margin(mt5_conn.ORDER_TYPE_BUY, symbolX, 1.0, price_x)
    print(f"Margin per lot {symbolX}: {margin_x_per_lot}")
    
    # Apply grid/position scaling
    grid_count = (total_positions/2)
    fibo_index = min(int(grid_count), len(FIBO_VOLUME_FACTORS) - 1)
    fibo_multiplier = FIBO_VOLUME_FACTORS[fibo_index]
    
    # Calculate available risk for this trade (scaled by grid position and volume factor)
    risk_per_trade = (total_max_lots / VOLUME_FACTOR) * fibo_multiplier
    
    # METHOD 3: Notional Value Balanced
    # We want: Notional_X = hedge_ratio × Notional_Y
    # Where: Notional = Price × Contract_Size × Volume
    # Therefore: (price_x × contract_x × volume_x) = hedge_ratio × (price_y × contract_y × volume_y)
    
    abs_hedge_ratio = abs(hedge_ratio)
    
    # Calculate the notional ratio
    # volume_x = notional_ratio × volume_y
    notional_ratio = abs_hedge_ratio * (price_y * contract_y) / (price_x * contract_x)
    
    # Solve for volume_y:
    # margin_y × volume_y + margin_x × notional_ratio × volume_y = risk_per_trade
    # volume_y × (margin_y + margin_x × notional_ratio) = risk_per_trade
    # volume_y = risk_per_trade / (margin_y + margin_x × notional_ratio)
    
    denominator = margin_y_per_lot + (margin_x_per_lot * notional_ratio)
    
    if denominator > 0:
        volume_y_calculated = risk_per_trade / denominator
        volume_x_calculated = volume_y_calculated * notional_ratio
    else:
        # Fallback to equal allocation if calculation fails
        print(f"Warning: Invalid denominator in METHOD 3 calculation, using equal allocation")
        volume_y_calculated = risk_per_trade / (2 * margin_y_per_lot)
        volume_x_calculated = risk_per_trade / (2 * margin_x_per_lot)
    
    # Round to volume steps
    volume_y = round(volume_y_calculated / info_y.volume_step) * info_y.volume_step
    volume_x = round(volume_x_calculated / info_x.volume_step) * info_x.volume_step
    
    # Enforce minimum volumes
    volume_y = max(volume_y, min_lot_Y)
    volume_x = max(volume_x, min_lot_X)
    
    # Enforce maximum volumes
    volume_y = min(volume_y, info_y.volume_max)
    volume_x = min(volume_x, info_x.volume_max)
    
    # Calculate actual notional values for verification
    notional_y = price_y * contract_y * volume_y
    notional_x = price_x * contract_x * volume_x
    notional_ratio_actual = notional_x / notional_y if notional_y > 0 else 0
    
    # Calculate actual margin usage
    margin_used_y = volume_y * margin_y_per_lot
    margin_used_x = volume_x * margin_x_per_lot
    total_margin_used = margin_used_y + margin_used_x
    
    print(f"Grid count: {grid_count}, Fibo index: {fibo_index}, Multiplier: {fibo_multiplier}")
    print(f"Risk per trade: ${risk_per_trade:.2f}")
    print(f"Hedge ratio: {hedge_ratio:.4f}")
    print(f"Volume {symbolY}: {volume_y:.2f} lots (notional: ${notional_y:.2f}, margin: ${margin_used_y:.2f})")
    print(f"Volume {symbolX}: {volume_x:.2f} lots (notional: ${notional_x:.2f}, margin: ${margin_used_x:.2f})")
    print(f"Notional ratio: {notional_ratio_actual:.4f} (target: {abs_hedge_ratio:.4f})")
    print(f"Total margin used: ${total_margin_used:.2f}")
    
    return float(volume_y), float(volume_x)

async def test_open_positions():
    """Check if the account has any open positions managed by this trading algorithm (MAGIC_NUMBER)."""
    from constants import MAGIC_NUMBER
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    positions = mt5_conn.get_open_positions()  # Already filtered by MAGIC_NUMBER

    if not positions:
        print(f"No open positions found for this algorithm (magic={MAGIC_NUMBER})")
    else:
        print(f"Found {len(positions)} open position(s) for magic={MAGIC_NUMBER}:")
        for pos in positions:
            direction = "BUY" if pos.type == 0 else "SELL"
            print(
                f"  Ticket={pos.ticket} | Symbol={pos.symbol} | {direction} | "
                f"Volume={pos.volume} | Open Price={pos.price_open:.5f} | "
                f"Profit={pos.profit:.2f} | Comment='{pos.comment}'"
            )

    mt5_conn.shutdown()

async def backtest_strategy():
    """
    Backtests the pair trading strategy from main.py using the current constants parameters.

    Entry logic mirrors main.py / PairTradingStrategy.scan_pairs_arbitrage():
      - Compute spread and z-score via Kalman Filter (KALMAN_FILTER_METHOD=True) or OLS.
      - Enter when |z_score| > Z_SCORE_ENTRY_THRESHOLD AND half-life < MAX_HALF_LIFE
        AND ADF cointegration test passes.
      - Direction: z < -threshold → long spread (BUY Y / SELL X for positive correlation);
                   z > +threshold → short spread (SELL Y / BUY X).
    Exit logic:
      - z-score crosses zero (mean reversion complete).
      - Stop-loss: |z_score| > 2.5 × Z_SCORE_ENTRY_THRESHOLD (adverse move).
      - Force-close at the last bar.

    Volumes are sized the same way as TradeExecution.execute_trade():
      grid_lot_investment = equity * MARGIN_PERCENT / VOLUME_FACTOR
      vol_x = grid_lot_investment / (1 + |hedge_ratio|)
      vol_y = grid_lot_investment - vol_x

    Reports: total trades, win rate, total P&L (points), total return,
             max drawdown and Sharpe ratio. Plots equity curve, z-scores
             and prices.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()

    # ── Date filter (set to None to use all available bars) ───────────────────
    BACKTEST_START = "2024-04-01"   # e.g. pd.Timestamp("2025-01-01")
    BACKTEST_END   = "2026-04-30"   # e.g. pd.Timestamp("2025-12-31")
    # ─────────────────────────────────────────────────────────────────────────

    if not mt5_conn.initialize():
        logger.error("MT5 initialization failed")
        return

    logger.info("MT5 initialized. Starting backtest...")

    try:
        # Fetch enough bars: 60-bar warmup + full simulation window.
        # 500 bars ≈ 2 years of daily data — covers any reasonable BACKTEST_START/END range.
        FETCH_BARS = 500
        # Session window (UTC) — mirrors check_trading_time() in utils.py.
        # Live strategy never carries positions overnight: force-close fires at session end.
        # Each D1 bar = one trading session → every bar triggers end-of-session close.
        SESSION_START_MIN = START_TIME_HOUR * 60 + START_TIME_MINUTE
        SESSION_END_MIN   = SESSION_START_MIN + TRADE_WINDOW_TIME_HOURS * 60 + TRADE_WINDOW_TIME_MINUTES
        assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0], n_bars=FETCH_BARS)
        assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0], n_bars=FETCH_BARS)

        min_required = PERIODS  # need at least 60 bars of history before simulating
        if assets_y is None or assets_x is None or len(assets_y) < min_required or len(assets_x) < min_required:
            logger.error(f"Insufficient data: need at least {min_required} bars (got {len(assets_y) if assets_y is not None else 0})")
            return

        min_len = min(len(assets_y), len(assets_x))
        prices_y  = assets_y['close'].values[:min_len]   # session-close  (z-score / indicators)
        prices_x  = assets_x['close'].values[:min_len]
        opens_y   = assets_y['open'].values[:min_len]    # session-open   (trade entry price)
        opens_x   = assets_x['open'].values[:min_len]
        dates     = pd.to_datetime(assets_y['time'].values[:min_len])

        # Trading window: START_TIME_HOUR:START_TIME_MINUTE UTC, duration TRADE_WINDOW_TIME_HOURS:TRADE_WINDOW_TIME_MINUTES
        # With D1 bars each bar = one trading session → positions must be closed by session end (no overnight carry).
        # Entry uses session open; end-of-session force-close uses session close.
        session_start_utc = f"{START_TIME_HOUR:02d}:{START_TIME_MINUTE:02d} UTC"
        session_end_minutes = START_TIME_HOUR * 60 + START_TIME_MINUTE + TRADE_WINDOW_TIME_HOURS * 60 + TRADE_WINDOW_TIME_MINUTES
        session_end_utc = f"{session_end_minutes // 60:02d}:{session_end_minutes % 60:02d} UTC"
        logger.info(f"Trading window: {session_start_utc} → {session_end_utc}  (no overnight carry enforced)")

        logger.info(f"Data loaded: {min_len} bars | {TRADING_PAIR_Y[0]} / {TRADING_PAIR_X[0]} | "
                    f"range {pd.to_datetime(assets_y['time'].values[0]).date()} → "
                    f"{pd.to_datetime(assets_y['time'].values[min_len-1]).date()}")

        # Compute spread and z-scores over the full series (all bars used for warmup)
        full_df_y = pd.DataFrame({'close': prices_y, 'time': dates})
        full_df_x = pd.DataFrame({'close': prices_x, 'time': dates})

        if KALMAN_FILTER_METHOD:
            results = get_dynamic_spread_zscores(full_df_y, full_df_x)
        else:
            results = get_linear_regression_spread_zscores(full_df_y, full_df_x)

        z_scores = results['z_scores'].values
        hedge_ratios = results['hedge_ratio'].values
        spreads_series = pd.Series(results['spread'].values)

        # Determine simulation window — date filter restricts trading, not indicator warmup
        sim_start = 0
        sim_end = min_len
        if BACKTEST_START is not None:
            start_ts = pd.Timestamp(BACKTEST_START)
            idx = np.searchsorted(dates, start_ts)
            sim_start = max(sim_start, int(idx))
        if BACKTEST_END is not None:
            end_ts = pd.Timestamp(BACKTEST_END)
            idx = np.searchsorted(dates, end_ts, side='right')
            sim_end = min(sim_end, int(idx))

        sim_bars = sim_end - sim_start
        if sim_bars < 1:
            logger.error(f"No bars found in date range {BACKTEST_START} → {BACKTEST_END}")
            return
        logger.info(f"Simulation window: {dates[sim_start].date()} → {dates[sim_end - 1].date()} ({sim_bars} bars)")

        # Pre-compute half-life and cointegration on the full spread for reporting only.
        # The actual entry gate uses a rolling per-bar check (see simulation loop).
        half_life = get_half_life(spreads_series)
        cointegration_ok = check_cointegration(full_df_y, full_df_x)
        logger.info(f"Half-life: {half_life:.2f} | Cointegrated: {cointegration_ok} | "
                    f"Kalman: {KALMAN_FILTER_METHOD} | Z threshold: {Z_SCORE_ENTRY_THRESHOLD}")

        # Note: no global abort — the simulation loop gates each entry individually.

        # ── Walk-forward simulation ────────────────────────────────────────────
        # Correlation sign determines order types, mirroring trade_execution.py:
        #   correlation > 0 → long_spread: BUY Y / SELL X  |  short_spread: SELL Y / BUY X
        #   correlation < 0 → long_spread: BUY Y / BUY X   |  short_spread: SELL Y / SELL X
        # Correlation is computed at each entry using the preceding PERIODS bars,
        # exactly as the live strategy does via get_correlation(assets_y, assets_x)
        # where assets_y/assets_x are fetched with n_bars=PERIODS (default 60).

        # Seed equity from the real MT5 account so P&L is in account currency (BRL)
        account_info = mt5_conn.get_account_info()
        INITIAL_EQUITY = float(account_info.equity) if account_info is not None else 10_000.0
        logger.info(f"Account equity (backtest seed): {INITIAL_EQUITY:.2f}")
        equity = INITIAL_EQUITY
        equity_curve = [equity]

        trades = []
        in_trade = False
        direction = None          # 'long_spread' | 'short_spread'
        entry_price_y = None
        entry_price_x = None
        entry_bar = None          # bar index of entry
        inv_y = 0.0               # monetary allocation at entry (equity fraction)
        inv_x = 0.0
        x_long_sign = -1          # will be set at each entry
        x_short_sign = 1
        max_loss_pts = 0.0        # set at each entry: equity * MARGIN_PERCENT * MAX_RISK

        for i in range(sim_start, sim_end):
            # ── Walk-forward causality ──────────────────────────────────────
            # z[i] and hedge_ratio[i] are computed from close[i] (end of bar i).
            # Entry happens at open[i], so the decision MUST use information
            # available BEFORE bar i opens — i.e. z[i-1] / hedge_ratio[i-1].
            # Using z[i] for the open[i] entry is look-ahead bias and
            # systematically loses money: a low z[i] means the spread fell
            # during bar i, so you'd be buying at open (the high) and
            # selling at close (the low). This is exactly the inverted-PnL
            # symptom of a backtest that uses end-of-bar data for an
            # open-of-bar entry decision.
            if i == 0:
                equity_curve.append(equity)
                continue
            z_signal      = z_scores[i - 1]   # signal from previous close
            hedge_ratio_i = float(hedge_ratios[i - 1])
            z_now         = z_scores[i]       # used only for exit (mark-to-market)
            if np.isnan(z_signal):
                equity_curve.append(equity)
                continue
            # Volume sizing mirrors TradeExecution / strategy logic
            grid_lot_investment = equity * MARGIN_PERCENT / VOLUME_FACTOR
            inv_x = grid_lot_investment / (1 + abs(hedge_ratio_i))
            inv_y = grid_lot_investment - inv_x

            if not in_trade:
                if z_signal < -Z_SCORE_ENTRY_THRESHOLD or z_signal > Z_SCORE_ENTRY_THRESHOLD:
                    # Compute correlation over the same PERIODS lookback the live strategy uses
                    # (use bars STRICTLY BEFORE i to avoid look-ahead).
                    corr_start = max(0, i - PERIODS)
                    corr_y = pd.Series(prices_y[corr_start:i])
                    corr_x = pd.Series(prices_x[corr_start:i])
                    entry_correlation = corr_y.corr(corr_x)
                    x_long_sign  =  1 if entry_correlation < 0 else -1
                    x_short_sign = -1 if entry_correlation < 0 else  1
                    entry_dir = 'long_spread' if z_signal < -Z_SCORE_ENTRY_THRESHOLD else 'short_spread'
                    in_trade = True
                    direction = entry_dir
                    entry_bar = i
                    # Entry at session OPEN (trade_manager starts at session open, not close)
                    entry_price_y = opens_y[i]
                    entry_price_x = opens_x[i]
                    # inv_y / inv_x = monetary allocation (equity fraction); used for % PnL calc
                    # max_loss mirrors trade_manager.py: equity * MARGIN_PERCENT * MAX_RISK
                    entry_equity = equity
                    max_loss_pts = entry_equity * MARGIN_PERCENT * MAX_RISK
                    logger.info(
                        f"ENTRY bar={i} date={dates[i].date()} dir={direction} "
                        f"z_signal={z_signal:.3f} (from close[{i-1}]) corr={entry_correlation:.4f} "
                        f"x_sign={'BUY' if x_long_sign==1 else 'SELL'} X  "
                        f"max_loss={max_loss_pts:.4f}"
                    )
                    trades.append({
                        'entry_bar': i, 'entry_date': dates[i],
                        'entry_z': z_signal, 'direction': direction,
                        'entry_correlation': entry_correlation,
                        'order_y': 'BUY' if direction == 'long_spread' else 'SELL',
                        'order_x': ('BUY' if x_long_sign == 1 else 'SELL') if direction == 'long_spread'
                                   else ('SELL' if x_short_sign == -1 else 'BUY'),
                    })

            # ── Exit checks ───────────────────────────────────────────────────
            # D1 bars: one bar = one trading session (12:05→19:35 UTC for B3).
            # The live strategy calls check_trading_time() and force-closes at session end —
            # no overnight carry. Mirroring that: end_of_session fires on every bar.
            # Mean reversion within the same session (open→close z-cross) is captured;
            # stop-loss is evaluated at the session close mark.
            if in_trade:
                # Mark-to-market at session close
                exit_price_y = prices_y[i]   # close ≈ end-of-session price
                exit_price_x = prices_x[i]
                ret_y = (exit_price_y - entry_price_y) / entry_price_y
                ret_x = (exit_price_x - entry_price_x) / entry_price_x
                if direction == 'long_spread':
                    unrealized_pnl = ret_y * inv_y + x_long_sign * ret_x * inv_x
                else:
                    unrealized_pnl = -ret_y * inv_y + x_short_sign * ret_x * inv_x

                stop_loss_hit    = unrealized_pnl <= -max_loss_pts
                mean_reversion   = (
                    (direction == 'long_spread'  and z_now >= 0) or
                    (direction == 'short_spread' and z_now <= 0)
                )
                # end_of_session: live strategy force-closes at 19:35 UTC (session end).
                # With D1 bars every bar represents exactly one session → always True.
                end_of_session   = True
                force_close_data = i == sim_end - 1

                if mean_reversion or stop_loss_hit or end_of_session or force_close_data:
                    pnl = unrealized_pnl
                    equity += pnl
                    if mean_reversion:
                        exit_reason = 'reversion'
                    elif stop_loss_hit:
                        exit_reason = 'stop_loss'
                    elif end_of_session:
                        exit_reason = 'end_of_session'
                    else:
                        exit_reason = 'end_of_data'
                    trades[-1].update({
                        'exit_bar': i, 'exit_date': dates[i],
                        'exit_z': z_now, 'pnl': pnl,
                        'hold_bars': i - entry_bar,
                        'is_win': pnl > 0, 'exit_reason': exit_reason,
                    })
                    in_trade  = False
                    direction = None
                    entry_bar = None
                    max_loss_pts = 0.0

            equity_curve.append(equity)

        # ── Performance metrics ────────────────────────────────────────────────
        closed = [t for t in trades if 'pnl' in t]
        total_trades = len(closed)
        wins = sum(1 for t in closed if t['is_win'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl = sum(t['pnl'] for t in closed)
        total_return = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100

        eq_series = pd.Series(equity_curve)
        rolling_max = eq_series.cummax()
        max_drawdown = ((eq_series - rolling_max) / rolling_max).min() * 100

        pnl_series = pd.Series([t['pnl'] for t in closed])
        sharpe = (
            (pnl_series.mean() / pnl_series.std() * np.sqrt(252))
            if len(pnl_series) > 1 and pnl_series.std() > 0 else 0.0
        )
        avg_pnl = pnl_series.mean() if len(pnl_series) > 0 else 0.0
        stop_losses     = sum(1 for t in closed if t.get('exit_reason') == 'stop_loss')
        end_of_sessions = sum(1 for t in closed if t.get('exit_reason') == 'end_of_session')
        reversions      = sum(1 for t in closed if t.get('exit_reason') == 'reversion')

        print("\n========== BACKTEST RESULTS ==========")
        print(f"Pair               : {TRADING_PAIR_Y[0]} (Y) / {TRADING_PAIR_X[0]} (X)")
        print(f"Bars (total/sim)   : {min_len} / {sim_bars}  ({dates[sim_start].date()} → {dates[sim_end-1].date()})")
        print(f"Method             : {'Kalman Filter' if KALMAN_FILTER_METHOD else 'OLS Rolling'}")
        print(f"Z-Score threshold  : {Z_SCORE_ENTRY_THRESHOLD}")
        print(f"Half-life (full)   : {half_life:.2f}")
        print(f"Cointegrated (full): {cointegration_ok}")
        print(f"--------------------------------------")
        print(f"Total trades       : {total_trades}")
        print(f"  Wins             : {wins}  ({win_rate:.1f}%)")
        print(f"  Reversions       : {reversions}")
        print(f"  Stop-losses      : {stop_losses}")
        print(f"  End-of-session   : {end_of_sessions}  (force-closed at session end, no overnight carry)")
        print(f"Avg P&L / trade    : {avg_pnl:.4f}")
        print(f"Total P&L (points) : {total_pnl:.4f}")
        print(f"Total return       : {total_return:.2f}%")
        print(f"Max drawdown       : {max_drawdown:.2f}%")
        print(f"Sharpe ratio       : {sharpe:.2f}")
        print("--------------------------------------")
        print(f"{'#':<3} {'Entry':<12} {'Exit':<12} {'Dir':<12} {TRADING_PAIR_Y[0]:<8} {TRADING_PAIR_X[0]:<8} {'Corr':>6} {'EntryZ':>7} {'ExitZ':>7} {'P&L':>10} {'Reason'}")
        for idx, t in enumerate(closed, 1):
            print(
                f"{idx:<3} {str(t['entry_date'].date()):<12} {str(t['exit_date'].date()):<12} "
                f"{t['direction']:<12} {t.get('order_y','?'):<8} {t.get('order_x','?'):<8} "
                f"{t.get('entry_correlation', float('nan')):>6.3f} "
                f"{t['entry_z']:>7.3f} {t['exit_z']:>7.3f} "
                f"{t['pnl']:>10.4f} {t.get('exit_reason','?')}"
            )
        print("======================================\n")

        # ── Plots ──────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), layout='constrained')
        fig.suptitle(
            f'Backtest: {TRADING_PAIR_Y[0]} / {TRADING_PAIR_X[0]} | '
            f'{"Kalman" if KALMAN_FILTER_METHOD else "OLS"} | '
            f'Threshold={Z_SCORE_ENTRY_THRESHOLD}'
        )

        # Panel 1 – Price series (sim window only)
        plot_dates = dates[sim_start:sim_end]
        ax1a = axes[0]
        ax1b = ax1a.twinx()
        ax1a.plot(plot_dates, prices_y[sim_start:sim_end], color='blue', label=TRADING_PAIR_Y[0])
        ax1b.plot(plot_dates, prices_x[sim_start:sim_end], color='orange', alpha=0.75, label=TRADING_PAIR_X[0])
        ax1a.set_ylabel(f'{TRADING_PAIR_Y[0]} Price', color='blue')
        ax1b.set_ylabel(f'{TRADING_PAIR_X[0]} Price', color='orange')
        ax1a.set_title('Price Series')
        ax1a.legend(loc='upper left')
        ax1b.legend(loc='upper right')
        ax1a.grid(True)

        # Panel 2 – Z-scores with trade entry markers
        axes[1].plot(plot_dates, z_scores[sim_start:sim_end], color='purple', linewidth=0.8, label='Z-Score')
        axes[1].axhline(Z_SCORE_ENTRY_THRESHOLD, color='green', linestyle='--',
                        linewidth=0.9, label=f'+{Z_SCORE_ENTRY_THRESHOLD}')
        axes[1].axhline(-Z_SCORE_ENTRY_THRESHOLD, color='red', linestyle='--',
                        linewidth=0.9, label=f'-{Z_SCORE_ENTRY_THRESHOLD}')
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.4)
        for t in closed:
            color_e = 'green' if t['is_win'] else 'red'
            axes[1].axvline(t['entry_date'], color=color_e, alpha=0.25, linewidth=0.7)
        axes[1].set_ylabel('Z-Score')
        axes[1].set_title(
            f'Z-Scores | Trades: {total_trades} | Wins: {wins} ({win_rate:.1f}%) | '
            f'Stop-losses: {stop_losses}'
        )
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        # Panel 3 – Equity curve
        # equity_curve[0] is the seed value before bar sim_start; align dates accordingly
        eq_dates = dates[sim_start: sim_end]
        eq_values = equity_curve[1: len(eq_dates) + 1]  # drop seed, match length
        axes[2].plot(eq_dates, eq_values, color='navy', label='Equity')
        axes[2].axhline(INITIAL_EQUITY, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('Equity (points)')
        axes[2].set_title(
            f'Equity Curve | Return: {total_return:.2f}% | '
            f'Max DD: {max_drawdown:.2f}% | Sharpe: {sharpe:.2f}'
        )
        axes[2].legend()
        axes[2].grid(True)

        plt.show()

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
    finally:
        mt5_conn.shutdown()
        logger.info("MT5 shutdown.")


async def analyze_vecm_threshold():
    """
    Fetch full historical WIN*/WDO* data, compute the VECM ECT z-score series
    over all available bars, and print a statistical report to help calibrate
    VECM_ECT_THRESHOLD in constants.py.

    Outputs:
      - Descriptive statistics (mean, std, min/max, skew, kurtosis)
      - Percentile table (|ECT z-score| at 50/75/80/85/90/95/99 %)
      - Frequency table: how often |z| exceeds candidate thresholds
      - Plot: ECT z-score time series + histogram
    """
    from constants import ROLLING_PERIODS, TRADING_PAIR_Y, TRADING_PAIR_X

    logging.basicConfig(level=logging.WARNING)
    mt5_conn = MT5Connector()
    if not mt5_conn.initialize():
        print("MT5 initialization failed")
        return

    try:
        # Fetch maximum available history (2000 bars)
        assets_y = mt5_conn.get_data_futures_btg(TRADING_PAIR_Y[0], n_bars=2000)
        assets_x = mt5_conn.get_data_futures_btg(TRADING_PAIR_X[0], n_bars=2000)

        min_len = min(len(assets_y), len(assets_x))
        if min_len < ROLLING_PERIODS + 10:
            print(f"Not enough data: only {min_len} bars available.")
            return

        assets_y = assets_y.iloc[:min_len].reset_index(drop=True)
        assets_x = assets_x.iloc[:min_len].reset_index(drop=True)

        log_y = np.log(assets_y['close'].values)
        log_x = np.log(assets_x['close'].values)
        dates_arr = pd.to_datetime(assets_y['time'].values, unit='s') if assets_y['time'].dtype != 'datetime64[ns]' else assets_y['time'].values

        # Johansen cointegration — get long-run beta once on full sample
        data_matrix = np.column_stack([log_y, log_x])
        johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
        ev = johansen_result.evec[:, 0]
        beta = ev[0] / ev[1]
        print(f"\nJohansen cointegrating vector: beta = {beta:.6f}")
        print(f"(ECT = log({TRADING_PAIR_Y[0]}) - {beta:.4f} * log({TRADING_PAIR_X[0]}))")

        # Build full ECT series and rolling z-score
        ect = pd.Series(log_y - beta * log_x, index=dates_arr)
        rolling_mean = ect.rolling(window=ROLLING_PERIODS).mean()
        rolling_std  = ect.rolling(window=ROLLING_PERIODS).std()
        ect_z = ((ect - rolling_mean) / rolling_std).dropna()

        abs_z = ect_z.abs()

        # ── Statistical report ──────────────────────────────────────────────
        print(f"\n{'='*52}")
        print(f"  VECM ECT Z-Score Analysis: {TRADING_PAIR_Y[0]} / {TRADING_PAIR_X[0]}")
        print(f"  Bars used: {len(ect_z)}  (rolling window: {ROLLING_PERIODS})")
        print(f"{'='*52}")
        desc = ect_z.describe()
        print(f"  Mean   : {desc['mean']:+.4f}")
        print(f"  Std    : {desc['std']:.4f}")
        print(f"  Min    : {desc['min']:+.4f}")
        print(f"  Max    : {desc['max']:+.4f}")
        print(f"  Skew   : {ect_z.skew():+.4f}")
        print(f"  Kurt   : {ect_z.kurtosis():+.4f}")

        print(f"\n  |ECT z-score| percentiles:")
        for pct in [50, 75, 80, 85, 90, 95, 99]:
            print(f"    {pct:>3}th percentile : {np.percentile(abs_z, pct):.4f}")

        print(f"\n  Frequency above candidate thresholds (% of bars):")
        print(f"  {'Threshold':>10}  {'Bars above':>10}  {'% of total':>10}  {'Trades/year (252d)':>18}")
        for thr in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            above = int((abs_z > thr).sum())
            pct_above = 100.0 * above / len(abs_z)
            trades_yr = above / (len(ect_z) / 252) if len(ect_z) >= 252 else above
            print(f"  {thr:>10.2f}  {above:>10}  {pct_above:>9.2f}%  {trades_yr:>18.1f}")
        print(f"{'='*52}\n")

        # ── Plots ──────────────────────────────────────────────────────────
        from constants import VECM_ECT_THRESHOLD
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
        fig.suptitle(f'VECM ECT Z-Score — {TRADING_PAIR_Y[0]} / {TRADING_PAIR_X[0]}  |  beta={beta:.4f}')

        # Panel 1: time series
        axes[0].plot(ect_z.index, ect_z.values, color='steelblue', linewidth=0.8, label='ECT z-score')
        for thr, col in [(VECM_ECT_THRESHOLD, 'red'), (1.0, 'orange'), (2.0, 'green')]:
            axes[0].axhline( thr, color=col, linestyle='--', linewidth=0.8, label=f'+{thr}')
            axes[0].axhline(-thr, color=col, linestyle='--', linewidth=0.8)
        axes[0].axhline(0, color='black', alpha=0.3)
        axes[0].set_ylabel('ECT Z-Score')
        axes[0].set_title('ECT Z-Score over time')
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        # Panel 2: histogram of |z|
        axes[1].hist(abs_z.values, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
        for thr, col in [(VECM_ECT_THRESHOLD, 'red'), (1.0, 'orange'), (2.0, 'green')]:
            axes[1].axvline(thr, color=col, linestyle='--', linewidth=1.2, label=f'|z|={thr}')
        axes[1].set_xlabel('|ECT Z-Score|')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of |ECT Z-Score|  (current threshold shown in red)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True)

        plt.show()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        mt5_conn.shutdown()


asyncio.run(plot_data_prices())