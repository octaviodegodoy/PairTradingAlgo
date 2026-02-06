import asyncio
import logging
from constants import (
    MAGIC_NUMBER,
    START_TIME_HOUR,
    START_TIME_MINUTE,
    TRADE_WINDOW_TIME_HOURS,
    TRADE_WINDOW_TIME_MINUTES
)

from mt5_connector import MT5Connector
from utils import check_trading_time
from strategy import PairTradingStrategy
from trade_manager import TradeManager  # Trade manager not implemented yet
from trade_execution import TradeExecution  # Trade execution not implemented yet

async def main():
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    pair_trading_strategy = PairTradingStrategy()  # Trade manager not implemented yet
    trade_execution = TradeExecution(MAGIC_NUMBER)
    trade_manager = TradeManager()  # Trade manager not implemented yet
    
    try:
        if not mt5_conn.initialize():
            logger.error("MT5 initialization failed")
            return
        
        logger.info(f"MT5 initialized successfully, trading will start at {START_TIME_HOUR}:{START_TIME_MINUTE} UTC for {TRADE_WINDOW_TIME_HOURS} hours and {TRADE_WINDOW_TIME_MINUTES} minutes")
        while True:
            if not check_trading_time():
                positions = mt5_conn.get_open_positions()
                if positions: 
                    mt5_conn.close_all_positions()
                    logger.info("Outside trading hours. Closed all positions.")
                else:
                    await asyncio.sleep(5)
                    continue
            while check_trading_time():
                positions = mt5_conn.get_open_positions()
                if positions:
                    logger.info(f"Currently open positions: {len(positions)}")
                    logger.info("Managing trades...")
                    trade_manager.manage_trades()
                else:
                    logger.info("No open positions currently.")
                    logger.info("Start scanning for trading opportunities...")
                    correlation, hedge_ratio, spreads, rolling_z_scores, pair, arbitrage_found = pair_trading_strategy.scan_pairs_arbitrage()                
                    if arbitrage_found:
                        logger.info(f"Arbitrage was found : {arbitrage_found} for pair y as {pair[0]} and x as {pair[1]} hedge ratio is {hedge_ratio} and spreads length is {len(spreads)} and rolling z scores length is {len(rolling_z_scores)}")
                        logger.info("Arbitrage opportunity detected. Executing trades...")
                        # Here you would add logic to execute trades and manage them
                        # For example:
                        # trade_execution.execute_trade(...)
                        logger.info(f"Starting trade execution for {pair[0]} and {pair[1]}")
                        trade_execution.execute_trade(pair[0], pair[1], correlation,hedge_ratio, rolling_z_scores.iloc[-1])
                
            await asyncio.sleep(15)
    except KeyboardInterrupt:
        logger.info("Terminating script by user.")
    finally:
        mt5_conn.shutdown()
    logger.info("Outside trading hours. Sleeping for 5 seconds.")
    
    await asyncio.sleep(15)

asyncio.run(main())