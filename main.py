import asyncio
import logging
from constants import (
    MAGIC_NUMBER
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

        while True:
            if not check_trading_time():
                logger.info("Outside trading hours. Sleeping for 5 seconds.")
                await asyncio.sleep(5)
                continue
            elif check_trading_time():
                logger.info("Start scanning for trading opportunities...")
                hedge_ratio, spreads, rolling_z_scores, pair, correlation, arbitrage_found = pair_trading_strategy.scan_pairs_arbitrage()                
                if arbitrage_found:
                    logger.info(f"Arbitrage was found : {arbitrage_found} for pair y as {pair[0]} and x as {pair[1]} hedge ratio is {hedge_ratio[-1]} and spreads length is {len(spreads)} and rolling z scores length is {len(rolling_z_scores)}")
                    logger.info("Arbitrage opportunity detected. Executing trades...")
                    # Here you would add logic to execute trades and manage them
                    # For example:
                    # trade_execution.execute_trade(...)
                    logger.info(f"Starting trade execution for {pair[0]} and {pair[1]}")
                    trade_execution.execute_trade(pair[0], pair[1], hedge_ratio[-1],rolling_z_scores[-1], correlation)
                    # trade_manager.manage_trades(...)               
                
                logger.info("Starting trade management...")
                trade_manager.manage_trades()

            await asyncio.sleep(5)
    except KeyboardInterrupt:
        logger.info("Terminating script by user.")
    finally:
        mt5_conn.shutdown()
    logger.info("Outside trading hours. Sleeping for 5 seconds.")
    
    await asyncio.sleep(5)

asyncio.run(main())