import asyncio
import logging
from config import (
    PERIODS, ROLLING_PERIODS, MARGIN_Y, MARGIN_X, TRAILING_DISTANCE_POINTS,
    PROFIT_THRESHOLD_MULTIPLIER, MARGIN_PERCENT, MAX_POSITIONS, MAX_RISK,
    MIN_ZSCORE, MAX_HALF_LIFE, MAGIC_NUMBER
)

from mt5_connector import MT5Connector
from utils import Utils
from strategy import PairTradingStrategy
from trade_manager import TradeManager  # Trade manager not implemented yet
from trade_execution import TradeExecution  # Trade execution not implemented yet
import time

async def main():
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    utils = Utils()
    pair_trading_strategy = PairTradingStrategy(MAX_HALF_LIFE,MIN_ZSCORE)  # Trade manager not implemented yet
    trade_execution = TradeExecution(MAGIC_NUMBER)
    trade_manager = TradeManager(MAGIC_NUMBER)  # Trade manager not implemented yet
    
    try:
        if not mt5_conn.initialize():
            logger.error("MT5 initialization failed")
            return

        while True:
            if not utils.check_trading_time():
                logger.info("Outside trading hours. Sleeping for 5 seconds.")
                await asyncio.sleep(5)
                continue
            elif utils.check_trading_time():
                logger.info("Start scanning for trading opportunities...")
                arbitrage = pair_trading_strategy.scan_pairs_arbitrage()
                print(f"Arbitrage was found : {arbitrage}")
                if arbitrage:
                    logger.info("Arbitrage opportunity detected. Executing trades...")
                    # Here you would add logic to execute trades and manage them
                    # For example:
                    # trade_execution.execute_trade(...)
                    trade_execution.execute_trade("WIN", "WDO", 0.7, 0.8)
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