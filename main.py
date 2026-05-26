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
from trade_manager import TradeManager
from trade_execution import TradeExecution
from observability import setup_logging, TelegramAlerter

async def main():
    
    setup_logging()
    logger = logging.getLogger(__name__)
    alerter = TelegramAlerter()

    # --- composition root: create the broker connector once and inject everywhere ---
    broker = MT5Connector()
    pair_trading_strategy = PairTradingStrategy(broker)
    trade_execution = TradeExecution(MAGIC_NUMBER, broker)
    trade_manager = TradeManager(broker)
    
    try:
        if not broker.initialize():
            logger.error("Broker initialization failed")
            alerter.send_error("Broker initialization failed — bot not started.")
            return
        
        logger.info(f"Broker initialized successfully, trading will start at {START_TIME_HOUR}:{START_TIME_MINUTE} UTC for {TRADE_WINDOW_TIME_HOURS} hours and {TRADE_WINDOW_TIME_MINUTES} minutes")
        while True:
            if not check_trading_time():
                positions = broker.get_open_positions()
                if positions: 
                    broker.close_all_positions()
                    logger.info("Outside trading hours. Closed all positions.")
                    alerter.send_stop_triggered(
                        reason="outside trading hours",
                        pnl=broker.get_profit(),
                    )
                else:
                    await asyncio.sleep(5)
                    continue
            while check_trading_time():
                positions = broker.get_open_positions()
                if positions:
                    logger.info(f"Currently open positions: {len(positions)}")
                    logger.info("Managing trades...")
                    trade_manager.manage_trades()
                else:
                    logger.info("No open positions currently.")
                    logger.info("Start scanning for trading opportunities...")
                    correlation, hedge_ratio, spreads, rolling_z_scores, pair, arbitrage_found = pair_trading_strategy.scan_pairs_arbitrage()                
                    if arbitrage_found:
                        logger.info(f"Arbitrage was found : {arbitrage_found} for pair y as {pair[0]} and x as {pair[1]} hedge ratio is {hedge_ratio} and spreads is {spreads} and rolling z scores is {rolling_z_scores}")
                        logger.info("Arbitrage opportunity detected. Executing trades...")
                        logger.info(f"Starting trade execution for {pair[0]} and {pair[1]} and hedge ratio {hedge_ratio} and z score {rolling_z_scores} and spreads {spreads}")
                        alerter.send_trade_open(pair[0], pair[1], z_score=rolling_z_scores, hedge_ratio=hedge_ratio)
                        trade_execution.execute_trade(pair[0], pair[1], correlation, hedge_ratio, rolling_z_scores)
                
            await asyncio.sleep(15)
    except KeyboardInterrupt:
        logger.info("Terminating script by user.")
    except Exception as exc:
        logger.exception("Unhandled exception in main loop")
        alerter.send_error(str(exc))
        raise
    finally:
        broker.shutdown()
    logger.info("Outside trading hours. Sleeping for 5 seconds.")
    
    await asyncio.sleep(15)

asyncio.run(main())