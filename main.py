import logging
from config import (
    PERIODS, ROLLING_PERIODS, MARGIN_Y, MARGIN_X, TRAILING_DISTANCE_POINTS,
    PROFIT_THRESHOLD_MULTIPLIER, MARGIN_PERCENT, MAX_POSITIONS, MAX_RISK,
    MIN_ZSCORE, MAX_HALF_LIFE, MAGIC_NUMBER
)

from mt5_connector import MT5Connector
from utils import check_trading_time
from trade_manager import TradeManager

def main():
     # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize MT5 connection and managers
    mt5_conn = MT5Connector()
    trade_manager = TradeManager(MAGIC_NUMBER)
    
    try:
        while True:
            if not check_trading_time():
                if mt5_conn.get_open_positions_count() > 0:
                    logger.info("Trading window closed, closing all positions.")
                    
            else:
                # Check risk and profit
                profit, highest_z, total_profit = total_daily_risk(mt5_conn)
                if abs(profit) > MAX_RISK:
                    logger.info("Max risk exceeded, closing all positions.")
                    trade_manager.close_all_positions()
                    break

                # Active trading logic
                strategy.run_trading_cycle()
            mt5_conn.sleep(15)
    except KeyboardInterrupt:
        logger.info("Terminating script by user.")
    finally:
        mt5_conn.shutdown()

if __name__ == "__main__":
    main()