import MetaTrader5 as mt5
import logging
from constants import MARGIN_PERCENT, MAX_RISK, MARGIN_Y, MARGIN_X


class RiskManager:
    """
    Centralised risk-sizing logic.

    Responsibilities:
      - Session-level monetary loss budget (max_loss)
      - Maximum lot allocation per session (max_lots)
      - Converting a monetary risk budget into a hard stop-loss price (calc_sl_price)

    No position management or order-sending lives here — only pure math and
    read-only MT5 symbol queries needed for the SL calculation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def max_loss(self, equity: float) -> float:
        """Session-level maximum tolerated loss in account currency."""
        return equity * MARGIN_PERCENT * MAX_RISK

    def max_lots(self, equity: float) -> float:
        """Maximum combined lot allocation given available equity."""
        total_margin = equity * MARGIN_PERCENT
        return total_margin / MARGIN_Y + total_margin / MARGIN_X

    def calc_sl_price(self, symbol: str, order_type: int, entry_price: float,
                      volume: float, risk_amount: float) -> float:
        """
        Convert a monetary risk budget into a hard stop-loss price.

            sl_distance = risk_amount / (volume * tick_value / tick_size)

        For BUY  : sl = entry_price - sl_distance
        For SELL : sl = entry_price + sl_distance

        Returns 0.0 when risk_amount is 0 or symbol info is unavailable.
        """
        if risk_amount <= 0 or volume <= 0:
            return 0.0
        info = mt5.symbol_info(symbol)
        if info is None or info.trade_tick_value == 0 or info.trade_tick_size == 0:
            self.logger.warning(f"Cannot compute SL for {symbol}: missing tick info")
            return 0.0
        loss_per_point = volume * info.trade_tick_value / info.trade_tick_size
        sl_distance = risk_amount / loss_per_point
        if order_type == mt5.ORDER_TYPE_BUY:
            return round(entry_price - sl_distance, info.digits)
        return round(entry_price + sl_distance, info.digits)
