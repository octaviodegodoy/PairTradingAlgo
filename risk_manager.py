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

    def calc_sl_price(self, order_type: int, entry_price: float, volume: float,
                      risk_amount: float, tick_value: float, tick_size: float,
                      digits: int, order_type_buy: int = 0) -> float:
        """
        Convert a monetary risk budget into a hard stop-loss price.

            sl_distance = risk_amount / (volume * tick_value / tick_size)

        For BUY  : sl = entry_price - sl_distance
        For SELL : sl = entry_price + sl_distance

        Parameters
        ----------
        order_type      : broker's BUY/SELL constant (compared against order_type_buy)
        entry_price     : fill price of the order
        volume          : lot size
        risk_amount     : max monetary loss for this leg
        tick_value      : value per tick per lot (from broker symbol info)
        tick_size       : minimum price increment (from broker symbol info)
        digits          : decimal places for price rounding
        order_type_buy  : the broker's BUY constant (default 0, pass
                          connector.ORDER_TYPE_BUY for non-MT5 brokers)

        Returns 0.0 when risk_amount is 0 or inputs are invalid.
        """
        if risk_amount <= 0 or volume <= 0 or tick_value == 0 or tick_size == 0:
            return 0.0
        loss_per_point = volume * tick_value / tick_size
        sl_distance = risk_amount / loss_per_point
        if order_type == order_type_buy:
            return round(entry_price - sl_distance, digits)
        return round(entry_price + sl_distance, digits)
