import logging
import math
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
                      digits: int, order_type_buy: int = 0,
                      stops_level: float = 0) -> float:
        """
        Convert a monetary risk budget into a hard stop-loss price.

            sl_distance = risk_amount / (volume * tick_value / tick_size)

        For BUY  : sl = entry_price - sl_distance
        For SELL : sl = entry_price + sl_distance

        The result is snapped to the nearest valid tick boundary and respects
        the broker's minimum stop distance (stops_level).

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
        stops_level     : broker minimum stop distance in points (trade_stops_level);
                          the SL will be at least (stops_level + 1) * tick_size away.

        Returns 0.0 when risk_amount is 0 or inputs are invalid.
        """
        if risk_amount <= 0 or volume <= 0 or tick_value == 0 or tick_size == 0:
            return 0.0
        loss_per_point = volume * tick_value / tick_size
        sl_distance = risk_amount / loss_per_point

        # Enforce broker minimum stop distance (stops_level is in points/ticks)
        if stops_level > 0:
            min_distance = (stops_level + 1) * tick_size
            sl_distance = max(sl_distance, min_distance)

        if order_type == order_type_buy:
            raw_sl = entry_price - sl_distance
            # Snap DOWN to nearest valid tick (BUY SL must be below entry)
            sl = math.floor(round(raw_sl / tick_size, 8)) * tick_size
            return round(sl, digits)
        else:
            raw_sl = entry_price + sl_distance
            # Snap UP to nearest valid tick (SELL SL must be above entry)
            sl = math.ceil(round(raw_sl / tick_size, 8)) * tick_size
            return round(sl, digits)
