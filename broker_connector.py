"""
broker_connector.py
-------------------
Abstract interface that every broker connector must implement.
Business logic (strategy, execution, risk) depends only on this class,
never on MetaTrader5 or any other vendor API directly.

To support a new broker, subclass BrokerConnector and implement all
@abstractmethod methods.  No other file needs to change.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BrokerConnector(ABC):
    """
    Broker-neutral interface.

    Class attributes below follow the 0 = BUY / 1 = SELL convention used by
    most APIs (MT5, FIX, IB TWS ...).  Subclasses must override them with the
    actual integer values from their vendor library so that code comparing
    ``position.type == self.broker.ORDER_TYPE_BUY`` remains correct.
    """

    # --- order / position direction constants (override in each subclass) ---
    ORDER_TYPE_BUY: int = 0
    ORDER_TYPE_SELL: int = 1
    POSITION_TYPE_BUY: int = 0
    POSITION_TYPE_SELL: int = 1
    # Timeframe: subclasses set this to the vendor's "daily bar" constant.
    TIMEFRAME_D1: int = 1440

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """Connect / authenticate with the broker.  Returns True on success."""

    @abstractmethod
    def shutdown(self) -> None:
        """Disconnect cleanly."""

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Pause execution (allows subclasses to use async-aware sleep)."""

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abstractmethod
    def get_data_futures_btg(self, symbol: str, n_bars: int = None) -> pd.DataFrame:
        """
        Return a DataFrame of OHLCV bars for *symbol*, stitched across
        all front-month contracts that make up the continuous series.
        Columns must include at least: ``time``, ``open``, ``high``,
        ``low``, ``close``, ``tick_volume``, ``symbol``.
        """

    @abstractmethod
    def get_symbol_futures(self, group_name: str) -> tuple:
        """Return ``(expiry_timestamp, symbol_name)`` for the front-month contract."""

    @abstractmethod
    def get_symbol_info(self, symbol: str):
        """Return an object/namedtuple with at least: ``volume_min``, ``trade_tick_size``,
        ``trade_tick_value``, ``digits``, ``ask``, ``bid``."""

    @abstractmethod
    def get_symbol_tick(self, symbol: str):
        """Return the latest tick with at least ``ask`` and ``bid`` attributes."""

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @abstractmethod
    def get_account_info(self):
        """Return an object/namedtuple with at least: ``equity``, ``profit``."""

    @abstractmethod
    def get_profit(self) -> float:
        """Return the current floating P&L for all open positions."""

    @abstractmethod
    def get_order_calc_margin(self, order_type: int, symbol: str,
                               volume: float, price: float) -> float:
        """Return the margin required for a hypothetical order."""

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @abstractmethod
    def get_open_positions(self) -> list:
        """Return all open positions belonging to this algo (magic filter applied)."""

    @abstractmethod
    def get_total_positions(self) -> int:
        """Return the count of open positions + pending orders for this algo."""

    @abstractmethod
    def check_positions_type(self, symbol: str, position_type: int) -> bool:
        """Return True if any open position on *symbol* matches *position_type*."""

    @abstractmethod
    def total_daily_risk(self) -> tuple:
        """
        Return ``(highest_zscore_period, total_profit, total_volume, grid_deals_count)``
        from today's trading history.
        """

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abstractmethod
    def place_order(self, symbolY: str, symbolX: str,
                    volumeY: float, volumeX: float,
                    orders_type: list, zscore: float,
                    sl_y: float = 0.0, sl_x: float = 0.0) -> bool:
        """
        Submit a paired entry order for both legs with retry logic.
        *orders_type* is ``[leg_Y_direction, leg_X_direction]`` using this
        class's ORDER_TYPE_* constants.
        Returns True when both legs were filled successfully.
        """

    @abstractmethod
    def close_all_positions(self) -> None:
        """Market-close every open position belonging to this algo."""

    @abstractmethod
    def modify_position_sl(self, ticket: int, symbol: str,
                            sl_price: float, tp_price: float = 0.0) -> bool:
        """Modify the stop-loss (and optionally take-profit) of an open position."""

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    @abstractmethod
    def get_options_puts(self, underlying_symbol: str) -> list:
        """
        Return a list of put-option symbol-info objects for *underlying_symbol*.

        Each element must expose at least the following attributes:

        * ``name``             – ticker string of the option contract
        * ``option_strike``    – strike price (float)
        * ``expiration_time``  – expiry as a UNIX timestamp (int)
        * ``volume_real``      – traded volume used for ranking (float)

        Returns an empty list when no options are found.
        """
