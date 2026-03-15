"""
Main trading bot orchestrator.

Coordinates all components: data fetching, strategy execution,
risk management, order execution, and notifications.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytz
import yaml

from .data.market_data import MarketDataProvider
from .database.models import (
    DailyPerformance,
    DailyState,
    OrderSide,
    OrderStatus,
    Position as DBPosition,
    Signal as DBSignal,
    SignalType as DBSignalType,
    Trade,
    TrackedOrder,
    init_db,
    get_db_session,
)
from .execution.broker import AlpacaBroker, Broker, Order, OrderSide as BrokerOrderSide, OrderType
from .execution.sim_broker import SimulatedBroker, SimulatedBrokerConfig
from .notifications.alerts import NotificationManager
from .risk.risk_manager import AccountInfo, PositionRisk, RiskManager
from .strategies.base import PositionInfo, SignalType
from .strategies.factory import get_strategy_from_settings
from .config.validator import ConfigValidationError
from .utils.audit import AuditLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def validate_config(settings: dict) -> list[str]:
    """
    Validate configuration at startup. Returns list of errors.

    Checks:
    - Required sections exist
    - API keys are present (or env vars set)
    - Numeric values are in valid ranges
    - Symbol list is non-empty
    - Risk parameters are sane
    """
    errors = []

    # Trading section
    trading = settings.get("trading", {})
    if not trading:
        errors.append("Missing 'trading' section in config")
    else:
        symbols = trading.get("symbols", [])
        if not symbols:
            errors.append("No symbols configured in trading.symbols")

        interval = trading.get("check_interval_seconds", 60)
        if interval < 10:
            errors.append(f"check_interval_seconds={interval} is too low (min 10)")

        tz = trading.get("timezone", "America/New_York")
        try:
            pytz.timezone(tz)
        except pytz.exceptions.UnknownTimeZoneError:
            errors.append(f"Unknown timezone: {tz}")

    # API section - only required for non-simulated brokers
    broker_type = settings.get("broker", {}).get("type", "alpaca").lower()
    if broker_type != "simulated":
        api = settings.get("api", {})
        alpaca = api.get("alpaca", {})
        api_key = alpaca.get("key") or os.getenv("ALPACA_API_KEY", "")
        secret_key = alpaca.get("secret") or os.getenv("ALPACA_SECRET_KEY", "")
        if broker_type == "alpaca" and (not api_key or not secret_key):
            errors.append(
                "Alpaca API credentials not configured. "
                "Set api.alpaca.key/secret in config or ALPACA_API_KEY/ALPACA_SECRET_KEY env vars. "
                "Or use broker.type: simulated for offline testing."
            )

    # Risk section
    risk = settings.get("risk", {})
    if risk:
        max_pos = risk.get("max_position_pct", 0.10)
        if not (0 < max_pos <= 1.0):
            errors.append(f"max_position_pct={max_pos} must be between 0 and 1.0")

        max_loss = risk.get("max_daily_loss_pct", 0.02)
        if not (0 < max_loss <= 1.0):
            errors.append(f"max_daily_loss_pct={max_loss} must be between 0 and 1.0")

        stop_loss = risk.get("stop_loss_pct", 0.05)
        if not (0 < stop_loss <= 0.5):
            errors.append(f"stop_loss_pct={stop_loss} must be between 0 and 0.5")

        trailing = risk.get("trailing_stop_pct", 0.03)
        if not (0 < trailing <= 0.5):
            errors.append(f"trailing_stop_pct={trailing} must be between 0 and 0.5")

        max_positions = risk.get("max_open_positions", 10)
        if max_positions < 1:
            errors.append(f"max_open_positions={max_positions} must be >= 1")

    # Database section
    db = settings.get("database", {})
    db_url = db.get("url", "sqlite:///data/trading_bot.db")
    if not db_url:
        errors.append("Database URL is empty")

    return errors


class TradingBot:
    """
    Main trading bot class.

    Orchestrates the trading loop:
    1. Reconcile positions with broker
    2. Fetch market data
    3. Update positions
    4. Check for exit signals
    5. Generate entry signals
    6. Run risk checks
    7. Execute orders
    8. Track order lifecycle
    9. Send notifications
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.settings = self._load_settings()

        # Validate config at startup (4.5)
        errors = validate_config(self.settings)
        if errors:
            for err in errors:
                logger.error(f"Config error: {err}")
            raise ConfigValidationError(
                f"Configuration validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        # Core settings
        self.paper_mode = self.settings.get("trading", {}).get("paper_mode", True)
        self.symbols = self.settings.get("trading", {}).get("symbols", [])
        self.check_interval = self.settings.get("trading", {}).get("check_interval_seconds", 60)
        self.timezone = pytz.timezone(
            self.settings.get("trading", {}).get("timezone", "America/New_York")
        )

        # Safety check
        if not self.paper_mode:
            logger.warning("LIVE TRADING MODE - Real money at risk!")
            self._confirm_live_trading()

        # Initialize components
        self._init_components()

        # State
        self._running = False
        self._shutting_down = False
        self._last_daily_summary: Optional[datetime] = None
        self._market_open_recorded: Optional[datetime] = None
        self._last_reconciliation: Optional[datetime] = None
        self._reconciliation_interval = timedelta(minutes=5)
        self._last_heartbeat: Optional[datetime] = None
        self._pending_orders: set[str] = set()  # Symbols with in-flight orders (Fix #2)
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 10  # Halt after this many consecutive failures

        # Register signal handlers (1.5)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _load_settings(self) -> dict:
        """Load settings from YAML file."""
        config_file = Path(self.config_path)

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return {}

        with open(config_file) as f:
            settings = yaml.safe_load(f)

        logger.info(f"Loaded settings from {config_file}")
        return settings or {}

    def _confirm_live_trading(self) -> None:
        """Require confirmation for live trading mode."""
        confirm = os.getenv("CONFIRM_LIVE_TRADING", "").lower()
        if confirm != "yes":
            logger.error(
                "Live trading requires CONFIRM_LIVE_TRADING=yes environment variable"
            )
            sys.exit(1)

    def _init_components(self) -> None:
        """Initialize all bot components."""
        logger.info("Initializing bot components...")

        # Database
        db_url = self.settings.get("database", {}).get("url", "sqlite:///data/trading_bot.db")
        init_db(db_url)
        logger.info("Database initialized")

        # Market data provider
        self.data_provider = MarketDataProvider.from_settings(self.settings)
        logger.info("Market data provider initialized")

        # Strategy (now uses factory for any strategy)
        self.strategy = get_strategy_from_settings(self.settings)
        logger.info(f"Strategy initialized: {self.strategy.name}")

        # Risk manager
        self.risk_manager = RiskManager.from_settings(self.settings)
        logger.info("Risk manager initialized")

        # Broker (Fix #16: select broker via config)
        self.broker = self._create_broker()
        logger.info(f"Broker initialized: {type(self.broker).__name__} (paper={self.paper_mode})")

        # Notifications
        self.notifications = NotificationManager.from_settings(self.settings)
        logger.info("Notification manager initialized")

        # Audit logger (4.3)
        audit_dir = self.settings.get("logging", {}).get("audit_dir", "logs")
        self.audit = AuditLogger(log_dir=audit_dir)
        logger.info("Audit logger initialized")

        # Set up file logging
        log_settings = self.settings.get("logging", {})
        log_file = log_settings.get("file", "logs/trading_bot.log")
        log_level = getattr(logging, log_settings.get("level", "INFO").upper())

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    def _create_broker(self) -> Broker:
        """Create broker instance based on config (Fix #16).

        Supports three broker types via ``broker.type`` config key:
        - ``"alpaca"`` (default): Alpaca Markets API (paper or live)
        - ``"simulated"``: Fully in-memory simulated broker, no API needed
        - ``"ibkr"``: Interactive Brokers via ib_insync (experimental)

        Example config::

            broker:
              type: simulated        # or "alpaca" or "ibkr"
              initial_capital: 100000
        """
        broker_settings = self.settings.get("broker", {})
        broker_type = broker_settings.get("type", "alpaca").lower()

        if broker_type == "simulated":
            initial_capital = broker_settings.get("initial_capital", 100000)
            commission = broker_settings.get("commission_per_share", 0.0)
            slippage = broker_settings.get("slippage_pct", 0.001)
            config = SimulatedBrokerConfig(
                initial_capital=initial_capital,
                commission_per_share=commission,
                slippage_pct=slippage,
            )
            logger.info(
                f"Using SimulatedBroker (capital=${initial_capital:,.2f}, "
                f"slippage={slippage*100:.2f}%)"
            )
            return SimulatedBroker(config)

        elif broker_type == "ibkr":
            from .execution.ibkr_broker import IBKRBroker
            logger.info("Using IBKRBroker (experimental)")
            return IBKRBroker.from_settings(self.settings)

        else:
            # Default: Alpaca
            return AlpacaBroker.from_settings(self.settings)

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully (1.5)."""
        if self._shutting_down:
            logger.warning("Forced shutdown")
            sys.exit(1)

        logger.info(f"Received shutdown signal ({signum}), initiating graceful shutdown...")
        self._shutting_down = True
        self._running = False

    def _graceful_shutdown(self) -> None:
        """Perform graceful shutdown: cancel orders, flush DB, notify (1.5)."""
        logger.info("Performing graceful shutdown...")

        # 1. Cancel all open orders
        try:
            open_orders = self.broker.get_open_orders()
            if open_orders:
                logger.info(f"Cancelling {len(open_orders)} open orders...")
                for order in open_orders:
                    try:
                        self.broker.cancel_order(order["id"])
                        logger.info(f"Cancelled order {order['id']} ({order['symbol']})")
                        self.audit.log_order_cancelled(
                            order_id=order["id"],
                            symbol=order["symbol"],
                            reason="graceful_shutdown",
                        )
                    except Exception as e:
                        logger.error(f"Failed to cancel order {order['id']}: {e}")
        except Exception as e:
            logger.error(f"Failed to get/cancel open orders: {e}")

        # 2. Update all tracked orders to final state
        self._update_tracked_orders()

        # 3. Final position sync
        try:
            self._reconcile_positions()
        except Exception as e:
            logger.error(f"Failed to reconcile positions during shutdown: {e}")

        # 4. Send shutdown notification
        try:
            self.notifications.notify_bot_stopped("Graceful shutdown completed")
        except Exception as e:
            logger.error(f"Failed to send shutdown notification: {e}")

        self.audit.log_event("bot_shutdown", {"reason": "graceful_shutdown"})
        logger.info("Graceful shutdown complete")

    def _reconcile_positions(self) -> None:
        """Reconcile local DB positions with broker state (1.2)."""
        logger.info("Reconciling positions with broker...")

        try:
            broker_positions = {p.symbol: p for p in self.broker.get_positions()}
        except Exception as e:
            logger.error(f"Failed to fetch broker positions for reconciliation: {e}")
            return

        with get_db_session() as session:
            db_positions = {p.symbol: p for p in session.query(DBPosition).all()}

            # Check for positions in broker but not in DB
            for symbol, broker_pos in broker_positions.items():
                if symbol not in db_positions:
                    logger.warning(
                        f"RECONCILIATION: Position {symbol} exists in broker "
                        f"but not in DB. Adding to DB."
                    )
                    new_pos = DBPosition(
                        symbol=symbol,
                        quantity=broker_pos.quantity,
                        avg_entry_price=broker_pos.avg_entry_price,
                        current_price=broker_pos.current_price,
                        market_value=broker_pos.market_value,
                        cost_basis=broker_pos.cost_basis,
                        unrealized_pnl=broker_pos.unrealized_pnl,
                        unrealized_pnl_pct=broker_pos.unrealized_pnl_pct,
                        highest_price=broker_pos.current_price,
                        strategy_name="unknown_reconciled",
                    )
                    session.add(new_pos)
                    self.audit.log_reconciliation(
                        symbol=symbol,
                        action="added_to_db",
                        broker_qty=broker_pos.quantity,
                        db_qty=0,
                    )
                else:
                    # Check for quantity mismatch
                    db_pos = db_positions[symbol]
                    if abs(db_pos.quantity - broker_pos.quantity) > 0.001:
                        logger.warning(
                            f"RECONCILIATION: Position {symbol} quantity mismatch: "
                            f"DB={db_pos.quantity}, Broker={broker_pos.quantity}. "
                            f"Updating DB to match broker."
                        )
                        self.audit.log_reconciliation(
                            symbol=symbol,
                            action="quantity_updated",
                            broker_qty=broker_pos.quantity,
                            db_qty=db_pos.quantity,
                        )
                        db_pos.quantity = broker_pos.quantity
                        db_pos.avg_entry_price = broker_pos.avg_entry_price
                        db_pos.current_price = broker_pos.current_price
                        db_pos.market_value = broker_pos.market_value
                        db_pos.cost_basis = broker_pos.cost_basis
                        db_pos.unrealized_pnl = broker_pos.unrealized_pnl
                        db_pos.unrealized_pnl_pct = broker_pos.unrealized_pnl_pct

            # Check for positions in DB but not in broker
            for symbol, db_pos in db_positions.items():
                if symbol not in broker_positions:
                    logger.warning(
                        f"RECONCILIATION: Position {symbol} exists in DB "
                        f"but not in broker. Removing from DB."
                    )
                    self.audit.log_reconciliation(
                        symbol=symbol,
                        action="removed_from_db",
                        broker_qty=0,
                        db_qty=db_pos.quantity,
                    )
                    session.delete(db_pos)

        self._last_reconciliation = datetime.now(self.timezone)
        logger.info("Position reconciliation complete")

    def _should_reconcile(self) -> bool:
        """Check if it's time for periodic reconciliation."""
        if self._last_reconciliation is None:
            return True
        return datetime.now(self.timezone) - self._last_reconciliation > self._reconciliation_interval

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        trading_settings = self.settings.get("trading", {})
        trading_hours = trading_settings.get("trading_hours", {})

        start_str = trading_hours.get("start", "09:30")
        end_str = trading_hours.get("end", "16:00")

        now = datetime.now(self.timezone)
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()

        return start_time <= now.time() < end_time  # Strict < at end (Fix #9)

    def _store_market_open_equity(self, equity: float, cash: float, positions_value: float) -> None:
        """Store market open equity for today."""
        with get_db_session() as session:
            today = datetime.now(self.timezone).date()
            today_start = datetime.combine(today, datetime.min.time())

            existing = session.query(DailyState).filter(
                DailyState.date >= today_start
            ).first()

            if not existing:
                daily_state = DailyState(
                    date=datetime.now(self.timezone),
                    market_open_equity=equity,
                    market_open_cash=cash,
                    market_open_positions_value=positions_value,
                )
                session.add(daily_state)
                logger.info(f"Stored market open equity: ${equity:,.2f}")

    def _get_market_open_equity(self) -> Optional[float]:
        """Get market open equity for today."""
        with get_db_session() as session:
            today = datetime.now(self.timezone).date()
            today_start = datetime.combine(today, datetime.min.time())

            daily_state = session.query(DailyState).filter(
                DailyState.date >= today_start
            ).first()

            return daily_state.market_open_equity if daily_state else None

    def _get_account_info(self) -> AccountInfo:
        """Get current account information."""
        account = self.broker.get_account()

        # Get starting equity for the day from database
        starting_equity = self._get_market_open_equity()
        if starting_equity is None:
            starting_equity = account.equity

        # Daily P&L = current equity - starting equity (Fix #4)
        # This captures BOTH realized and unrealized losses for the circuit breaker
        daily_pnl = account.equity - starting_equity

        return AccountInfo(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            positions_value=account.positions_value,
            day_trade_count=account.day_trade_count,
            daily_pnl=daily_pnl,
            starting_equity=starting_equity,
        )

    def _get_position_info(self) -> dict[str, PositionInfo]:
        """Get current positions as PositionInfo objects."""
        positions = self.broker.get_positions()
        result = {}

        with get_db_session() as session:
            for pos in positions:
                # Get database position for additional info
                db_pos = session.query(DBPosition).filter_by(symbol=pos.symbol).first()

                result[pos.symbol] = PositionInfo(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    avg_entry_price=pos.avg_entry_price,
                    current_price=pos.current_price,
                    unrealized_pnl=pos.unrealized_pnl,
                    unrealized_pnl_pct=pos.unrealized_pnl_pct,
                    highest_price=db_pos.highest_price if db_pos else pos.current_price,
                    stop_loss_price=db_pos.stop_loss_price if db_pos else None,
                    trailing_stop_price=db_pos.trailing_stop_price if db_pos else None,
                    take_profit_price=db_pos.take_profit_price if db_pos else None,
                    opened_at=db_pos.opened_at if db_pos else None,
                )

        return result

    def _get_position_risks(self, account: AccountInfo) -> list[PositionRisk]:
        """Convert positions to PositionRisk objects."""
        positions = self.broker.get_positions()
        return [
            PositionRisk(
                symbol=pos.symbol,
                quantity=pos.quantity,
                market_value=pos.market_value,
                cost_basis=pos.cost_basis,
                unrealized_pnl=pos.unrealized_pnl,
                weight_pct=(pos.market_value / account.equity * 100) if account.equity > 0 else 0,
            )
            for pos in positions
        ]

    def _update_position_in_db(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> None:
        """Update or create position in database."""
        with get_db_session() as session:
            pos = session.query(DBPosition).filter_by(symbol=symbol).first()

            if pos:
                pos.quantity = quantity
                pos.current_price = current_price
                pos.market_value = quantity * current_price
                pos.unrealized_pnl = (current_price - pos.avg_entry_price) * quantity
                pos.unrealized_pnl_pct = (
                    (current_price - pos.avg_entry_price) / pos.avg_entry_price * 100
                    if pos.avg_entry_price > 0 else 0
                )
                if current_price > pos.highest_price:
                    pos.highest_price = current_price
                if stop_loss:
                    pos.stop_loss_price = stop_loss
                if trailing_stop:
                    pos.trailing_stop_price = trailing_stop
                if take_profit:
                    pos.take_profit_price = take_profit
            else:
                pos = DBPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=avg_price,
                    current_price=current_price,
                    market_value=quantity * current_price,
                    cost_basis=quantity * avg_price,
                    unrealized_pnl=0,
                    unrealized_pnl_pct=0,
                    highest_price=current_price,
                    stop_loss_price=stop_loss,
                    trailing_stop_price=trailing_stop,
                    take_profit_price=take_profit,
                    strategy_name=self.strategy.name,
                )
                session.add(pos)

    def _remove_position_from_db(self, symbol: str) -> None:
        """Remove position from database."""
        with get_db_session() as session:
            session.query(DBPosition).filter_by(symbol=symbol).delete()

    def _record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: str,
        reason: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        realized_pnl_pct: Optional[float] = None,
    ) -> None:
        """Record trade in database."""
        with get_db_session() as session:
            trade = Trade(
                symbol=symbol,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=price,
                total_value=quantity * price,
                order_id=order_id,
                status=OrderStatus.FILLED,
                strategy_name=self.strategy.name,
                signal_reason=reason,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                realized_pnl=realized_pnl,
                realized_pnl_pct=realized_pnl_pct,
                is_paper_trade=self.paper_mode,
                executed_at=datetime.utcnow(),
            )
            session.add(trade)
            logger.info(f"Recorded trade: {trade}")

        # Audit log
        self.audit.log_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_id=order_id,
            reason=reason,
            realized_pnl=realized_pnl,
        )

    def _record_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float,
        price: float,
        reason: str,
        indicators: dict,
        was_executed: bool = False,
        trade_id: Optional[int] = None,
    ) -> None:
        """Record signal in database."""
        with get_db_session() as session:
            db_signal_type = {
                SignalType.BUY: DBSignalType.BUY,
                SignalType.SELL: DBSignalType.SELL,
                SignalType.HOLD: DBSignalType.HOLD,
            }.get(signal_type, DBSignalType.HOLD)

            signal_record = DBSignal(
                symbol=symbol,
                signal_type=db_signal_type,
                strength=strength,
                price_at_signal=price,
                rsi=indicators.get("rsi"),
                macd=indicators.get("macd"),
                macd_signal=indicators.get("macd_signal"),
                bollinger_position=indicators.get("bb_percent"),
                strategy_name=self.strategy.name,
                reason=reason,
                was_executed=was_executed,
                trade_id=trade_id,
            )
            session.add(signal_record)

        # Audit log
        self.audit.log_signal(
            symbol=symbol,
            signal_type=signal_type.value,
            strength=strength,
            price=price,
            reason=reason,
            was_executed=was_executed,
        )

    def _track_order(self, order_id: str, symbol: str, side: str, quantity: float,
                     order_type: str, price: Optional[float] = None) -> None:
        """Track order in database for lifecycle management (1.4)."""
        with get_db_session() as session:
            tracked = TrackedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                submitted_price=price,
                status="submitted",
                strategy_name=self.strategy.name,
                submitted_at=datetime.utcnow(),
            )
            session.add(tracked)

    def _update_tracked_orders(self) -> None:
        """Poll broker for order status updates (1.4)."""
        with get_db_session() as session:
            pending_orders = session.query(TrackedOrder).filter(
                TrackedOrder.status.in_(["submitted", "accepted", "partially_filled", "new", "pending"])
            ).all()

            for tracked in pending_orders:
                try:
                    broker_order = self.broker.get_order(tracked.order_id)
                    if broker_order is None:
                        continue

                    new_status = broker_order.get("status", tracked.status)
                    if new_status != tracked.status:
                        old_status = tracked.status
                        tracked.status = new_status
                        tracked.updated_at = datetime.utcnow()

                        filled_qty = broker_order.get("filled_qty", 0)
                        if filled_qty > 0:
                            tracked.filled_quantity = filled_qty
                            tracked.filled_price = broker_order.get("filled_avg_price")

                        if new_status == "filled":
                            tracked.filled_at = datetime.utcnow()

                        self.audit.log_order_status_change(
                            order_id=tracked.order_id,
                            symbol=tracked.symbol,
                            old_status=old_status,
                            new_status=new_status,
                            filled_qty=filled_qty,
                        )

                        logger.info(
                            f"Order {tracked.order_id} ({tracked.symbol}): "
                            f"{old_status} -> {new_status}"
                        )

                        # Handle partial fills
                        if new_status == "partially_filled" and filled_qty > 0:
                            logger.warning(
                                f"Partial fill for {tracked.symbol}: "
                                f"{filled_qty}/{tracked.quantity} shares"
                            )

                except Exception as e:
                    logger.error(f"Failed to update order {tracked.order_id}: {e}")

    def _check_exit_signals(
        self,
        market_data: dict,
        positions: dict[str, PositionInfo],
    ) -> None:
        """Check and execute exit signals for open positions."""
        for symbol, position in positions.items():
            if symbol not in market_data:
                logger.warning(f"No market data for position {symbol}")
                continue

            data = market_data[symbol]

            # Update position with current price
            position.current_price = data.last_price
            if data.last_price > position.highest_price:
                position.highest_price = data.last_price

            # Update trailing stop if applicable
            if hasattr(self.strategy, 'update_trailing_stop'):
                new_trailing_stop = self.strategy.update_trailing_stop(position, data.last_price)
                if new_trailing_stop:
                    position.trailing_stop_price = new_trailing_stop
                    self._update_position_in_db(
                        symbol,
                        position.quantity,
                        position.avg_entry_price,
                        position.current_price,
                        trailing_stop=new_trailing_stop,
                    )

            # Check for exit signal
            exit_signal = self.strategy.should_exit(position, data)

            if exit_signal:
                logger.info(f"Exit signal for {symbol}: {exit_signal}")
                self._execute_exit(position, exit_signal)

    def _execute_exit(self, position: PositionInfo, exit_signal) -> None:
        """Execute exit order for a position."""
        symbol = position.symbol

        # Guard against duplicate exit if already pending (Fix #2)
        if symbol in self._pending_orders:
            logger.info(f"Skipping exit for {symbol}: order already in-flight")
            return

        # Use broker's close_position to handle fractional shares correctly (Fix #3)
        # This ensures the entire position is closed, not just int(quantity)
        sell_qty = position.quantity

        # Create sell order
        order = Order(
            symbol=symbol,
            side=BrokerOrderSide.SELL,
            quantity=sell_qty,
            order_type=OrderType.MARKET,
        )

        # Log risk check result for audit
        self.audit.log_risk_check(
            symbol=symbol,
            action="exit",
            checks=[{"check": "exit_signal", "passed": True, "reason": exit_signal.description}],
        )

        # Submit order
        result = self.broker.submit_order(order)

        if result.success:
            self._pending_orders.add(symbol)

            # Track order lifecycle (1.4)
            self._track_order(
                order_id=result.order_id,
                symbol=symbol,
                side="sell",
                quantity=sell_qty,
                order_type="market",
                price=position.current_price,
            )

            # Use actual fill price if available, otherwise estimate (Fix #1)
            fill_price = result.filled_price or position.current_price
            pnl = (fill_price - position.avg_entry_price) * position.quantity
            pnl_pct = (
                (fill_price - position.avg_entry_price) / position.avg_entry_price * 100
                if position.avg_entry_price > 0 else 0
            )

            # Record trade
            self._record_trade(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                price=fill_price,
                order_id=result.order_id,
                reason=exit_signal.description,
                realized_pnl=pnl,
                realized_pnl_pct=pnl_pct,
            )

            # Remove from DB
            self._remove_position_from_db(symbol)
            self._pending_orders.discard(symbol)

            # Track day trade
            self.risk_manager.record_trade()

            # Send notification
            if exit_signal.reason.value in ("stop_loss", "trailing_stop"):
                self.notifications.notify_stop_loss(
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    exit_price=fill_price,
                    loss=pnl,
                    loss_pct=pnl_pct,
                )
            elif exit_signal.reason.value == "take_profit":
                self.notifications.notify_take_profit(
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    exit_price=fill_price,
                    profit=pnl,
                    profit_pct=pnl_pct,
                )
            else:
                self.notifications.notify_trade(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                    price=fill_price,
                    total=fill_price * position.quantity,
                    is_paper=self.paper_mode,
                )

            logger.info(f"Closed position {symbol}: PnL ${pnl:.2f} ({pnl_pct:.2f}%)")
        else:
            logger.error(f"Failed to close position {symbol}: {result.message}")
            self.notifications.notify_error("Order Failed", f"Failed to close {symbol}", result.message)

    def _process_entry_signals(
        self,
        market_data: dict,
        positions: dict[str, PositionInfo],
        account: AccountInfo,
    ) -> None:
        """Generate and execute entry signals."""
        # Generate signals
        signals = self.strategy.generate_signals(market_data, positions)

        for signal in signals:
            logger.info(f"Processing signal: {signal}")

            # Record signal
            self._record_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                strength=signal.strength,
                price=signal.price,
                reason=signal.reason,
                indicators=signal.indicators,
            )

            if signal.signal_type != SignalType.BUY:
                continue

            # Guard against duplicate entry if order already in-flight (Fix #2)
            if signal.symbol in self._pending_orders:
                logger.info(f"Skipping {signal.symbol}: order already in-flight")
                continue

            # Calculate position size
            shares = self.risk_manager.calculate_position_size(
                account=account,
                price=signal.price,
                signal_strength=signal.strength,
            )

            if shares <= 0:
                logger.info(f"Skipping {signal.symbol}: calculated 0 shares")
                continue

            trade_value = shares * signal.price

            # Risk checks
            position_risks = self._get_position_risks(account)
            can_trade, risk_results = self.risk_manager.can_trade(
                account=account,
                positions=position_risks,
                proposed_trade_value=trade_value,
                proposed_symbol=signal.symbol,
            )

            # Audit risk checks
            self.audit.log_risk_check(
                symbol=signal.symbol,
                action="entry",
                checks=[
                    {"check": r.check_type.value, "passed": r.passed, "message": r.message}
                    for r in risk_results
                ],
            )

            if not can_trade:
                failed_checks = [r for r in risk_results if not r.passed]
                logger.warning(
                    f"Trade blocked for {signal.symbol}: {[r.message for r in failed_checks]}"
                )
                continue

            # Execute trade
            self._execute_entry(signal, shares, account)

    def _execute_entry(self, signal, shares: int, account: AccountInfo) -> None:
        """Execute entry order for a signal."""
        symbol = signal.symbol
        price = signal.price

        # Mark as in-flight before submission (Fix #2)
        self._pending_orders.add(symbol)

        # Create buy order
        order = Order(
            symbol=symbol,
            side=BrokerOrderSide.BUY,
            quantity=shares,
            order_type=OrderType.MARKET,
        )

        # Submit order
        result = self.broker.submit_order(order)

        if result.success:
            # Track order lifecycle (1.4)
            self._track_order(
                order_id=result.order_id,
                symbol=symbol,
                side="buy",
                quantity=shares,
                order_type="market",
                price=price,
            )

            fill_price = result.filled_price or price
            stop_loss = self.risk_manager.calculate_stop_loss_price(fill_price)
            take_profit = signal.suggested_take_profit or (fill_price * 1.02)

            # Record trade
            self._record_trade(
                symbol=symbol,
                side="buy",
                quantity=shares,
                price=fill_price,
                order_id=result.order_id,
                reason=signal.reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            # Update position in DB
            self._update_position_in_db(
                symbol=symbol,
                quantity=shares,
                avg_price=fill_price,
                current_price=fill_price,
                stop_loss=stop_loss,
                trailing_stop=fill_price * (1 - self.risk_manager.config.trailing_stop_pct),
                take_profit=take_profit,
            )

            # Track day trade
            self.risk_manager.record_trade()

            # Send notification
            self.notifications.notify_trade(
                symbol=symbol,
                side="buy",
                quantity=shares,
                price=fill_price,
                total=fill_price * shares,
                is_paper=self.paper_mode,
            )

            self._pending_orders.discard(symbol)

            logger.info(
                f"Opened position {symbol}: {shares} shares @ ${fill_price:.2f} "
                f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
            )
        else:
            self._pending_orders.discard(symbol)
            logger.error(f"Failed to open position {symbol}: {result.message}")
            self.notifications.notify_error("Order Failed", f"Failed to buy {symbol}", result.message)

    def _send_daily_summary(self) -> None:
        """Send end-of-day summary."""
        now = datetime.now(self.timezone)

        # Only send once per day, after market close
        if self._last_daily_summary and self._last_daily_summary.date() == now.date():
            return

        account = self._get_account_info()
        positions = self._get_position_risks(account)

        with get_db_session() as session:
            today_start = datetime.combine(now.date(), datetime.min.time())
            trades = session.query(Trade).filter(Trade.executed_at >= today_start).all()

            trades_count = len(trades)
            winning = sum(1 for t in trades if (t.realized_pnl or 0) > 0)
            win_rate = (winning / trades_count * 100) if trades_count > 0 else 0

            # Calculate daily P&L
            realized_pnl = sum(t.realized_pnl or 0 for t in trades)
            unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            total_pnl = realized_pnl + unrealized_pnl
            total_pnl_pct = (total_pnl / account.starting_equity * 100) if account.starting_equity > 0 else 0

            # Record daily performance
            daily_perf = DailyPerformance(
                date=now,
                starting_balance=account.starting_equity,
                ending_balance=account.equity,
                cash_balance=account.cash,
                positions_value=account.positions_value,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                trades_count=trades_count,
                winning_trades=winning,
                losing_trades=trades_count - winning,
                positions_count=len(positions),
            )
            session.add(daily_perf)

        # Send notification (outside session)
        try:
            self.notifications.notify_daily_summary(
                date=now.strftime("%Y-%m-%d"),
                starting_balance=account.starting_equity,
                ending_balance=account.equity,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                trades_count=trades_count,
                win_rate=win_rate,
            )
            logger.info(f"Daily summary sent: PnL ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        except NameError:
            # DB session failed before variables were defined
            logger.error("Failed to send daily summary: DB query failed")
        except Exception as e:
            logger.error(f"Failed to send daily summary notification: {e}")

        self._last_daily_summary = now

    def run(self) -> None:
        """Main trading loop."""
        logger.info(f"Starting trading bot (paper={self.paper_mode})")
        logger.info(f"Watching symbols: {self.symbols}")

        self.audit.log_event("bot_started", {
            "paper_mode": self.paper_mode,
            "symbols": self.symbols,
            "strategy": self.strategy.name,
        })

        # Send startup notification
        mode = "PAPER" if self.paper_mode else "LIVE"
        self.notifications.notify_bot_started(mode, self.symbols)

        # Initial position reconciliation (1.2)
        try:
            self._reconcile_positions()
        except Exception as e:
            logger.error(f"Initial reconciliation failed: {e}")

        # Rebuild in-flight order set from database to prevent duplicate orders after restart
        try:
            with get_db_session() as session:
                pending = session.query(TrackedOrder).filter(
                    TrackedOrder.status.in_(["submitted", "accepted", "partially_filled", "new", "pending"])
                ).all()
                for order in pending:
                    self._pending_orders.add(order.symbol)
                if pending:
                    logger.info(
                        f"Restored {len(pending)} in-flight orders from DB: "
                        f"{[o.symbol for o in pending]}"
                    )
        except Exception as e:
            logger.error(f"Failed to restore in-flight orders: {e}")

        self._running = True

        while self._running:
            try:
                # Check market status
                if not self.broker.is_market_open():
                    logger.debug("Market is closed")

                    # Send daily summary after market close
                    now = datetime.now(self.timezone)
                    if now.hour >= 16 and now.hour < 17:
                        self._send_daily_summary()

                    time.sleep(60)
                    continue

                # Check trading hours
                if not self._is_trading_hours():
                    logger.debug("Outside trading hours")
                    time.sleep(60)
                    continue

                # Check circuit breaker
                if self.risk_manager.is_circuit_breaker_triggered:
                    logger.warning("Circuit breaker triggered - no trading")
                    time.sleep(60)
                    continue

                # Main trading cycle
                logger.debug("Starting trading cycle")

                # Periodic position reconciliation (1.2)
                if self._should_reconcile():
                    try:
                        self._reconcile_positions()
                    except Exception as e:
                        logger.error(f"Periodic reconciliation failed: {e}")

                # Update tracked orders (1.4)
                try:
                    self._update_tracked_orders()
                except Exception as e:
                    logger.error(f"Order tracking update failed: {e}")

                # Record market open equity once per day
                now = datetime.now(self.timezone)
                if (self._market_open_recorded is None or
                    self._market_open_recorded.date() != now.date()):
                    broker_account = self.broker.get_account()
                    self._store_market_open_equity(
                        broker_account.equity,
                        broker_account.cash,
                        broker_account.positions_value
                    )
                    self._market_open_recorded = now

                # 1. Get account info
                account = self._get_account_info()

                # 2. Get current positions
                positions = self._get_position_info()

                # 3. Fetch market data
                market_data = self.data_provider.get_multiple_market_data(
                    list(set(self.symbols) | set(positions.keys()))
                )

                # 4. Check exit signals for existing positions
                self._check_exit_signals(market_data, positions)

                # 5. Generate and process entry signals
                self._process_entry_signals(market_data, positions, account)

                # 6. Heartbeat (4.4)
                self._last_heartbeat = datetime.now(self.timezone)
                self._consecutive_errors = 0  # Reset on successful cycle

                # 7. Sleep until next cycle
                logger.debug(f"Cycle complete, sleeping {self.check_interval}s")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break

            except Exception as e:
                self._consecutive_errors += 1
                logger.error(
                    f"Error in trading loop ({self._consecutive_errors}/"
                    f"{self._max_consecutive_errors}): {e}",
                    exc_info=True,
                )
                self.audit.log_event("trading_loop_error", {
                    "error": str(e),
                    "consecutive_count": self._consecutive_errors,
                })

                if self._consecutive_errors >= self._max_consecutive_errors:
                    msg = (
                        f"CRITICAL: {self._consecutive_errors} consecutive errors. "
                        f"Bot is halting to prevent further damage. "
                        f"Last error: {e}"
                    )
                    logger.critical(msg)
                    self.notifications.notify_error("Bot Halted", msg)
                    self.audit.log_event("bot_halted", {
                        "reason": "consecutive_errors",
                        "count": self._consecutive_errors,
                    })
                    break

                self.notifications.notify_error("Trading Loop Error", str(e))
                time.sleep(self.check_interval)

        # Graceful shutdown (1.5)
        self._graceful_shutdown()

    def run_once(self) -> dict[str, Any]:
        """Run a single trading cycle (useful for testing)."""
        account = self._get_account_info()
        positions = self._get_position_info()
        market_data = self.data_provider.get_multiple_market_data(
            list(set(self.symbols) | set(positions.keys()))
        )

        # Generate signals (but don't execute)
        signals = self.strategy.generate_signals(market_data, positions)

        # Check exit signals
        exit_signals = {}
        for symbol, pos in positions.items():
            if symbol in market_data:
                exit_sig = self.strategy.should_exit(pos, market_data[symbol])
                if exit_sig:
                    exit_signals[symbol] = exit_sig

        return {
            "account": account,
            "positions": positions,
            "signals": signals,
            "exit_signals": exit_signals,
            "market_data": {s: {"price": d.last_price} for s, d in market_data.items()},
        }

    def get_health(self) -> dict[str, Any]:
        """Get bot health status (4.4)."""
        now = datetime.now(self.timezone)
        heartbeat_age = None
        if self._last_heartbeat:
            heartbeat_age = (now - self._last_heartbeat).total_seconds()

        return {
            "status": "running" if self._running else "stopped",
            "paper_mode": self.paper_mode,
            "strategy": self.strategy.name,
            "symbols": self.symbols,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "heartbeat_age_seconds": heartbeat_age,
            "circuit_breaker": self.risk_manager.is_circuit_breaker_triggered,
            "last_reconciliation": self._last_reconciliation.isoformat() if self._last_reconciliation else None,
        }


def main():
    """Entry point for the trading bot."""
    import argparse

    parser = argparse.ArgumentParser(description="Stock Trading Bot")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--test-notifications",
        action="store_true",
        help="Send test notifications and exit",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit",
    )

    args = parser.parse_args()

    # Config validation mode
    if args.validate_config:
        with open(args.config) as f:
            settings = yaml.safe_load(f) or {}
        errors = validate_config(settings)
        if errors:
            print("Configuration errors:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("Configuration is valid")
            return

    try:
        bot = TradingBot(config_path=args.config)
    except ConfigValidationError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.test_notifications:
        results = bot.notifications.test_notifications()
        print(f"Notification test results: {results}")
        return

    if args.once:
        result = bot.run_once()
        print("Single cycle result:")
        print(f"  Account equity: ${result['account'].equity:,.2f}")
        print(f"  Positions: {len(result['positions'])}")
        print(f"  Signals: {len(result['signals'])}")
        for sig in result["signals"]:
            print(f"    - {sig}")
        return

    bot.run()


if __name__ == "__main__":
    main()
