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
    init_db,
    get_session,
)
from .execution.broker import AlpacaBroker, Order, OrderSide as BrokerOrderSide, OrderType
from .notifications.alerts import NotificationManager, NotificationType
from .risk.risk_manager import AccountInfo, PositionRisk, RiskManager
from .strategies.base import PositionInfo, SignalType
from .strategies.daily_profit_taker import DailyProfitTakerStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot class.

    Orchestrates the trading loop:
    1. Fetch market data
    2. Update positions
    3. Check for exit signals
    4. Generate entry signals
    5. Run risk checks
    6. Execute orders
    7. Send notifications
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.settings = self._load_settings()

        # Core settings
        self.paper_mode = self.settings.get("trading", {}).get("paper_mode", True)
        self.symbols = self.settings.get("trading", {}).get("symbols", [])
        self.check_interval = self.settings.get("trading", {}).get("check_interval_seconds", 60)
        self.timezone = pytz.timezone(
            self.settings.get("trading", {}).get("timezone", "America/New_York")
        )

        # Safety check
        if not self.paper_mode:
            logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
            self._confirm_live_trading()

        # Initialize components
        self._init_components()

        # State
        self._running = False
        self._last_daily_summary: Optional[datetime] = None
        self._market_open_recorded: Optional[datetime] = None

        # Register signal handlers
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

        # Strategy
        self.strategy = DailyProfitTakerStrategy.from_settings(self.settings)
        logger.info(f"Strategy initialized: {self.strategy.name}")

        # Risk manager
        self.risk_manager = RiskManager.from_settings(self.settings)
        logger.info("Risk manager initialized")

        # Broker
        self.broker = AlpacaBroker.from_settings(self.settings)
        logger.info(f"Broker initialized (paper={self.paper_mode})")

        # Notifications
        self.notifications = NotificationManager.from_settings(self.settings)
        logger.info("Notification manager initialized")

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

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received shutdown signal ({signum})")
        self._running = False

    def _is_trading_hours(self) -> bool:
        """Check if current time is within trading hours."""
        trading_settings = self.settings.get("trading", {})
        trading_hours = trading_settings.get("trading_hours", {})

        start_str = trading_hours.get("start", "09:30")
        end_str = trading_hours.get("end", "16:00")

        now = datetime.now(self.timezone)
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()

        return start_time <= now.time() <= end_time

    def _store_market_open_equity(self, equity: float, cash: float, positions_value: float) -> None:
        """Store market open equity for today."""
        session = get_session()
        try:
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
                session.commit()
                logger.info(f"Stored market open equity: ${equity:,.2f}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store market open equity: {e}")
        finally:
            session.close()

    def _get_market_open_equity(self) -> Optional[float]:
        """Get market open equity for today."""
        session = get_session()
        try:
            today = datetime.now(self.timezone).date()
            today_start = datetime.combine(today, datetime.min.time())

            daily_state = session.query(DailyState).filter(
                DailyState.date >= today_start
            ).first()

            return daily_state.market_open_equity if daily_state else None
        except Exception as e:
            logger.error(f"Failed to get market open equity: {e}")
            return None
        finally:
            session.close()

    def _calculate_daily_pnl(self) -> float:
        """Calculate today's realized P&L from closed trades."""
        session = get_session()
        try:
            today = datetime.now(self.timezone).date()
            today_start = datetime.combine(today, datetime.min.time())

            trades = session.query(Trade).filter(
                Trade.executed_at >= today_start,
                Trade.realized_pnl.isnot(None)
            ).all()

            return sum(t.realized_pnl or 0 for t in trades)
        except Exception as e:
            logger.error(f"Failed to calculate daily P&L: {e}")
            return 0.0
        finally:
            session.close()

    def _get_account_info(self) -> AccountInfo:
        """Get current account information."""
        account = self.broker.get_account()

        # Get starting equity for the day from database
        starting_equity = self._get_market_open_equity()
        if starting_equity is None:
            starting_equity = account.equity

        # Calculate daily P&L from realized trades
        daily_pnl = self._calculate_daily_pnl()

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

        session = get_session()
        try:
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
        finally:
            session.close()

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
        session = get_session()
        try:
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

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update position in DB: {e}")
        finally:
            session.close()

    def _remove_position_from_db(self, symbol: str) -> None:
        """Remove position from database."""
        session = get_session()
        try:
            session.query(DBPosition).filter_by(symbol=symbol).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to remove position from DB: {e}")
        finally:
            session.close()

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
        session = get_session()
        try:
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
            session.commit()
            logger.info(f"Recorded trade: {trade}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record trade: {e}")
        finally:
            session.close()

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
        session = get_session()
        try:
            db_signal_type = {
                SignalType.BUY: DBSignalType.BUY,
                SignalType.SELL: DBSignalType.SELL,
                SignalType.HOLD: DBSignalType.HOLD,
            }.get(signal_type, DBSignalType.HOLD)

            signal = DBSignal(
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
            session.add(signal)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record signal: {e}")
        finally:
            session.close()

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

        # Create sell order
        order = Order(
            symbol=symbol,
            side=BrokerOrderSide.SELL,
            quantity=int(position.quantity),
            order_type=OrderType.MARKET,
        )

        # Submit order
        result = self.broker.submit_order(order)

        if result.success:
            # Calculate P&L
            pnl = position.unrealized_pnl
            pnl_pct = position.unrealized_pnl_pct

            # Record trade
            self._record_trade(
                symbol=symbol,
                side="sell",
                quantity=position.quantity,
                price=position.current_price,
                order_id=result.order_id,
                reason=exit_signal.description,
                realized_pnl=pnl,
                realized_pnl_pct=pnl_pct,
            )

            # Remove from DB
            self._remove_position_from_db(symbol)

            # Track day trade
            self.risk_manager.record_trade()

            # Send notification
            if exit_signal.reason.value in ("stop_loss", "trailing_stop"):
                self.notifications.notify_stop_loss(
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    exit_price=position.current_price,
                    loss=pnl,
                    loss_pct=pnl_pct,
                )
            elif exit_signal.reason.value == "take_profit":
                self.notifications.notify_take_profit(
                    symbol=symbol,
                    quantity=position.quantity,
                    entry_price=position.avg_entry_price,
                    exit_price=position.current_price,
                    profit=pnl,
                    profit_pct=pnl_pct,
                )
            else:
                self.notifications.notify_trade(
                    symbol=symbol,
                    side="sell",
                    quantity=position.quantity,
                    price=position.current_price,
                    total=position.current_price * position.quantity,
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

            logger.info(
                f"Opened position {symbol}: {shares} shares @ ${fill_price:.2f} "
                f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
            )
        else:
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

        # Get today's trades
        session = get_session()
        try:
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
            session.commit()

            # Send notification
            self.notifications.notify_daily_summary(
                date=now.strftime("%Y-%m-%d"),
                starting_balance=account.starting_equity,
                ending_balance=account.equity,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                trades_count=trades_count,
                win_rate=win_rate,
            )

            self._last_daily_summary = now
            logger.info(f"Daily summary sent: PnL ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to send daily summary: {e}")
        finally:
            session.close()

    def run(self) -> None:
        """Main trading loop."""
        logger.info(f"Starting trading bot (paper={self.paper_mode})")
        logger.info(f"Watching symbols: {self.symbols}")

        # Send startup notification
        mode = "PAPER" if self.paper_mode else "LIVE"
        self.notifications.notify_bot_started(mode, self.symbols)

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

                # 6. Sleep until next cycle
                logger.debug(f"Cycle complete, sleeping {self.check_interval}s")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt")
                break

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                self.notifications.notify_error("Trading Loop Error", str(e))
                time.sleep(self.check_interval)

        # Shutdown
        logger.info("Shutting down trading bot")
        self.notifications.notify_bot_stopped("Manual shutdown")

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

    args = parser.parse_args()

    bot = TradingBot(config_path=args.config)

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
