"""SQLAlchemy database models for the trading bot."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLEnum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SignalType(str, Enum):
    """Signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Trade(Base):
    """Represents an executed trade."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    side: Mapped[OrderSide] = mapped_column(SQLEnum(OrderSide), nullable=False)
    order_type: Mapped[OrderType] = mapped_column(SQLEnum(OrderType), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    total_value: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, default=0.0)

    # Order tracking
    order_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[OrderStatus] = mapped_column(SQLEnum(OrderStatus), nullable=False)

    # P&L (calculated on close)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Strategy info
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Risk management
    stop_loss_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    is_paper_trade: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side.value}, "
            f"qty={self.quantity}, price={self.price}, status={self.status.value})>"
        )


class Position(Base):
    """Represents a current open position."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True, index=True)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    avg_entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    market_value: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis: Mapped[float] = mapped_column(Float, nullable=False)

    # Unrealized P&L
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Risk levels
    stop_loss_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trailing_stop_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    highest_price: Mapped[float] = mapped_column(Float, nullable=False)  # For trailing stop

    # Metadata
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<Position(symbol={self.symbol}, qty={self.quantity}, "
            f"entry={self.avg_entry_price}, pnl={self.unrealized_pnl:.2f})>"
        )

    def update_price(self, current_price: float) -> None:
        """Update position with current market price."""
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (self.unrealized_pnl / self.cost_basis) * 100 if self.cost_basis else 0

        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price


class DailyPerformance(Base):
    """End-of-day performance snapshot."""

    __tablename__ = "daily_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, unique=True, index=True)

    # Account values
    starting_balance: Mapped[float] = mapped_column(Float, nullable=False)
    ending_balance: Mapped[float] = mapped_column(Float, nullable=False)
    cash_balance: Mapped[float] = mapped_column(Float, nullable=False)
    positions_value: Mapped[float] = mapped_column(Float, nullable=False)

    # Daily P&L
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)

    # Trading activity
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)

    # Risk metrics
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Positions held
    positions_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<DailyPerformance(date={self.date.date()}, "
            f"pnl={self.total_pnl:.2f}, trades={self.trades_count})>"
        )


class DailyState(Base):
    """Stores intraday state like market open equity."""

    __tablename__ = "daily_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, unique=True, index=True)
    market_open_equity: Mapped[float] = mapped_column(Float, nullable=False)
    market_open_cash: Mapped[float] = mapped_column(Float, nullable=False)
    market_open_positions_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<DailyState(date={self.date.date()}, equity={self.market_open_equity:.2f})>"


class Signal(Base):
    """Generated trading signal (for analysis/backtesting)."""

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    signal_type: Mapped[SignalType] = mapped_column(SQLEnum(SignalType), nullable=False)
    strength: Mapped[float] = mapped_column(Float, default=1.0)  # 0.0 to 1.0

    # Price at signal
    price_at_signal: Mapped[float] = mapped_column(Float, nullable=False)

    # Indicator values at signal
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bollinger_position: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # -1 to 1

    # Strategy info
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)

    # Execution
    was_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    trade_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return (
            f"<Signal(symbol={self.symbol}, type={self.signal_type.value}, "
            f"strength={self.strength:.2f}, executed={self.was_executed})>"
        )


# Database initialization
_engine = None
_SessionLocal = None


def init_db(database_url: str = "sqlite:///data/trading_bot.db", echo: bool = False):
    """Initialize the database connection and create tables."""
    global _engine, _SessionLocal

    _engine = create_engine(database_url, echo=echo)
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine)

    return _engine


def get_session():
    """Get a database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
