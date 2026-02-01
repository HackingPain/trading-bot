"""
Streamlit dashboard for monitoring the trading bot.

Run with: streamlit run src/dashboard/app.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.models import (
    DailyPerformance,
    Position,
    Signal,
    Trade,
    init_db,
    get_session,
)

# Page configuration
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_settings():
    """Load settings from config file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def init_database():
    """Initialize database connection."""
    settings = load_settings()
    db_url = settings.get("database", {}).get("url", "sqlite:///data/trading_bot.db")
    init_db(db_url)


@st.cache_data(ttl=30)
def get_recent_trades(days: int = 30) -> pd.DataFrame:
    """Get recent trades from database."""
    session = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        trades = session.query(Trade).filter(Trade.created_at >= cutoff).all()

        if not trades:
            return pd.DataFrame()

        data = [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": t.price,
                "total": t.total_value,
                "pnl": t.realized_pnl or 0,
                "pnl_pct": t.realized_pnl_pct or 0,
                "strategy": t.strategy_name,
                "reason": t.signal_reason,
                "paper": t.is_paper_trade,
                "executed_at": t.executed_at,
            }
            for t in trades
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


@st.cache_data(ttl=30)
def get_positions() -> pd.DataFrame:
    """Get current positions from database."""
    session = get_session()
    try:
        positions = session.query(Position).all()

        if not positions:
            return pd.DataFrame()

        data = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "entry_price": p.avg_entry_price,
                "current_price": p.current_price,
                "market_value": p.market_value,
                "cost_basis": p.cost_basis,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct,
                "stop_loss": p.stop_loss_price,
                "trailing_stop": p.trailing_stop_price,
                "take_profit": p.take_profit_price,
                "opened_at": p.opened_at,
            }
            for p in positions
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


@st.cache_data(ttl=30)
def get_daily_performance(days: int = 30) -> pd.DataFrame:
    """Get daily performance data."""
    session = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        perfs = session.query(DailyPerformance).filter(DailyPerformance.date >= cutoff).all()

        if not perfs:
            return pd.DataFrame()

        data = [
            {
                "date": p.date.date(),
                "starting_balance": p.starting_balance,
                "ending_balance": p.ending_balance,
                "total_pnl": p.total_pnl,
                "total_pnl_pct": p.total_pnl_pct,
                "trades_count": p.trades_count,
                "winning_trades": p.winning_trades,
                "losing_trades": p.losing_trades,
                "positions_count": p.positions_count,
            }
            for p in perfs
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


@st.cache_data(ttl=30)
def get_recent_signals(limit: int = 50) -> pd.DataFrame:
    """Get recent signals from database."""
    session = get_session()
    try:
        signals = (
            session.query(Signal)
            .order_by(Signal.created_at.desc())
            .limit(limit)
            .all()
        )

        if not signals:
            return pd.DataFrame()

        data = [
            {
                "symbol": s.symbol,
                "type": s.signal_type.value,
                "strength": s.strength,
                "price": s.price_at_signal,
                "rsi": s.rsi,
                "macd": s.macd,
                "reason": s.reason,
                "executed": s.was_executed,
                "created_at": s.created_at,
            }
            for s in signals
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


def calculate_metrics(trades_df: pd.DataFrame, positions_df: pd.DataFrame) -> dict:
    """Calculate key trading metrics."""
    metrics = {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0,
        "total_pnl": 0,
        "avg_pnl": 0,
        "max_win": 0,
        "max_loss": 0,
        "open_positions": 0,
        "unrealized_pnl": 0,
    }

    if not trades_df.empty:
        sell_trades = trades_df[trades_df["side"] == "sell"]
        metrics["total_trades"] = len(sell_trades)
        metrics["winning_trades"] = len(sell_trades[sell_trades["pnl"] > 0])
        metrics["losing_trades"] = len(sell_trades[sell_trades["pnl"] < 0])
        metrics["win_rate"] = (
            metrics["winning_trades"] / metrics["total_trades"] * 100
            if metrics["total_trades"] > 0
            else 0
        )
        metrics["total_pnl"] = sell_trades["pnl"].sum()
        metrics["avg_pnl"] = sell_trades["pnl"].mean() if len(sell_trades) > 0 else 0
        metrics["max_win"] = sell_trades["pnl"].max() if len(sell_trades) > 0 else 0
        metrics["max_loss"] = sell_trades["pnl"].min() if len(sell_trades) > 0 else 0

    if not positions_df.empty:
        metrics["open_positions"] = len(positions_df)
        metrics["unrealized_pnl"] = positions_df["unrealized_pnl"].sum()

    return metrics


def render_header():
    """Render dashboard header."""
    st.title("📈 Trading Bot Dashboard")

    settings = load_settings()
    paper_mode = settings.get("trading", {}).get("paper_mode", True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        mode_badge = "🟢 PAPER TRADING" if paper_mode else "🔴 LIVE TRADING"
        st.markdown(f"### {mode_badge}")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col3:
        st.markdown(f"**Last updated:** {datetime.now().strftime('%H:%M:%S')}")


def render_metrics(metrics: dict):
    """Render key metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total P&L",
            value=f"${metrics['total_pnl']:,.2f}",
            delta=f"{metrics['total_pnl']:+,.2f}",
        )

    with col2:
        st.metric(
            label="Win Rate",
            value=f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['winning_trades']}/{metrics['total_trades']} trades",
        )

    with col3:
        st.metric(
            label="Open Positions",
            value=str(metrics["open_positions"]),
            delta=f"${metrics['unrealized_pnl']:+,.2f}" if metrics["unrealized_pnl"] else None,
        )

    with col4:
        st.metric(
            label="Avg Trade P&L",
            value=f"${metrics['avg_pnl']:,.2f}",
        )


def render_positions(positions_df: pd.DataFrame):
    """Render current positions table."""
    st.subheader("📊 Current Positions")

    if positions_df.empty:
        st.info("No open positions")
        return

    # Format columns
    display_df = positions_df.copy()
    display_df["entry_price"] = display_df["entry_price"].apply(lambda x: f"${x:.2f}")
    display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}")
    display_df["market_value"] = display_df["market_value"].apply(lambda x: f"${x:,.2f}")
    display_df["unrealized_pnl"] = display_df["unrealized_pnl"].apply(
        lambda x: f"${x:+,.2f}" if x else "$0.00"
    )
    display_df["unrealized_pnl_pct"] = display_df["unrealized_pnl_pct"].apply(
        lambda x: f"{x:+.2f}%" if x else "0.00%"
    )
    display_df["stop_loss"] = display_df["stop_loss"].apply(
        lambda x: f"${x:.2f}" if x else "-"
    )
    display_df["take_profit"] = display_df["take_profit"].apply(
        lambda x: f"${x:.2f}" if x else "-"
    )

    st.dataframe(
        display_df[
            [
                "symbol",
                "quantity",
                "entry_price",
                "current_price",
                "unrealized_pnl",
                "unrealized_pnl_pct",
                "stop_loss",
                "take_profit",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_trades(trades_df: pd.DataFrame):
    """Render recent trades table."""
    st.subheader("📜 Recent Trades")

    if trades_df.empty:
        st.info("No trades yet")
        return

    # Format columns
    display_df = trades_df.copy()
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
    display_df["total"] = display_df["total"].apply(lambda x: f"${x:,.2f}")
    display_df["pnl"] = display_df["pnl"].apply(lambda x: f"${x:+,.2f}" if x else "-")
    display_df["pnl_pct"] = display_df["pnl_pct"].apply(
        lambda x: f"{x:+.2f}%" if x else "-"
    )
    display_df["executed_at"] = pd.to_datetime(display_df["executed_at"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    st.dataframe(
        display_df[
            [
                "executed_at",
                "symbol",
                "side",
                "quantity",
                "price",
                "total",
                "pnl",
                "pnl_pct",
                "reason",
            ]
        ].sort_values("executed_at", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render_performance_chart(daily_df: pd.DataFrame):
    """Render performance chart."""
    st.subheader("📈 Performance Over Time")

    if daily_df.empty:
        st.info("No performance data yet")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["ending_balance"],
            mode="lines+markers",
            name="Account Balance",
            line=dict(color="#00C853", width=2),
        )
    )

    fig.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["total_pnl"],
            name="Daily P&L",
            marker_color=daily_df["total_pnl"].apply(
                lambda x: "#00C853" if x >= 0 else "#FF5252"
            ),
            yaxis="y2",
            opacity=0.6,
        )
    )

    fig.update_layout(
        yaxis=dict(title="Account Balance ($)", side="left"),
        yaxis2=dict(title="Daily P&L ($)", side="right", overlaying="y"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_signals(signals_df: pd.DataFrame):
    """Render recent signals."""
    st.subheader("🎯 Recent Signals")

    if signals_df.empty:
        st.info("No signals generated yet")
        return

    # Format columns
    display_df = signals_df.copy()
    display_df["price"] = display_df["price"].apply(lambda x: f"${x:.2f}")
    display_df["strength"] = display_df["strength"].apply(lambda x: f"{x:.2f}")
    display_df["rsi"] = display_df["rsi"].apply(lambda x: f"{x:.1f}" if x else "-")
    display_df["executed"] = display_df["executed"].apply(lambda x: "✅" if x else "❌")
    display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )

    st.dataframe(
        display_df[
            [
                "created_at",
                "symbol",
                "type",
                "strength",
                "price",
                "rsi",
                "reason",
                "executed",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.header("⚙️ Settings")

        settings = load_settings()

        st.subheader("Trading Configuration")
        st.write(f"**Symbols:** {', '.join(settings.get('trading', {}).get('symbols', []))}")
        st.write(f"**Check Interval:** {settings.get('trading', {}).get('check_interval_seconds', 60)}s")

        st.subheader("Risk Parameters")
        risk = settings.get("risk", {})
        st.write(f"**Max Position:** {risk.get('max_position_pct', 0.1) * 100:.0f}%")
        st.write(f"**Daily Loss Limit:** {risk.get('max_daily_loss_pct', 0.02) * 100:.0f}%")
        st.write(f"**Stop Loss:** {risk.get('stop_loss_pct', 0.05) * 100:.0f}%")
        st.write(f"**Max Daily Trades:** {risk.get('max_daily_trades', 3)}")

        st.subheader("Strategy")
        strategy = settings.get("strategy", {})
        st.write(f"**Profit Target:** {strategy.get('profit_target_pct', 0.02) * 100:.0f}%")
        st.write(f"**RSI Oversold:** {strategy.get('rsi_oversold', 30)}")
        st.write(f"**RSI Overbought:** {strategy.get('rsi_overbought', 70)}")

        st.divider()

        st.subheader("Data Range")
        days = st.slider("Days to show", 7, 90, 30)
        return days


def main():
    """Main dashboard function."""
    # Initialize database
    try:
        init_database()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return

    # Render header
    render_header()

    # Render sidebar and get settings
    days = render_sidebar()

    # Load data
    trades_df = get_recent_trades(days)
    positions_df = get_positions()
    daily_df = get_daily_performance(days)
    signals_df = get_recent_signals()

    # Calculate metrics
    metrics = calculate_metrics(trades_df, positions_df)

    # Render metrics
    render_metrics(metrics)

    st.divider()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Positions", "📜 Trades", "📈 Performance", "🎯 Signals"]
    )

    with tab1:
        render_positions(positions_df)

    with tab2:
        render_trades(trades_df)

    with tab3:
        render_performance_chart(daily_df)

    with tab4:
        render_signals(signals_df)


if __name__ == "__main__":
    main()
