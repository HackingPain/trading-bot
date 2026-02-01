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
from src.backtest.metrics import PerformanceMetrics, calculate_metrics, calculate_daily_metrics

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


def render_analytics(trades_df: pd.DataFrame, daily_df: pd.DataFrame):
    """Render advanced analytics section."""
    st.subheader("📊 Performance Analytics")

    if daily_df.empty and trades_df.empty:
        st.info("Not enough data for analytics. Start trading to see metrics.")
        return

    # Create two columns for metrics
    col1, col2 = st.columns(2)

    # Calculate metrics from trades
    sell_trades = trades_df[trades_df["side"] == "sell"] if not trades_df.empty else pd.DataFrame()

    with col1:
        st.markdown("### Trade Statistics")
        if not sell_trades.empty:
            total_trades = len(sell_trades)
            winning = len(sell_trades[sell_trades["pnl"] > 0])
            losing = len(sell_trades[sell_trades["pnl"] < 0])
            win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

            gross_profit = sell_trades[sell_trades["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(sell_trades[sell_trades["pnl"] < 0]["pnl"].sum())
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

            avg_win = sell_trades[sell_trades["pnl"] > 0]["pnl"].mean() if winning > 0 else 0
            avg_loss = sell_trades[sell_trades["pnl"] < 0]["pnl"].mean() if losing > 0 else 0

            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
            st.metric("Avg Winner", f"${avg_win:,.2f}")
            st.metric("Avg Loser", f"${avg_loss:,.2f}")
        else:
            st.write("No completed trades yet")

    with col2:
        st.markdown("### P&L Breakdown")
        if not sell_trades.empty:
            total_pnl = sell_trades["pnl"].sum()
            max_win = sell_trades["pnl"].max()
            max_loss = sell_trades["pnl"].min()

            # Calculate expectancy
            if len(sell_trades) > 0:
                expectancy = sell_trades["pnl"].mean()
            else:
                expectancy = 0

            st.metric("Net P&L", f"${total_pnl:,.2f}")
            st.metric("Best Trade", f"${max_win:,.2f}")
            st.metric("Worst Trade", f"${max_loss:,.2f}")
            st.metric("Expectancy", f"${expectancy:,.2f}")
            st.metric("Gross Profit", f"${gross_profit:,.2f}")
            st.metric("Gross Loss", f"${gross_loss:,.2f}")
        else:
            st.write("No P&L data yet")

    # Monthly returns chart
    st.markdown("### Monthly Returns")
    if not daily_df.empty:
        daily_df_copy = daily_df.copy()
        daily_df_copy["month"] = pd.to_datetime(daily_df_copy["date"]).dt.to_period("M").astype(str)
        monthly_pnl = daily_df_copy.groupby("month")["total_pnl"].sum().reset_index()

        if not monthly_pnl.empty:
            fig = go.Figure()
            colors = ["#00C853" if x >= 0 else "#FF5252" for x in monthly_pnl["total_pnl"]]
            fig.add_trace(go.Bar(
                x=monthly_pnl["month"],
                y=monthly_pnl["total_pnl"],
                marker_color=colors,
                text=[f"${x:,.0f}" for x in monthly_pnl["total_pnl"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Monthly P&L",
                xaxis_title="Month",
                yaxis_title="P&L ($)",
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data available")

    # Symbol performance breakdown
    st.markdown("### Performance by Symbol")
    if not sell_trades.empty:
        symbol_stats = sell_trades.groupby("symbol").agg({
            "pnl": ["sum", "count", "mean"],
        }).reset_index()
        symbol_stats.columns = ["Symbol", "Total P&L", "Trades", "Avg P&L"]

        # Calculate win rate per symbol
        def calc_win_rate(group):
            wins = (group["pnl"] > 0).sum()
            total = len(group)
            return (wins / total * 100) if total > 0 else 0

        win_rates = sell_trades.groupby("symbol").apply(calc_win_rate).reset_index()
        win_rates.columns = ["Symbol", "Win Rate"]

        symbol_stats = symbol_stats.merge(win_rates, on="Symbol")
        symbol_stats = symbol_stats.sort_values("Total P&L", ascending=False)

        # Format for display
        symbol_stats["Total P&L"] = symbol_stats["Total P&L"].apply(lambda x: f"${x:,.2f}")
        symbol_stats["Avg P&L"] = symbol_stats["Avg P&L"].apply(lambda x: f"${x:,.2f}")
        symbol_stats["Win Rate"] = symbol_stats["Win Rate"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(symbol_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No symbol data available")

    # Drawdown chart
    st.markdown("### Equity Curve & Drawdown")
    if not daily_df.empty and "ending_balance" in daily_df.columns:
        equity = daily_df.set_index("date")["ending_balance"]
        rolling_max = equity.cummax()
        drawdown = ((equity - rolling_max) / rolling_max) * 100

        fig = go.Figure()

        # Equity curve
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=2),
        ))

        # Drawdown
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#FF5252", width=1),
            yaxis="y2",
        ))

        fig.update_layout(
            title="Equity Curve with Drawdown",
            yaxis=dict(title="Equity ($)", side="left"),
            yaxis2=dict(title="Drawdown (%)", side="right", overlaying="y", range=[-50, 5]),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Max drawdown stat
        max_dd = drawdown.min()
        st.metric("Maximum Drawdown", f"{max_dd:.2f}%")
    else:
        st.info("No equity data available for drawdown analysis")


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Positions", "📜 Trades", "📈 Performance", "🎯 Signals", "📉 Analytics"]
    )

    with tab1:
        render_positions(positions_df)

    with tab2:
        render_trades(trades_df)

    with tab3:
        render_performance_chart(daily_df)

    with tab4:
        render_signals(signals_df)

    with tab5:
        render_analytics(trades_df, daily_df)


if __name__ == "__main__":
    main()
