#!/usr/bin/env python3
"""
Backtest CLI - Run backtests from the command line.

Usage:
    python -m src.cli.backtest --symbols AAPL MSFT --start 2023-01-01 --end 2024-01-01
    python -m src.cli.backtest --config config/backtest.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yaml

from ..backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from ..data.market_data import MarketDataProvider, MarketDataConfig
from ..strategies.daily_profit_taker import DailyProfitTakerStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_strategy(strategy_name: str, settings: dict):
    """Load strategy by name."""
    strategies = {
        "daily_profit_taker": DailyProfitTakerStrategy,
    }

    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")

    return strategies[strategy_name].from_settings(settings)


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%m-%d-%Y", "%m/%d/%Y"]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse date: {date_str}. Use YYYY-MM-DD format.")


def run_backtest(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    strategy_name: str = "daily_profit_taker",
    initial_capital: float = 100000.0,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.001,
    max_position_pct: float = 0.10,
    stop_loss_pct: float = 0.05,
    take_profit_pct: float = 0.02,
    output_file: Optional[str] = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    Run a backtest with the specified parameters.

    Args:
        symbols: List of stock symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        strategy_name: Name of strategy to use
        initial_capital: Starting capital
        commission_pct: Commission as percentage of trade value
        slippage_pct: Estimated slippage percentage
        max_position_pct: Maximum position size as percentage of portfolio
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
        output_file: Optional file to save results
        verbose: Enable verbose output

    Returns:
        BacktestResult object
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Starting backtest: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")

    # Create configuration
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    # Create data provider
    data_provider = MarketDataProvider(MarketDataConfig())

    # Load strategy
    settings = {
        "strategy": {
            "profit_target_pct": take_profit_pct,
            "use_trailing_stop": True,
        },
        "risk": {
            "stop_loss_pct": stop_loss_pct,
        },
    }
    strategy = load_strategy(strategy_name, settings)

    # Create and run backtest engine
    engine = BacktestEngine(strategy, data_provider, config)
    result = engine.run(symbols, start_date, end_date)

    # Print results
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Print trade list if verbose
    if verbose and result.trades:
        print("\nTrade History:")
        print("-" * 60)
        for i, trade in enumerate(result.trades, 1):
            pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
            print(
                f"{i:3}. {trade.symbol:5} | "
                f"{trade.entry_date.date()} -> {trade.exit_date.date()} | "
                f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f} | "
                f"{pnl_str} ({trade.pnl_pct:+.2f}%)"
            )

    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        elif output_file.endswith(".yaml") or output_file.endswith(".yml"):
            with open(output_path, "w") as f:
                yaml.dump(result.to_dict(), f, default_flow_style=False)
        else:
            # Default to text summary
            with open(output_path, "w") as f:
                f.write(result.summary())

        logger.info(f"Results saved to: {output_path}")

    return result


def main():
    """Main entry point for backtest CLI."""
    parser = argparse.ArgumentParser(
        description="Run backtests on trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest with defaults
  python -m src.cli.backtest -s AAPL MSFT GOOGL

  # Custom date range
  python -m src.cli.backtest -s AAPL --start 2023-01-01 --end 2024-01-01

  # With custom parameters
  python -m src.cli.backtest -s AAPL MSFT --capital 50000 --stop-loss 0.03

  # Save results to file
  python -m src.cli.backtest -s AAPL -o results/backtest.json

  # Verbose output with trade list
  python -m src.cli.backtest -s AAPL MSFT -v
        """,
    )

    # Required arguments
    parser.add_argument(
        "-s", "--symbols",
        nargs="+",
        required=True,
        help="Stock symbols to backtest (e.g., AAPL MSFT GOOGL)",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 1 year ago",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today",
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="daily_profit_taker",
        choices=["daily_profit_taker"],
        help="Trading strategy to use",
    )

    # Capital and risk
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=0.10,
        help="Max position size as decimal (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss as decimal (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.02,
        help="Take profit as decimal (default: 0.02 = 2%%)",
    )

    # Costs
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission as decimal (default: 0.0)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.001,
        help="Slippage estimate as decimal (default: 0.001 = 0.1%%)",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file for results (.json, .yaml, or .txt)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = parse_date(args.end)
    else:
        end_date = datetime.now()

    if args.start:
        start_date = parse_date(args.start)
    else:
        start_date = end_date - timedelta(days=365)

    # Validate dates
    if start_date >= end_date:
        print("Error: Start date must be before end date", file=sys.stderr)
        sys.exit(1)

    # Run backtest
    try:
        result = run_backtest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            strategy_name=args.strategy,
            initial_capital=args.capital,
            commission_pct=args.commission,
            slippage_pct=args.slippage,
            max_position_pct=args.max_position,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            output_file=args.output,
            verbose=args.verbose,
        )

        # Exit with appropriate code
        if result.metrics.total_trades == 0:
            print("\nWarning: No trades were executed during backtest period")
            sys.exit(0)

        sys.exit(0)

    except KeyboardInterrupt:
        print("\nBacktest cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
