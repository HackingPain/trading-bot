"""Walk-forward analysis and Monte Carlo simulation for strategy validation."""

import copy
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import PerformanceMetrics, Trade, calculate_metrics
from ..strategies.base import Strategy
from ..data.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold."""
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_result: BacktestResult
    out_of_sample_result: BacktestResult

    @property
    def overfitting_ratio(self) -> float:
        """Ratio of out-of-sample to in-sample return.

        A value close to 1.0 indicates low overfitting.
        Values much less than 1.0 suggest the strategy is overfit to training data.
        """
        is_return = self.in_sample_result.metrics.total_return_pct
        oos_return = self.out_of_sample_result.metrics.total_return_pct
        if is_return == 0:
            return 0.0
        return oos_return / is_return


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis."""
    folds: list[WalkForwardFold]
    config: BacktestConfig
    train_window_days: int
    test_window_days: int

    # Aggregated out-of-sample metrics (computed post-init)
    aggregated_oos_metrics: Optional[PerformanceMetrics] = field(default=None)

    def __post_init__(self) -> None:
        if self.aggregated_oos_metrics is None:
            self.aggregated_oos_metrics = self._aggregate_oos_metrics()

    # -- Derived properties --------------------------------------------------

    @property
    def overfitting_ratio(self) -> float:
        """Average overfitting ratio across all folds."""
        ratios = [f.overfitting_ratio for f in self.folds]
        return float(np.mean(ratios)) if ratios else 0.0

    @property
    def in_sample_metrics(self) -> list[PerformanceMetrics]:
        return [f.in_sample_result.metrics for f in self.folds]

    @property
    def out_of_sample_metrics(self) -> list[PerformanceMetrics]:
        return [f.out_of_sample_result.metrics for f in self.folds]

    @property
    def oos_sharpe_ratios(self) -> list[float]:
        return [m.sharpe_ratio for m in self.out_of_sample_metrics]

    @property
    def is_vs_oos_comparison(self) -> pd.DataFrame:
        """DataFrame comparing in-sample vs out-of-sample metrics per fold."""
        rows: list[dict] = []
        for fold in self.folds:
            is_m = fold.in_sample_result.metrics
            oos_m = fold.out_of_sample_result.metrics
            rows.append({
                "fold": fold.fold_index,
                "is_return_pct": is_m.total_return_pct,
                "oos_return_pct": oos_m.total_return_pct,
                "is_sharpe": is_m.sharpe_ratio,
                "oos_sharpe": oos_m.sharpe_ratio,
                "is_max_dd_pct": is_m.max_drawdown_pct,
                "oos_max_dd_pct": oos_m.max_drawdown_pct,
                "is_win_rate": is_m.win_rate,
                "oos_win_rate": oos_m.win_rate,
                "is_trades": is_m.total_trades,
                "oos_trades": oos_m.total_trades,
                "overfitting_ratio": fold.overfitting_ratio,
            })
        return pd.DataFrame(rows)

    # -- Private helpers -----------------------------------------------------

    def _aggregate_oos_metrics(self) -> PerformanceMetrics:
        """Combine all out-of-sample equity curves and trades into one metric set."""
        if not self.folds:
            return PerformanceMetrics()

        all_trades: list[Trade] = []
        equity_parts: list[pd.Series] = []

        for fold in self.folds:
            result = fold.out_of_sample_result
            all_trades.extend(result.trades)
            if not result.equity_curve.empty:
                equity_parts.append(result.equity_curve)

        if not equity_parts:
            return PerformanceMetrics()

        # Chain equity curves: scale each subsequent curve so that it starts
        # where the previous one ended.
        combined = equity_parts[0].copy()
        for part in equity_parts[1:]:
            if part.empty or combined.empty:
                continue
            scale = combined.iloc[-1] / part.iloc[0]
            scaled = part * scale
            # Drop the first point to avoid duplicate at the boundary
            combined = pd.concat([combined, scaled.iloc[1:]])

        return calculate_metrics(combined, all_trades)

    # -- Summary -------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of walk-forward results."""
        n = len(self.folds)
        agg = self.aggregated_oos_metrics or PerformanceMetrics()
        comp = self.is_vs_oos_comparison

        lines = [
            "Walk-Forward Analysis Results",
            "=" * 40,
            f"Folds:                {n}",
            f"Train window:         {self.train_window_days} days",
            f"Test window:          {self.test_window_days} days",
            "",
            "Aggregated Out-of-Sample",
            "-" * 40,
            f"Total Return:         {agg.total_return_pct:.2f}%",
            f"Sharpe Ratio:         {agg.sharpe_ratio:.2f}",
            f"Max Drawdown:         {agg.max_drawdown_pct:.2f}%",
            f"Win Rate:             {agg.win_rate:.1f}%",
            f"Total Trades:         {agg.total_trades}",
            "",
            f"Avg Overfitting Ratio: {self.overfitting_ratio:.3f}",
            "",
            "Per-Fold Comparison (IS vs OOS return %)",
            "-" * 40,
        ]

        if not comp.empty:
            for _, row in comp.iterrows():
                lines.append(
                    f"  Fold {int(row['fold'])}: IS={row['is_return_pct']:+.2f}%  "
                    f"OOS={row['oos_return_pct']:+.2f}%  "
                    f"Ratio={row['overfitting_ratio']:.3f}"
                )

        return "\n".join(lines)


class WalkForwardAnalyzer:
    """
    Performs walk-forward analysis on a trading strategy.

    The data is divided into successive train/test windows.  For each fold the
    strategy is backtested on the training window (in-sample) and then on the
    immediately following test window (out-of-sample).  The test window is then
    advanced forward and the process repeats until the data is exhausted.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_provider: MarketDataProvider,
        config: Optional[BacktestConfig] = None,
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.config = config or BacktestConfig()

    def run(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        train_window_days: int = 252,
        test_window_days: int = 63,
    ) -> WalkForwardResult:
        """
        Execute walk-forward analysis.

        Args:
            symbols: Symbols to trade.
            start_date: Overall start date.
            end_date: Overall end date.
            train_window_days: Length of the training (in-sample) window in days.
            test_window_days: Length of the testing (out-of-sample) window in days.

        Returns:
            WalkForwardResult containing all fold results and aggregated metrics.
        """
        logger.info(
            f"Starting walk-forward analysis: {start_date.date()} to {end_date.date()}, "
            f"train={train_window_days}d, test={test_window_days}d"
        )

        folds: list[WalkForwardFold] = []
        fold_index = 0

        current_train_start = start_date

        while True:
            train_end = current_train_start + timedelta(days=train_window_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_window_days)

            # Stop if the test window extends past the data boundary
            if test_end > end_date:
                # Allow a partial final test window if at least half the days remain
                remaining = (end_date - test_start).days
                if remaining >= test_window_days // 2:
                    test_end = end_date
                else:
                    break

            logger.info(
                f"Fold {fold_index}: train {current_train_start.date()} -> "
                f"{train_end.date()}, test {test_start.date()} -> {test_end.date()}"
            )

            # --- In-sample backtest --- (Fix #14: deep-copy strategy to isolate folds)
            is_strategy = copy.deepcopy(self.strategy)
            is_engine = BacktestEngine(is_strategy, self.data_provider, self.config)
            is_result = is_engine.run(symbols, current_train_start, train_end)

            # --- Out-of-sample backtest ---
            oos_strategy = copy.deepcopy(self.strategy)
            oos_engine = BacktestEngine(oos_strategy, self.data_provider, self.config)
            oos_result = oos_engine.run(symbols, test_start, test_end)

            fold = WalkForwardFold(
                fold_index=fold_index,
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_result=is_result,
                out_of_sample_result=oos_result,
            )
            folds.append(fold)

            fold_index += 1
            # Advance by the test window length (anchored walk-forward would keep
            # train_start fixed; here we use a rolling window).
            current_train_start += timedelta(days=test_window_days)

        logger.info(f"Walk-forward analysis complete: {len(folds)} folds")

        return WalkForwardResult(
            folds=folds,
            config=self.config,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
        )


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation of trade sequences."""
    n_simulations: int
    confidence_levels: list[float]

    # Distributions (length == n_simulations)
    final_equity_distribution: np.ndarray  # final equity for each sim
    max_drawdown_distribution: np.ndarray  # max drawdown pct for each sim

    # Percentile equity curves: dict mapping percentile -> equity array
    percentile_curves: dict[float, np.ndarray]

    # Confidence intervals on total return %
    return_confidence_intervals: dict[float, float]

    # Original stats for reference
    original_final_equity: float
    initial_capital: float

    @property
    def median_final_equity(self) -> float:
        return float(np.median(self.final_equity_distribution))

    @property
    def mean_final_equity(self) -> float:
        return float(np.mean(self.final_equity_distribution))

    @property
    def median_max_drawdown(self) -> float:
        return float(np.median(self.max_drawdown_distribution))

    @property
    def worst_case_drawdown(self) -> float:
        """95th-percentile worst drawdown."""
        return float(np.percentile(self.max_drawdown_distribution, 95))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Monte Carlo Simulation Results",
            "=" * 40,
            f"Simulations:          {self.n_simulations}",
            f"Initial Capital:      ${self.initial_capital:,.2f}",
            f"Original Final Equity:${self.original_final_equity:,.2f}",
            "",
            "Final Equity Distribution",
            "-" * 40,
            f"  Mean:               ${self.mean_final_equity:,.2f}",
            f"  Median:             ${self.median_final_equity:,.2f}",
            f"  Std Dev:            ${np.std(self.final_equity_distribution):,.2f}",
            "",
            "Return Confidence Intervals",
            "-" * 40,
        ]
        for level in sorted(self.return_confidence_intervals.keys()):
            val = self.return_confidence_intervals[level]
            lines.append(f"  {level*100:5.1f}th percentile: {val:+.2f}%")

        lines += [
            "",
            "Max Drawdown Distribution",
            "-" * 40,
            f"  Median Drawdown:    {self.median_max_drawdown:.2f}%",
            f"  95th Pctl (worst):  {self.worst_case_drawdown:.2f}%",
        ]
        return "\n".join(lines)


class MonteCarloSimulator:
    """
    Monte Carlo simulation that randomly reshuffles trade order to estimate
    the distribution of possible equity paths and risk metrics.

    This helps answer: *how much of the backtest result was due to the
    specific order trades occurred in?*
    """

    def __init__(
        self,
        trades: list[Trade],
        initial_capital: float = 100_000.0,
    ):
        if not trades:
            raise ValueError("At least one trade is required for Monte Carlo simulation")
        self.trades = trades
        self.initial_capital = initial_capital

        # Pre-compute PnL array once
        self._pnls = np.array([t.pnl for t in trades])

    def run(
        self,
        n_simulations: int = 1000,
        confidence_levels: Optional[list[float]] = None,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation by reshuffling trade PnLs.

        Args:
            n_simulations: Number of random permutations to generate.
            confidence_levels: Percentile levels for reporting (values 0-1).
            seed: Optional RNG seed for reproducibility.

        Returns:
            MonteCarloResult with distributions and confidence intervals.
        """
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

        logger.info(
            f"Running Monte Carlo simulation: {n_simulations} sims, "
            f"{len(self.trades)} trades"
        )

        rng = np.random.default_rng(seed)

        n_trades = len(self._pnls)
        # Matrix of shuffled PnL sequences: (n_simulations, n_trades)
        shuffled = np.tile(self._pnls, (n_simulations, 1))
        # Shuffle each row independently
        for i in range(n_simulations):
            rng.shuffle(shuffled[i])

        # Build equity curves: cumulative sum of PnLs starting from initial capital
        equity_curves = self.initial_capital + np.cumsum(shuffled, axis=1)

        # Prepend the initial capital column
        initial_col = np.full((n_simulations, 1), self.initial_capital)
        equity_curves = np.hstack([initial_col, equity_curves])

        # Final equity for each simulation
        final_equities = equity_curves[:, -1]

        # Max drawdown for each simulation
        max_drawdowns = self._compute_max_drawdowns(equity_curves)

        # Percentile equity curves
        percentile_curves: dict[float, np.ndarray] = {}
        for level in confidence_levels:
            pct = level * 100
            percentile_curves[level] = np.percentile(equity_curves, pct, axis=0)

        # Return confidence intervals
        returns = ((final_equities - self.initial_capital) / self.initial_capital) * 100
        return_ci: dict[float, float] = {}
        for level in confidence_levels:
            return_ci[level] = float(np.percentile(returns, level * 100))

        # Original (un-shuffled) final equity
        original_equity = self.initial_capital + float(np.sum(self._pnls))

        result = MonteCarloResult(
            n_simulations=n_simulations,
            confidence_levels=confidence_levels,
            final_equity_distribution=final_equities,
            max_drawdown_distribution=max_drawdowns,
            percentile_curves=percentile_curves,
            return_confidence_intervals=return_ci,
            original_final_equity=original_equity,
            initial_capital=self.initial_capital,
        )

        logger.info(
            f"Monte Carlo complete: median equity ${result.median_final_equity:,.2f}, "
            f"median drawdown {result.median_max_drawdown:.2f}%"
        )

        return result

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _compute_max_drawdowns(equity_curves: np.ndarray) -> np.ndarray:
        """Compute the max drawdown percentage for each equity path.

        Args:
            equity_curves: Shape (n_simulations, n_points).

        Returns:
            1-D array of max drawdown percentages (positive values).
        """
        running_max = np.maximum.accumulate(equity_curves, axis=1)

        # Avoid division by zero for paths that start at zero (shouldn't happen)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown_pct = ((equity_curves - running_max) / running_max) * 100

        # Replace any NaN/inf with 0
        drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Max drawdown is the most negative value per row, return as positive pct
        max_dd = np.abs(np.min(drawdown_pct, axis=1))
        return max_dd
