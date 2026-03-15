"""Automated model training pipeline.

Handles data fetching, feature construction, rolling-window train/validate
splits, metric logging, and model persistence.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ..data.market_data import MarketDataConfig, MarketDataProvider
from .features import FeatureEngine
from .model import SignalModel

logger = logging.getLogger(__name__)

# Default directory for persisted models
DEFAULT_MODEL_DIR = Path("models")


class ModelTrainer:
    """
    Orchestrates the full ML training lifecycle.

    Fetches historical data via MarketDataProvider, engineers features
    with FeatureEngine, trains a SignalModel, evaluates on a validation
    window, and persists the result to disk.

    Args:
        data_provider: MarketDataProvider for fetching OHLCV data.
        feature_engine: FeatureEngine for building feature matrices.
        model_dir: Directory where trained models are stored.
        forward_days: Look-ahead window for target construction.
        threshold: Minimum return for a positive label.
    """

    def __init__(
        self,
        data_provider: Optional[MarketDataProvider] = None,
        feature_engine: Optional[FeatureEngine] = None,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        forward_days: int = 5,
        threshold: float = 0.02,
    ):
        self.data_provider = data_provider or MarketDataProvider(MarketDataConfig())
        self.feature_engine = feature_engine or FeatureEngine()
        self.model_dir = Path(model_dir)
        self.forward_days = forward_days
        self.threshold = threshold

        self._last_trained: Optional[datetime] = None
        self._last_metrics: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_on_history(
        self,
        symbols: list[str],
        lookback_days: int = 365,
        validation_days: int = 60,
    ) -> SignalModel:
        """
        Fetch data for *symbols*, build features, and train a model.

        Uses a rolling-window approach: the most recent `validation_days`
        rows are held out for evaluation; the preceding rows form the
        training set.

        Args:
            symbols: List of ticker symbols to include in training data.
            lookback_days: Total calendar days of history to fetch.
            validation_days: Number of most-recent trading days reserved
                for validation.

        Returns:
            A trained SignalModel instance (also saved to disk).
        """
        logger.info(
            f"Starting training pipeline: {len(symbols)} symbols, "
            f"lookback={lookback_days}d, validation={validation_days}d"
        )

        # --- 1. Fetch & combine data ---
        all_features: list[pd.DataFrame] = []
        all_targets: list[pd.Series] = []

        period = self._days_to_period(lookback_days)

        for symbol in symbols:
            try:
                df = self.data_provider.get_historical_data(
                    symbol, period=period, interval="1d"
                )
                if df.empty or len(df) < 60:
                    logger.warning(
                        f"Skipping {symbol}: insufficient data ({len(df)} rows)"
                    )
                    continue

                features = self.feature_engine.build_features(df)
                target = self.feature_engine.build_target(
                    df,
                    forward_days=self.forward_days,
                    threshold=self.threshold,
                )

                # Tag with symbol for traceability (not used as feature)
                features["_symbol"] = symbol
                all_features.append(features)
                all_targets.append(target)

                logger.info(f"Processed {symbol}: {len(features)} rows")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                continue

        if not all_features:
            raise ValueError("No valid training data collected from any symbol.")

        combined_features = pd.concat(all_features, axis=0)
        combined_target = pd.concat(all_targets, axis=0)

        # Remove symbol tag before training
        symbols_col = combined_features.pop("_symbol")

        # Sort by date index to prevent look-ahead bias in train/val split (Fix #10)
        combined_features = combined_features.sort_index()
        combined_target = combined_target.reindex(combined_features.index)

        # --- 2. Clean ---
        mask = combined_features.notna().all(axis=1) & combined_target.notna()
        X = combined_features.loc[mask]
        y = combined_target.loc[mask]

        logger.info(
            f"Combined dataset: {len(X)} usable rows, "
            f"{len(X.columns)} features, "
            f"positive rate {y.mean():.3f}"
        )

        # --- 3. Rolling-window split ---
        X_train, X_val, y_train, y_val = self._rolling_split(
            X, y, validation_days
        )

        # --- 4. Train ---
        model = SignalModel()
        model.train(X_train, y_train)

        # --- 5. Evaluate ---
        self._evaluate_and_log(model, X_train, y_train, X_val, y_val)

        # --- 6. Persist ---
        model_path = self.model_dir / "signal_model.joblib"
        model.save(model_path)
        self._last_trained = datetime.now()

        logger.info(f"Training pipeline complete. Model saved to {model_path}")
        return model

    def should_retrain(self, days_since_last: int = 30) -> bool:
        """
        Determine whether the model should be retrained.

        Returns True if no model has been trained yet or if
        `days_since_last` calendar days have elapsed since the last
        training run.

        Args:
            days_since_last: Maximum age (in days) before retraining
                is recommended.

        Returns:
            True if retraining is recommended.
        """
        if self._last_trained is None:
            # Check if a persisted model exists and use its mtime
            model_path = self.model_dir / "signal_model.joblib"
            if model_path.exists():
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                age = (datetime.now() - mtime).days
                return age >= days_since_last
            return True

        age = (datetime.now() - self._last_trained).days
        return age >= days_since_last

    @property
    def last_metrics(self) -> dict:
        """Return metrics from the most recent training run."""
        return self._last_metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_to_period(days: int) -> str:
        """Convert a number of calendar days to a yfinance period string."""
        if days <= 5:
            return "5d"
        elif days <= 30:
            return "1mo"
        elif days <= 90:
            return "3mo"
        elif days <= 180:
            return "6mo"
        elif days <= 365:
            return "1y"
        elif days <= 730:
            return "2y"
        elif days <= 1825:
            return "5y"
        else:
            return "max"

    @staticmethod
    def _rolling_split(
        X: pd.DataFrame,
        y: pd.Series,
        validation_days: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data chronologically: train on older data, validate on
        the most recent `validation_days` rows.

        If the dataset has no DatetimeIndex or contains data from
        multiple symbols interleaved, falls back to a simple tail split
        by position.
        """
        split_point = max(len(X) - validation_days, int(len(X) * 0.7))

        X_train = X.iloc[:split_point]
        X_val = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_val = y.iloc[split_point:]

        logger.info(
            f"Train/val split: {len(X_train)} train, {len(X_val)} validation rows"
        )
        return X_train, X_val, y_train, y_val

    def _evaluate_and_log(
        self,
        model: SignalModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Compute and log classification metrics on train and validation sets."""
        val_proba = model.predict(X_val)
        val_preds = (val_proba >= 0.5).astype(int)

        train_proba = model.predict(X_train)
        train_preds = (train_proba >= 0.5).astype(int)

        metrics = {
            "train_accuracy": accuracy_score(y_train, train_preds),
            "train_precision": precision_score(y_train, train_preds, zero_division=0),
            "train_recall": recall_score(y_train, train_preds, zero_division=0),
            "val_accuracy": accuracy_score(y_val, val_preds),
            "val_precision": precision_score(y_val, val_preds, zero_division=0),
            "val_recall": recall_score(y_val, val_preds, zero_division=0),
            "val_positive_rate": float(y_val.mean()),
            "val_predicted_positive_rate": float(val_preds.mean()),
            "n_train": len(X_train),
            "n_val": len(X_val),
        }

        # Feature importance (top 10)
        importance = model.get_feature_importance()
        top_features = dict(list(importance.items())[:10])
        metrics["top_features"] = top_features

        self._last_metrics = metrics

        logger.info(
            f"Train  -> acc={metrics['train_accuracy']:.3f}  "
            f"prec={metrics['train_precision']:.3f}  "
            f"rec={metrics['train_recall']:.3f}"
        )
        logger.info(
            f"Val    -> acc={metrics['val_accuracy']:.3f}  "
            f"prec={metrics['val_precision']:.3f}  "
            f"rec={metrics['val_recall']:.3f}"
        )
        logger.info(f"Top features: {top_features}")
