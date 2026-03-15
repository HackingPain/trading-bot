"""ML model wrapper for trading signal prediction.

Provides a thin abstraction over scikit-learn classifiers with
train / predict / persist lifecycle methods.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


class SignalModel:
    """
    Wraps a scikit-learn classifier for trading signal prediction.

    Default classifier is GradientBoostingClassifier, which is lightweight
    and does not require GPU. Any sklearn classifier supporting
    `predict_proba` can be substituted.

    Args:
        classifier: A fitted or unfitted sklearn classifier. If None,
            a GradientBoostingClassifier with sensible defaults is created.
        feature_names: Ordered list of feature column names the model
            was trained on. Populated automatically during `train()`.
    """

    def __init__(
        self,
        classifier: Optional[ClassifierMixin] = None,
        feature_names: Optional[list[str]] = None,
    ):
        self.classifier: ClassifierMixin = classifier or GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=42,
        )
        self.feature_names: list[str] = feature_names or []
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> "SignalModel":
        """
        Fit the classifier on training data.

        Args:
            X: Feature matrix (rows = samples, columns = features).
            y: Binary target series (0 or 1).

        Returns:
            self, for method chaining.
        """
        self.feature_names = list(X.columns)

        logger.info(
            f"Training model on {len(X)} samples with {len(self.feature_names)} features"
        )
        self.classifier.fit(X, y)
        self._is_trained = True

        logger.info("Model training complete")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability scores for the positive class (1).

        Args:
            X: Feature matrix with the same columns used during training.

        Returns:
            1-D numpy array of probabilities in [0, 1].

        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() or load a saved model first.")

        # Ensure column order matches training
        X_aligned = X[self.feature_names] if self.feature_names else X

        probabilities = self.classifier.predict_proba(X_aligned)

        # predict_proba returns shape (n, 2) for binary classification
        # Column index 1 is the positive class probability
        return probabilities[:, 1]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return feature importances as a name -> importance dict.

        For tree-based models this uses the built-in `feature_importances_`.
        Falls back to uniform weights if the classifier does not expose
        importances.

        Returns:
            Dictionary mapping feature name to importance score, sorted
            descending by importance.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained.")

        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
        else:
            logger.warning(
                "Classifier does not expose feature_importances_; "
                "returning uniform weights."
            )
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        imp_dict = dict(zip(self.feature_names, importances))
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Persist the trained model to disk using joblib.

        Saves the classifier, feature names, and training flag as a
        single dictionary.

        Args:
            path: File path to write (e.g. 'models/signal_model.joblib').
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "is_trained": self._is_trained,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_file(cls, path: str | Path) -> "SignalModel":
        """
        Load a trained model from disk.

        Args:
            path: Path to a joblib file previously saved by `save()`.

        Returns:
            A SignalModel instance ready for prediction.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)

        model = cls(
            classifier=payload["classifier"],
            feature_names=payload["feature_names"],
        )
        model._is_trained = payload.get("is_trained", True)

        logger.info(
            f"Model loaded from {path} "
            f"({len(model.feature_names)} features)"
        )
        return model

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        clf_name = type(self.classifier).__name__
        return f"SignalModel(classifier={clf_name}, features={len(self.feature_names)}, {status})"
