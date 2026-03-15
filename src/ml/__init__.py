"""ML signal layer for trading bot.

Provides feature engineering, model training, and prediction
capabilities using scikit-learn classifiers on technical indicator data.

Requires: scikit-learn, joblib (pip install scikit-learn joblib)
"""

try:
    from .features import FeatureEngine
    from .model import SignalModel
    from .trainer import ModelTrainer

    __all__ = [
        "FeatureEngine",
        "SignalModel",
        "ModelTrainer",
    ]
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    __all__ = ["ML_AVAILABLE"]
