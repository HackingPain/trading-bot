"""Configuration validation for the trading bot."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed: {'; '.join(errors)}")


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# Configuration schema with validation rules
CONFIG_SCHEMA = {
    "trading": {
        "paper_mode": {"type": bool, "required": True, "default": True},
        "symbols": {"type": list, "required": True, "min_length": 1},
        "check_interval_seconds": {"type": int, "required": False, "default": 60, "min": 10, "max": 3600},
        "timezone": {"type": str, "required": False, "default": "America/New_York"},
    },
    "risk": {
        "max_position_pct": {"type": float, "required": False, "default": 0.10, "min": 0.01, "max": 1.0},
        "max_portfolio_risk_pct": {"type": float, "required": False, "default": 0.30, "min": 0.01, "max": 1.0},
        "max_daily_loss_pct": {"type": float, "required": False, "default": 0.02, "min": 0.001, "max": 0.20},
        "max_daily_trades": {"type": int, "required": False, "default": 3, "min": 1, "max": 100},
        "stop_loss_pct": {"type": float, "required": False, "default": 0.05, "min": 0.01, "max": 0.50},
        "trailing_stop_pct": {"type": float, "required": False, "default": 0.03, "min": 0.01, "max": 0.50},
        "min_account_balance": {"type": float, "required": False, "default": 1000, "min": 0},
    },
    "strategy": {
        "name": {"type": str, "required": False, "default": "daily_profit_taker"},
        "profit_target_pct": {"type": float, "required": False, "default": 0.02, "min": 0.001, "max": 0.50},
        "use_trailing_stop": {"type": bool, "required": False, "default": True},
        "rsi_oversold": {"type": int, "required": False, "default": 30, "min": 1, "max": 49},
        "rsi_overbought": {"type": int, "required": False, "default": 70, "min": 51, "max": 99},
    },
    "api": {
        "alpaca": {
            "key": {"type": str, "required": False, "env_var": "ALPACA_API_KEY"},
            "secret": {"type": str, "required": False, "env_var": "ALPACA_SECRET_KEY"},
            "base_url": {"type": str, "required": False, "default": "https://paper-api.alpaca.markets"},
        },
        "alpha_vantage": {
            "key": {"type": str, "required": False, "env_var": "ALPHA_VANTAGE_KEY"},
            "rate_limit_per_minute": {"type": int, "required": False, "default": 5, "min": 1, "max": 500},
        },
    },
    "notifications": {
        "enabled": {"type": bool, "required": False, "default": True},
    },
    "database": {
        "url": {"type": str, "required": False, "default": "sqlite:///data/trading_bot.db"},
    },
    "logging": {
        "level": {"type": str, "required": False, "default": "INFO", "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]},
        "file": {"type": str, "required": False, "default": "logs/trading_bot.log"},
    },
}


class ConfigValidator:
    """Validates trading bot configuration."""

    def __init__(self, schema: dict = None):
        self.schema = schema or CONFIG_SCHEMA

    def validate(self, config: dict) -> ValidationResult:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        self._validate_section(config, self.schema, "", errors, warnings)

        # Additional cross-field validations
        self._validate_cross_fields(config, errors, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_section(
        self,
        config: dict,
        schema: dict,
        path: str,
        errors: list,
        warnings: list,
    ) -> None:
        """Recursively validate a configuration section."""
        for key, rules in schema.items():
            full_path = f"{path}.{key}" if path else key
            value = config.get(key) if config else None

            # Check if this is a nested section
            if isinstance(rules, dict) and "type" not in rules:
                self._validate_section(
                    config.get(key, {}) if config else {},
                    rules,
                    full_path,
                    errors,
                    warnings,
                )
                continue

            # Validate field
            self._validate_field(full_path, value, rules, errors, warnings)

    def _validate_field(
        self,
        path: str,
        value: Any,
        rules: dict,
        errors: list,
        warnings: list,
    ) -> None:
        """Validate a single field against its rules."""
        # Check environment variable fallback
        if value is None or value == "":
            env_var = rules.get("env_var")
            if env_var:
                value = os.getenv(env_var)

        # Check required
        if rules.get("required") and (value is None or value == ""):
            errors.append(f"{path}: Required field is missing")
            return

        # If value is None and not required, skip further validation
        if value is None:
            return

        # Check type
        expected_type = rules.get("type")
        if expected_type:
            if expected_type == float and isinstance(value, int):
                value = float(value)  # Allow int for float fields
            elif not isinstance(value, expected_type):
                errors.append(
                    f"{path}: Expected {expected_type.__name__}, got {type(value).__name__}"
                )
                return

        # Check min/max for numbers
        if isinstance(value, (int, float)):
            if "min" in rules and value < rules["min"]:
                errors.append(f"{path}: Value {value} is below minimum {rules['min']}")
            if "max" in rules and value > rules["max"]:
                errors.append(f"{path}: Value {value} exceeds maximum {rules['max']}")

        # Check min_length for lists
        if isinstance(value, list) and "min_length" in rules:
            if len(value) < rules["min_length"]:
                errors.append(
                    f"{path}: List must have at least {rules['min_length']} items"
                )

        # Check choices
        if "choices" in rules and value not in rules["choices"]:
            errors.append(f"{path}: Value must be one of {rules['choices']}")

    def _validate_cross_fields(
        self,
        config: dict,
        errors: list,
        warnings: list,
    ) -> None:
        """Validate relationships between fields."""
        # Check RSI oversold < overbought
        strategy = config.get("strategy", {})
        rsi_oversold = strategy.get("rsi_oversold", 30)
        rsi_overbought = strategy.get("rsi_overbought", 70)
        if rsi_oversold >= rsi_overbought:
            errors.append(
                f"strategy.rsi_oversold ({rsi_oversold}) must be less than "
                f"strategy.rsi_overbought ({rsi_overbought})"
            )

        # Check trailing stop <= stop loss
        risk = config.get("risk", {})
        stop_loss = risk.get("stop_loss_pct", 0.05)
        trailing_stop = risk.get("trailing_stop_pct", 0.03)
        if trailing_stop > stop_loss:
            warnings.append(
                f"risk.trailing_stop_pct ({trailing_stop}) is greater than "
                f"risk.stop_loss_pct ({stop_loss}), trailing stop may trigger before stop loss"
            )

        # Warn if live trading without proper setup
        trading = config.get("trading", {})
        if not trading.get("paper_mode", True):
            warnings.append("LIVE TRADING MODE ENABLED - Ensure all settings are correct!")

            # Check API credentials for live trading
            api = config.get("api", {}).get("alpaca", {})
            api_key = api.get("key") or os.getenv("ALPACA_API_KEY")
            api_secret = api.get("secret") or os.getenv("ALPACA_SECRET_KEY")

            if not api_key or not api_secret:
                errors.append("API credentials required for live trading")

    def apply_defaults(self, config: dict) -> dict:
        """Apply default values to missing configuration fields."""
        return self._apply_defaults_section(config, self.schema)

    def _apply_defaults_section(self, config: dict, schema: dict) -> dict:
        """Recursively apply defaults to a section."""
        result = dict(config) if config else {}

        for key, rules in schema.items():
            # Check if nested section
            if isinstance(rules, dict) and "type" not in rules:
                result[key] = self._apply_defaults_section(
                    result.get(key, {}),
                    rules,
                )
            elif "default" in rules and key not in result:
                result[key] = rules["default"]

        return result


def validate_config(config_path: str = "config/settings.yaml") -> ValidationResult:
    """
    Validate configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        ValidationResult

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    validator = ConfigValidator()
    return validator.validate(config)


def load_and_validate_config(config_path: str = "config/settings.yaml") -> dict:
    """
    Load and validate configuration, raising on errors.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated configuration dict with defaults applied

    Raises:
        ConfigValidationError: If validation fails
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    validator = ConfigValidator()
    result = validator.validate(config)

    if not result.valid:
        raise ConfigValidationError(result.errors)

    # Log warnings
    for warning in result.warnings:
        logger.warning(f"Config warning: {warning}")

    # Apply defaults and return
    return validator.apply_defaults(config)
