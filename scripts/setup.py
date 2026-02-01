#!/usr/bin/env python3
"""
Cross-platform setup script for the trading bot.

Usage:
    python scripts/setup.py
    python scripts/setup.py --check  # Only check, don't install
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def print_color(text: str, color: str = "white") -> None:
    """Print colored text."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "white": "\033[0m",
    }
    end = "\033[0m"
    print(f"{colors.get(color, '')}{text}{end}")


def check_python_version() -> bool:
    """Check if Python version is 3.10+."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_color(f"✓ Python {version.major}.{version.minor}.{version.micro}", "green")
        return True
    else:
        print_color(f"✗ Python 3.10+ required, found {version.major}.{version.minor}", "red")
        return False


def check_dependencies() -> dict:
    """Check if required dependencies are available."""
    results = {}

    # Check pip
    try:
        import pip
        results["pip"] = True
    except ImportError:
        results["pip"] = False

    # Check if requirements are installed
    try:
        import yaml
        import pandas
        import streamlit
        results["packages"] = True
    except ImportError:
        results["packages"] = False

    return results


def install_dependencies() -> bool:
    """Install dependencies from requirements.txt."""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_path.exists():
        print_color("✗ requirements.txt not found", "red")
        return False

    print_color("Installing dependencies...", "yellow")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path), "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print_color("✓ Dependencies installed", "green")
        return True
    except subprocess.CalledProcessError:
        print_color("✗ Failed to install dependencies", "red")
        return False


def setup_directories() -> None:
    """Create required directories."""
    dirs = ["data", "logs", "results"]
    project_root = Path(__file__).parent.parent

    for d in dirs:
        dir_path = project_root / d
        dir_path.mkdir(exist_ok=True)

    print_color("✓ Directories created", "green")


def setup_env_file() -> bool:
    """Create .env file from template if it doesn't exist."""
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    example_path = project_root / ".env.example"

    if env_path.exists():
        print_color("✓ .env file exists", "green")
        return True

    if example_path.exists():
        shutil.copy(example_path, env_path)
        print_color("✓ Created .env from template", "green")
        print_color("  → Edit .env with your API keys", "yellow")
        return True
    else:
        print_color("✗ .env.example not found", "red")
        return False


def validate_config() -> bool:
    """Validate the configuration file."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "settings.yaml"

    if not config_path.exists():
        print_color("✗ config/settings.yaml not found", "red")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            print_color("✗ Configuration file is empty", "red")
            return False

        # Check required sections
        required_sections = ["trading", "risk"]
        for section in required_sections:
            if section not in config:
                print_color(f"✗ Missing section: {section}", "red")
                return False

        print_color("✓ Configuration valid", "green")
        return True

    except Exception as e:
        print_color(f"✗ Configuration error: {e}", "red")
        return False


def check_api_keys() -> dict:
    """Check for API keys in environment."""
    keys = {
        "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
        "ALPACA_SECRET_KEY": os.getenv("ALPACA_SECRET_KEY"),
        "ALPHA_VANTAGE_KEY": os.getenv("ALPHA_VANTAGE_KEY"),
    }

    # Also check .env file
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    if key in keys and keys[key] is None:
                        # Check if it's not a placeholder
                        if value and "your_" not in value.lower():
                            keys[key] = value

    return keys


def run_tests() -> bool:
    """Run the test suite."""
    print_color("Running tests...", "yellow")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            print_color("✓ All tests passed", "green")
            return True
        else:
            print_color("⚠ Some tests failed", "yellow")
            return False
    except Exception as e:
        print_color(f"⚠ Could not run tests: {e}", "yellow")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup the trading bot")
    parser.add_argument("--check", action="store_true", help="Only check, don't install")
    args = parser.parse_args()

    print_color("\n" + "=" * 50, "blue")
    print_color("    Stock Trading Bot - Setup", "blue")
    print_color("=" * 50 + "\n", "blue")

    # Check Python version
    print_color("Checking Python version...", "yellow")
    if not check_python_version():
        sys.exit(1)

    if not args.check:
        # Install dependencies
        if not install_dependencies():
            sys.exit(1)

        # Setup directories
        setup_directories()

        # Setup .env file
        setup_env_file()

    # Validate configuration
    print_color("Validating configuration...", "yellow")
    validate_config()

    # Check API keys
    print_color("Checking API keys...", "yellow")
    keys = check_api_keys()
    for key, value in keys.items():
        if value and "your_" not in str(value).lower():
            print_color(f"  ✓ {key} configured", "green")
        else:
            status = "required" if "ALPACA" in key else "optional"
            print_color(f"  ○ {key} not set ({status})", "yellow")

    if not args.check:
        # Run tests
        run_tests()

    # Print next steps
    print_color("\n" + "=" * 50, "blue")
    print_color("    Setup Complete!", "blue")
    print_color("=" * 50 + "\n", "blue")

    print("Next steps:")
    print("  1. Edit .env with your Alpaca API keys")
    print("  2. Review config/settings.yaml")
    print("  3. Run: python -m src.bot --once")
    print("  4. Run: python -m src.cli.backtest -s AAPL -v")
    print("")


if __name__ == "__main__":
    main()
