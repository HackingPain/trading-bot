# Stock Trading Bot - Makefile
# Convenience commands for development and deployment

.PHONY: help install dev test lint build run stop logs clean backtest validate

# Default target
help:
	@echo "Stock Trading Bot - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Development:"
	@echo "  make install        - Install dependencies"
	@echo "  make dev            - Run bot in development mode"
	@echo "  make dev-sim        - Run bot with simulated broker (no API keys)"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run linters (ruff + mypy)"
	@echo "  make validate       - Validate config file"
	@echo ""
	@echo "Docker:"
	@echo "  make build          - Build Docker images"
	@echo "  make run            - Start all services"
	@echo "  make stop           - Stop all services"
	@echo "  make logs           - View logs"
	@echo "  make clean          - Remove containers and images"
	@echo ""
	@echo "Trading:"
	@echo "  make backtest       - Run backtest with default strategy"
	@echo "  make walk-forward   - Run walk-forward analysis"
	@echo "  make dashboard      - Start dashboard only"
	@echo "  make train-ml       - Train ML signal model"

# Development commands
install:
	pip install -r requirements.txt

dev:
	python -m src.bot --config config/settings.yaml

dev-sim:
	python -c "\
	import yaml; \
	c = yaml.safe_load(open('config/settings.yaml')); \
	c.setdefault('broker', {})['type'] = 'simulated'; \
	yaml.dump(c, open('config/settings_sim.yaml', 'w')); \
	" && python -m src.bot --config config/settings_sim.yaml

dev-once:
	python -m src.bot --config config/settings.yaml --once

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	ruff check src/
	mypy src/ --ignore-missing-imports

validate:
	python -m src.bot --config config/settings.yaml --validate-config

# Docker commands
build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

logs:
	docker compose logs -f

logs-bot:
	docker compose logs -f trading-bot

logs-dashboard:
	docker compose logs -f dashboard

clean:
	docker compose down -v --rmi local
	rm -rf data/*.db logs/*.log logs/*.jsonl

# Trading commands
backtest:
	python -c "\
	from src.backtest.engine import run_backtest; \
	from src.strategies.daily_profit_taker import DailyProfitTakerStrategy; \
	from datetime import datetime, timedelta; \
	strategy = DailyProfitTakerStrategy.from_settings({}); \
	result = run_backtest(strategy, ['AAPL', 'MSFT', 'GOOGL'], \
	    datetime.now() - timedelta(days=365), datetime.now()); \
	print(result.summary())"

walk-forward:
	python -c "\
	from src.backtest.walk_forward import WalkForwardAnalyzer; \
	from src.backtest.engine import BacktestConfig; \
	from src.strategies.daily_profit_taker import DailyProfitTakerStrategy; \
	from src.data.market_data import MarketDataProvider, MarketDataConfig; \
	from datetime import datetime, timedelta; \
	strategy = DailyProfitTakerStrategy.from_settings({}); \
	dp = MarketDataProvider(MarketDataConfig()); \
	wfa = WalkForwardAnalyzer(strategy, dp, BacktestConfig()); \
	result = wfa.run(['AAPL', 'MSFT'], \
	    datetime.now() - timedelta(days=365), datetime.now()); \
	print(f'Folds: {len(result.folds)}'); \
	print(f'OOS return: {result.aggregated_oos_metrics.total_return_pct:.2f}%'); \
	print(f'Overfitting ratio: {result.overfitting_ratio:.2f}')"

train-ml:
	python -c "\
	from src.ml import ML_AVAILABLE; \
	assert ML_AVAILABLE, 'Install scikit-learn: pip install scikit-learn joblib'; \
	from src.ml.trainer import ModelTrainer; \
	from src.data.market_data import MarketDataProvider, MarketDataConfig; \
	dp = MarketDataProvider(MarketDataConfig()); \
	trainer = ModelTrainer(dp); \
	trainer.train_on_history(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']); \
	print('Model trained successfully'); \
	print(f'Metrics: {trainer.last_metrics}')"

dashboard:
	streamlit run src/dashboard/app.py

# Utility commands
shell:
	docker compose exec trading-bot /bin/bash

check-env:
	@echo "Checking environment variables..."
	@test -f .env || (echo "ERROR: .env file not found. Copy .env.example to .env and configure." && exit 1)
	@grep -q "ALPACA_API_KEY=your" .env && echo "WARNING: ALPACA_API_KEY not configured" || echo "OK: ALPACA_API_KEY set"
	@grep -q "ALPACA_SECRET_KEY=your" .env && echo "WARNING: ALPACA_SECRET_KEY not configured" || echo "OK: ALPACA_SECRET_KEY set"
