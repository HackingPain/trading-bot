# Stock Trading Bot - Makefile
# Convenience commands for development and deployment

.PHONY: help install dev test lint build run stop logs clean backtest

# Default target
help:
	@echo "Stock Trading Bot - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Run bot in development mode"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo ""
	@echo "Docker:"
	@echo "  make build      - Build Docker images"
	@echo "  make run        - Start all services"
	@echo "  make stop       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Remove containers and images"
	@echo ""
	@echo "Trading:"
	@echo "  make backtest   - Run backtest"
	@echo "  make dashboard  - Start dashboard only"

# Development commands
install:
	pip install -r requirements.txt

dev:
	python -m src.bot --config config/settings.yaml

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	mypy src/
	ruff check src/

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
	rm -rf data/*.db logs/*.log

# Trading commands
backtest:
	python -c "from src.backtest import run_backtest; from src.strategies.daily_profit_taker import DailyProfitTakerStrategy; from datetime import datetime, timedelta; \
	strategy = DailyProfitTakerStrategy.from_settings({}); \
	result = run_backtest(strategy, ['AAPL', 'MSFT', 'GOOGL'], datetime.now() - timedelta(days=365), datetime.now()); \
	print(result.summary())"

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
