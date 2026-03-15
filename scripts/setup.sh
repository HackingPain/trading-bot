#!/bin/bash
# Stock Trading Bot - Setup Script
# Run with: ./scripts/setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "     Stock Trading Bot - Setup Script"
echo "=============================================="
echo -e "${NC}"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.10+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt > /dev/null
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data logs results
echo -e "${GREEN}✓ Directories created${NC}"

# Setup environment file
echo -e "${YELLOW}Setting up environment file...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from template${NC}"
    echo -e "${YELLOW}  Please edit .env with your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Validate configuration
echo -e "${YELLOW}Validating configuration...${NC}"
python3 -c "
import yaml
from pathlib import Path

config_path = Path('config/settings.yaml')
if config_path.exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print('✓ Configuration file is valid')
else:
    print('✗ Configuration file not found')
    exit(1)
" 2>/dev/null && echo -e "${GREEN}✓ Configuration valid${NC}" || echo -e "${RED}✗ Configuration error${NC}"

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
if pytest tests/ -q --tb=no > /dev/null 2>&1; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed (run 'pytest tests/ -v' for details)${NC}"
fi

# Check for API keys
echo ""
echo -e "${BLUE}=============================================="
echo "             Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your API keys in .env:"
echo "   - ALPACA_API_KEY (required for trading)"
echo "   - ALPACA_SECRET_KEY (required for trading)"
echo "   - ALPHA_VANTAGE_KEY (optional, for real-time data)"
echo ""
echo "2. Review settings in config/settings.yaml"
echo ""
echo "3. Run the bot in paper mode:"
echo "   source venv/bin/activate"
echo "   python -m src.bot --once  # Test single cycle"
echo "   python -m src.bot         # Start trading"
echo ""
echo "4. Run a backtest:"
echo "   python -m src.cli.backtest -s AAPL MSFT GOOGL -v"
echo ""
echo "5. Start the dashboard:"
echo "   streamlit run src/dashboard/app.py"
echo ""
echo -e "${YELLOW}Remember: Always start with paper_mode: true!${NC}"
