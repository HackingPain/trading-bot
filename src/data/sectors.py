"""GICS sector mapping for portfolio risk management (2.5)."""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_sector_lock = threading.Lock()

# Static GICS sector mapping for common US stocks
# This avoids API calls for sector data which can be slow/unreliable
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "GOOG": "Technology", "META": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
    "ADBE": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "AVGO": "Technology", "TXN": "Technology", "QCOM": "Technology",
    "IBM": "Technology", "NOW": "Technology", "AMAT": "Technology",
    "MU": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "MRVL": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "FTNT": "Technology",

    # Communication Services
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CMCSA": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    "ATVI": "Communication Services", "EA": "Communication Services",
    "TTWO": "Communication Services",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",

    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "COST": "Consumer Staples",
    "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "CL": "Consumer Staples",
    "MDLZ": "Consumer Staples", "KHC": "Consumer Staples",
    "STZ": "Consumer Staples", "GIS": "Consumer Staples",
    "KR": "Consumer Staples", "SYY": "Consumer Staples",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "BLK": "Financials", "SCHW": "Financials", "AXP": "Financials",
    "BRK.B": "Financials", "USB": "Financials", "PNC": "Financials",
    "TFC": "Financials", "COF": "Financials", "MET": "Financials",
    "AIG": "Financials", "PRU": "Financials", "ALL": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "SQ": "Financials",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "ISRG": "Healthcare", "CVS": "Healthcare", "CI": "Healthcare",
    "SYK": "Healthcare", "MDT": "Healthcare", "REGN": "Healthcare",
    "VRTX": "Healthcare", "ZTS": "Healthcare", "MRNA": "Healthcare",

    # Industrials
    "BA": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "CAT": "Industrials", "DE": "Industrials", "GE": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "MMM": "Industrials",
    "UNP": "Industrials", "FDX": "Industrials", "WM": "Industrials",
    "EMR": "Industrials", "ETN": "Industrials", "ITW": "Industrials",
    "NSC": "Industrials", "CSX": "Industrials",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy", "MPC": "Energy",
    "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "PXD": "Energy", "DVN": "Energy", "HAL": "Energy",

    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials", "NUE": "Materials",
    "DOW": "Materials", "DD": "Materials", "ECL": "Materials",

    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "PSA": "Real Estate", "SPG": "Real Estate",
    "O": "Real Estate", "DLR": "Real Estate", "WELL": "Real Estate",

    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "SRE": "Utilities",
    "EXC": "Utilities", "XEL": "Utilities", "ED": "Utilities",
    "WEC": "Utilities",
}


def get_sector(symbol: str) -> Optional[str]:
    """Get GICS sector for a symbol from static mapping."""
    return SECTOR_MAP.get(symbol.upper())


def get_sector_dynamic(symbol: str) -> Optional[str]:
    """Get sector dynamically via yfinance (slower, used as fallback)."""
    # Check static map first
    sector = SECTOR_MAP.get(symbol.upper())
    if sector:
        return sector

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get("sector")
        if sector:
            # Cache for future lookups (thread-safe, Fix #12)
            with _sector_lock:
                SECTOR_MAP[symbol.upper()] = sector
            return sector
    except Exception as e:
        logger.debug(f"Could not fetch sector for {symbol}: {e}")

    return None


def get_sectors_for_symbols(symbols: list[str]) -> dict[str, str]:
    """Get sectors for multiple symbols."""
    result = {}
    for symbol in symbols:
        sector = get_sector(symbol)
        if sector:
            result[symbol] = sector
    return result


def calculate_sector_exposure(
    positions: list[dict],
    equity: float,
) -> dict[str, float]:
    """
    Calculate sector exposure as percentage of equity.

    Args:
        positions: List of dicts with 'symbol' and 'market_value' keys
        equity: Total account equity

    Returns:
        Dict mapping sector name to exposure percentage
    """
    if equity <= 0:
        return {}

    sector_values: dict[str, float] = {}
    for pos in positions:
        symbol = pos.get("symbol", "")
        market_value = pos.get("market_value", 0)
        sector = get_sector(symbol) or "Unknown"
        sector_values[sector] = sector_values.get(sector, 0) + market_value

    return {
        sector: (value / equity) * 100
        for sector, value in sector_values.items()
    }
