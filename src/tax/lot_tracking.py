"""
Tax lot tracking for trades.

Tracks cost basis, wash sales, and generates tax reports.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CostBasisMethod(str, Enum):
    """Method for selecting which lots to sell."""
    FIFO = "fifo"  # First In, First Out
    LIFO = "lifo"  # Last In, First Out
    HIFO = "hifo"  # Highest In, First Out (minimize gains)
    LIFO_MIN_TAX = "min_tax"  # Minimize taxable gains
    SPECIFIC = "specific"  # Specific lot identification


class HoldingPeriod(str, Enum):
    """Tax holding period classification."""
    SHORT_TERM = "short_term"  # < 1 year
    LONG_TERM = "long_term"  # >= 1 year


@dataclass
class TaxLot:
    """A single tax lot representing shares purchased at a specific time and price."""
    lot_id: str
    symbol: str
    quantity: float
    purchase_price: float
    purchase_date: datetime
    cost_basis: float  # quantity * purchase_price + fees
    fees: float = 0.0
    remaining_quantity: float = 0.0
    is_wash_sale: bool = False
    wash_sale_adjustment: float = 0.0

    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

    @property
    def holding_period(self) -> HoldingPeriod:
        """Determine if lot is short-term or long-term."""
        days_held = (datetime.now() - self.purchase_date).days
        if days_held >= 365:
            return HoldingPeriod.LONG_TERM
        return HoldingPeriod.SHORT_TERM

    @property
    def days_held(self) -> int:
        """Number of days the lot has been held."""
        return (datetime.now() - self.purchase_date).days

    @property
    def adjusted_cost_basis(self) -> float:
        """Cost basis adjusted for wash sales."""
        return self.cost_basis + self.wash_sale_adjustment


@dataclass
class SaleRecord:
    """Record of a sale for tax reporting."""
    symbol: str
    quantity: float
    sale_price: float
    sale_date: datetime
    proceeds: float
    lots_sold: list[dict] = field(default_factory=list)  # {lot_id, quantity, cost_basis, gain_loss}
    total_cost_basis: float = 0.0
    total_gain_loss: float = 0.0
    short_term_gain: float = 0.0
    long_term_gain: float = 0.0
    is_wash_sale: bool = False
    wash_sale_disallowed: float = 0.0
    fees: float = 0.0


@dataclass
class WashSaleInfo:
    """Information about a wash sale."""
    original_sale_date: datetime
    loss_amount: float
    replacement_lot_id: str
    adjustment_amount: float


class TaxLotTracker:
    """
    Tracks tax lots for trades and calculates gains/losses.

    Features:
    - Multiple cost basis methods (FIFO, LIFO, HIFO, etc.)
    - Wash sale detection and adjustment
    - Short-term vs long-term classification
    - Tax report generation
    """

    def __init__(
        self,
        cost_basis_method: CostBasisMethod = CostBasisMethod.FIFO,
        wash_sale_window_days: int = 30,
    ):
        """
        Initialize tax lot tracker.

        Args:
            cost_basis_method: Method for selecting lots to sell
            wash_sale_window_days: Days before/after sale to check for wash sale
        """
        self.cost_basis_method = cost_basis_method
        self.wash_sale_window_days = wash_sale_window_days

        # Storage
        self._lots: dict[str, list[TaxLot]] = {}  # symbol -> list of lots
        self._sales: list[SaleRecord] = []
        self._lot_counter = 0

    def _generate_lot_id(self) -> str:
        """Generate unique lot ID."""
        self._lot_counter += 1
        return f"LOT-{self._lot_counter:06d}"

    def add_purchase(
        self,
        symbol: str,
        quantity: float,
        price: float,
        purchase_date: Optional[datetime] = None,
        fees: float = 0.0,
    ) -> TaxLot:
        """
        Record a purchase as a new tax lot.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            purchase_date: Date of purchase (default: now)
            fees: Transaction fees

        Returns:
            Created TaxLot
        """
        symbol = symbol.upper()
        if purchase_date is None:
            purchase_date = datetime.now()

        lot = TaxLot(
            lot_id=self._generate_lot_id(),
            symbol=symbol,
            quantity=quantity,
            purchase_price=price,
            purchase_date=purchase_date,
            cost_basis=(quantity * price) + fees,
            fees=fees,
        )

        # Check for wash sale from prior losses
        self._check_and_apply_wash_sale(lot)

        if symbol not in self._lots:
            self._lots[symbol] = []
        self._lots[symbol].append(lot)

        logger.info(f"Added tax lot {lot.lot_id}: {quantity} {symbol} @ ${price:.2f}")

        return lot

    def record_sale(
        self,
        symbol: str,
        quantity: float,
        price: float,
        sale_date: Optional[datetime] = None,
        fees: float = 0.0,
        specific_lot_id: Optional[str] = None,
    ) -> SaleRecord:
        """
        Record a sale and calculate gains/losses.

        Args:
            symbol: Stock symbol
            quantity: Number of shares sold
            price: Sale price per share
            sale_date: Date of sale (default: now)
            fees: Transaction fees
            specific_lot_id: Specific lot to sell (for SPECIFIC method)

        Returns:
            SaleRecord with gain/loss details
        """
        symbol = symbol.upper()
        if sale_date is None:
            sale_date = datetime.now()

        if symbol not in self._lots or not self._lots[symbol]:
            raise ValueError(f"No lots found for {symbol}")

        proceeds = (quantity * price) - fees

        # Select lots to sell based on cost basis method
        lots_to_sell = self._select_lots(
            symbol,
            quantity,
            specific_lot_id,
        )

        sale_record = SaleRecord(
            symbol=symbol,
            quantity=quantity,
            sale_price=price,
            sale_date=sale_date,
            proceeds=proceeds,
            fees=fees,
        )

        remaining_qty = quantity

        for lot in lots_to_sell:
            if remaining_qty <= 0:
                break

            # How much from this lot?
            lot_qty = min(remaining_qty, lot.remaining_quantity)
            if lot_qty <= 0:
                continue

            # Calculate proportional cost basis
            lot_cost_per_share = lot.adjusted_cost_basis / lot.quantity
            lot_cost = lot_qty * lot_cost_per_share

            # Calculate gain/loss
            lot_proceeds = lot_qty * price
            gain_loss = lot_proceeds - lot_cost

            # Classify as short or long term
            if lot.holding_period == HoldingPeriod.SHORT_TERM:
                sale_record.short_term_gain += gain_loss
            else:
                sale_record.long_term_gain += gain_loss

            # Record lot sale
            sale_record.lots_sold.append({
                "lot_id": lot.lot_id,
                "quantity": lot_qty,
                "cost_basis": lot_cost,
                "gain_loss": gain_loss,
                "holding_period": lot.holding_period.value,
                "purchase_date": lot.purchase_date.isoformat(),
            })

            sale_record.total_cost_basis += lot_cost
            sale_record.total_gain_loss += gain_loss

            # Update lot
            lot.remaining_quantity -= lot_qty
            remaining_qty -= lot_qty

            logger.debug(
                f"Sold {lot_qty} from {lot.lot_id}: gain/loss ${gain_loss:.2f} "
                f"({lot.holding_period.value})"
            )

        # Check for wash sale
        if sale_record.total_gain_loss < 0:
            wash_sale_info = self._check_wash_sale(symbol, sale_date, sale_record.total_gain_loss)
            if wash_sale_info:
                sale_record.is_wash_sale = True
                sale_record.wash_sale_disallowed = wash_sale_info.adjustment_amount

        # Clean up empty lots
        self._lots[symbol] = [lot for lot in self._lots[symbol] if lot.remaining_quantity > 0]

        self._sales.append(sale_record)

        logger.info(
            f"Recorded sale: {quantity} {symbol} @ ${price:.2f}, "
            f"gain/loss: ${sale_record.total_gain_loss:.2f}"
        )

        return sale_record

    def _select_lots(
        self,
        symbol: str,
        quantity: float,
        specific_lot_id: Optional[str] = None,
    ) -> list[TaxLot]:
        """Select which lots to sell based on cost basis method."""
        lots = [lot for lot in self._lots.get(symbol, []) if lot.remaining_quantity > 0]

        if not lots:
            return []

        if specific_lot_id:
            # Specific lot identification
            return [lot for lot in lots if lot.lot_id == specific_lot_id]

        if self.cost_basis_method == CostBasisMethod.FIFO:
            # First In, First Out - sort by purchase date ascending
            return sorted(lots, key=lambda l: l.purchase_date)

        elif self.cost_basis_method == CostBasisMethod.LIFO:
            # Last In, First Out - sort by purchase date descending
            return sorted(lots, key=lambda l: l.purchase_date, reverse=True)

        elif self.cost_basis_method == CostBasisMethod.HIFO:
            # Highest In, First Out - sort by cost per share descending
            return sorted(
                lots,
                key=lambda l: l.adjusted_cost_basis / l.quantity,
                reverse=True,
            )

        elif self.cost_basis_method == CostBasisMethod.LIFO_MIN_TAX:
            # Minimize tax - prefer lots with losses, then long-term gains, then short-term
            def tax_priority(lot: TaxLot) -> tuple:
                cost_per_share = lot.adjusted_cost_basis / lot.quantity
                is_loss = cost_per_share > 0  # Higher cost = potential loss
                is_long_term = lot.holding_period == HoldingPeriod.LONG_TERM
                return (not is_loss, not is_long_term, -cost_per_share)

            return sorted(lots, key=tax_priority)

        return lots

    def _check_wash_sale(
        self,
        symbol: str,
        sale_date: datetime,
        loss_amount: float,
    ) -> Optional[WashSaleInfo]:
        """Check if a loss triggers wash sale rules."""
        if loss_amount >= 0:
            return None  # Only losses can be wash sales

        window_start = sale_date - timedelta(days=self.wash_sale_window_days)
        window_end = sale_date + timedelta(days=self.wash_sale_window_days)

        # Check for replacement purchases
        lots = self._lots.get(symbol, [])
        for lot in lots:
            if window_start <= lot.purchase_date <= window_end:
                if lot.purchase_date != sale_date:  # Not the same transaction
                    # This is a wash sale
                    adjustment = min(abs(loss_amount), lot.cost_basis)
                    lot.is_wash_sale = True
                    lot.wash_sale_adjustment += adjustment

                    logger.warning(
                        f"Wash sale detected for {symbol}: ${adjustment:.2f} loss disallowed"
                    )

                    return WashSaleInfo(
                        original_sale_date=sale_date,
                        loss_amount=loss_amount,
                        replacement_lot_id=lot.lot_id,
                        adjustment_amount=adjustment,
                    )

        return None

    def _check_and_apply_wash_sale(self, new_lot: TaxLot) -> None:
        """Check if a new purchase triggers wash sale from recent loss."""
        window_start = new_lot.purchase_date - timedelta(days=self.wash_sale_window_days)

        # Check recent sales for losses
        for sale in self._sales:
            if sale.symbol != new_lot.symbol:
                continue
            if sale.total_gain_loss >= 0:
                continue  # Only losses
            if not (window_start <= sale.sale_date <= new_lot.purchase_date):
                continue

            # Apply wash sale adjustment
            adjustment = min(abs(sale.total_gain_loss), new_lot.cost_basis)
            new_lot.is_wash_sale = True
            new_lot.wash_sale_adjustment += adjustment

            logger.warning(
                f"Wash sale: ${adjustment:.2f} added to cost basis of {new_lot.lot_id}"
            )

    def get_lots(self, symbol: Optional[str] = None) -> list[TaxLot]:
        """Get all lots, optionally filtered by symbol."""
        if symbol:
            return self._lots.get(symbol.upper(), [])

        all_lots = []
        for lots in self._lots.values():
            all_lots.extend(lots)
        return all_lots

    def get_unrealized_gains(self, current_prices: dict[str, float]) -> dict[str, Any]:
        """
        Calculate unrealized gains/losses for all lots.

        Args:
            current_prices: Dict mapping symbol to current price

        Returns:
            Dict with unrealized gain details
        """
        unrealized = {
            "total": 0.0,
            "short_term": 0.0,
            "long_term": 0.0,
            "by_symbol": {},
        }

        for symbol, lots in self._lots.items():
            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            symbol_data = {
                "current_price": price,
                "total_shares": 0,
                "total_cost": 0,
                "unrealized_gain": 0,
                "short_term_gain": 0,
                "long_term_gain": 0,
            }

            for lot in lots:
                if lot.remaining_quantity <= 0:
                    continue

                market_value = lot.remaining_quantity * price
                lot_cost = (lot.remaining_quantity / lot.quantity) * lot.adjusted_cost_basis
                gain = market_value - lot_cost

                symbol_data["total_shares"] += lot.remaining_quantity
                symbol_data["total_cost"] += lot_cost
                symbol_data["unrealized_gain"] += gain

                if lot.holding_period == HoldingPeriod.SHORT_TERM:
                    symbol_data["short_term_gain"] += gain
                    unrealized["short_term"] += gain
                else:
                    symbol_data["long_term_gain"] += gain
                    unrealized["long_term"] += gain

            unrealized["total"] += symbol_data["unrealized_gain"]
            unrealized["by_symbol"][symbol] = symbol_data

        return unrealized

    def get_realized_gains(
        self,
        year: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Get realized gains/losses for tax reporting.

        Args:
            year: Optional year filter (default: current year)

        Returns:
            Dict with realized gain details
        """
        if year is None:
            year = datetime.now().year

        realized = {
            "year": year,
            "total": 0.0,
            "short_term": 0.0,
            "long_term": 0.0,
            "wash_sales_disallowed": 0.0,
            "sales_count": 0,
            "sales": [],
        }

        for sale in self._sales:
            if sale.sale_date.year != year:
                continue

            realized["total"] += sale.total_gain_loss
            realized["short_term"] += sale.short_term_gain
            realized["long_term"] += sale.long_term_gain
            realized["wash_sales_disallowed"] += sale.wash_sale_disallowed
            realized["sales_count"] += 1

            realized["sales"].append({
                "symbol": sale.symbol,
                "sale_date": sale.sale_date.isoformat(),
                "quantity": sale.quantity,
                "proceeds": sale.proceeds,
                "cost_basis": sale.total_cost_basis,
                "gain_loss": sale.total_gain_loss,
                "short_term": sale.short_term_gain,
                "long_term": sale.long_term_gain,
                "wash_sale": sale.is_wash_sale,
            })

        return realized

    def generate_tax_report(self, year: Optional[int] = None) -> str:
        """Generate a text-based tax report."""
        realized = self.get_realized_gains(year)

        lines = [
            f"TAX REPORT - {realized['year']}",
            "=" * 50,
            "",
            "SUMMARY",
            "-" * 30,
            f"Total Realized Gain/Loss: ${realized['total']:,.2f}",
            f"  Short-Term: ${realized['short_term']:,.2f}",
            f"  Long-Term:  ${realized['long_term']:,.2f}",
            f"Wash Sales Disallowed: ${realized['wash_sales_disallowed']:,.2f}",
            f"Net Taxable: ${realized['total'] + realized['wash_sales_disallowed']:,.2f}",
            f"Total Sales: {realized['sales_count']}",
            "",
            "TRANSACTIONS",
            "-" * 30,
        ]

        for sale in realized["sales"]:
            status = " [WASH]" if sale["wash_sale"] else ""
            term = "ST" if sale["short_term"] != 0 else "LT"
            lines.append(
                f"{sale['sale_date'][:10]} {sale['symbol']:6} "
                f"Qty:{sale['quantity']:>6.0f}  "
                f"Proceeds:${sale['proceeds']:>10,.2f}  "
                f"Basis:${sale['cost_basis']:>10,.2f}  "
                f"G/L:${sale['gain_loss']:>+10,.2f} ({term}){status}"
            )

        return "\n".join(lines)

    def export_for_tax_software(self, year: Optional[int] = None) -> list[dict]:
        """Export sales in format compatible with tax software."""
        realized = self.get_realized_gains(year)
        return realized["sales"]
