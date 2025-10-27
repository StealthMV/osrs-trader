"""
Portfolio Tracker - Track your actual GE positions and profits
"""

import polars as pl
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class GESlot:
    """Represents one Grand Exchange slot"""
    slot_number: int
    item_name: str
    buy_price: int
    quantity: int
    bought_at: datetime
    status: str  # "BUYING", "BOUGHT", "SELLING", "SOLD", "EMPTY"
    sell_price: Optional[int] = None
    
    @property
    def total_cost(self) -> int:
        """Total GP invested"""
        return self.buy_price * self.quantity
    
    @property
    def current_value(self, current_market_price: int) -> int:
        """Current market value"""
        return current_market_price * self.quantity
    
    @property
    def unrealized_profit(self, current_market_price: int) -> int:
        """Profit if sold at current price"""
        return self.current_value(current_market_price) - self.total_cost
    
    @property
    def realized_profit(self) -> Optional[int]:
        """Actual profit from completed sale"""
        if self.status == "SOLD" and self.sell_price:
            return (self.sell_price * self.quantity) - self.total_cost
        return None

@dataclass
class PriceAlert:
    """Price alert configuration"""
    alert_id: str
    item_name: str
    alert_type: str  # "ABOVE", "BELOW", "TARGET_HIT"
    target_price: int
    condition: str  # "BUY", "SELL"
    triggered: bool = False
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class PortfolioTracker:
    """Track your real GE positions and set alerts"""
    
    def __init__(self):
        self.slots: List[GESlot] = []
        self.alerts: List[PriceAlert] = []
    
    def add_position(self, slot_number: int, item_name: str, buy_price: int, quantity: int) -> GESlot:
        """Add a new GE position"""
        slot = GESlot(
            slot_number=slot_number,
            item_name=item_name,
            buy_price=buy_price,
            quantity=quantity,
            bought_at=datetime.now(),
            status="BOUGHT"
        )
        
        # Remove old slot if exists
        self.slots = [s for s in self.slots if s.slot_number != slot_number]
        self.slots.append(slot)
        
        return slot
    
    def update_slot_status(self, slot_number: int, status: str, sell_price: Optional[int] = None):
        """Update the status of a slot"""
        for slot in self.slots:
            if slot.slot_number == slot_number:
                slot.status = status
                if sell_price:
                    slot.sell_price = sell_price
                break
    
    def remove_position(self, slot_number: int):
        """Remove a position (sold or cancelled)"""
        self.slots = [s for s in self.slots if s.slot_number != slot_number]
    
    def get_active_positions(self) -> List[GESlot]:
        """Get all active positions"""
        return [s for s in self.slots if s.status in ["BOUGHT", "SELLING"]]
    
    def get_available_slots(self) -> List[int]:
        """Get available GE slot numbers"""
        used_slots = {s.slot_number for s in self.slots if s.status != "EMPTY"}
        return [i for i in range(1, 9) if i not in used_slots]
    
    def calculate_portfolio_value(self, current_prices: Dict[str, int]) -> Dict:
        """Calculate total portfolio metrics"""
        active = self.get_active_positions()
        
        total_invested = sum(s.total_cost for s in active)
        total_current_value = sum(
            current_prices.get(s.item_name, s.buy_price) * s.quantity 
            for s in active
        )
        
        unrealized_profit = total_current_value - total_invested
        unrealized_roi = (unrealized_profit / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_invested': total_invested,
            'current_value': total_current_value,
            'unrealized_profit': unrealized_profit,
            'unrealized_roi': unrealized_roi,
            'num_positions': len(active),
            'slots_used': len(active),
            'slots_available': len(self.get_available_slots())
        }
    
    def get_position_details(self, current_prices: Dict[str, int]) -> List[Dict]:
        """Get detailed info for each position"""
        details = []
        
        for slot in self.get_active_positions():
            current_price = current_prices.get(slot.item_name, slot.buy_price)
            profit = (current_price - slot.buy_price) * slot.quantity
            roi = (profit / slot.total_cost * 100) if slot.total_cost > 0 else 0
            
            details.append({
                'slot': slot.slot_number,
                'item': slot.item_name,
                'buy_price': slot.buy_price,
                'quantity': slot.quantity,
                'invested': slot.total_cost,
                'current_price': current_price,
                'current_value': current_price * slot.quantity,
                'profit': profit,
                'roi': roi,
                'status': slot.status,
                'held_minutes': int((datetime.now() - slot.bought_at).total_seconds() / 60)
            })
        
        return sorted(details, key=lambda x: x['roi'], reverse=True)
    
    def add_alert(self, item_name: str, target_price: int, condition: str = "SELL") -> PriceAlert:
        """Add a price alert"""
        alert_type = "ABOVE" if condition == "SELL" else "BELOW"
        
        alert = PriceAlert(
            alert_id=f"{item_name}_{target_price}_{condition}",
            item_name=item_name,
            alert_type=alert_type,
            target_price=target_price,
            condition=condition
        )
        
        self.alerts.append(alert)
        return alert
    
    def check_alerts(self, current_prices: Dict[str, int]) -> List[PriceAlert]:
        """Check if any alerts have triggered"""
        triggered = []
        
        for alert in self.alerts:
            if alert.triggered:
                continue
            
            current_price = current_prices.get(alert.item_name)
            if current_price is None:
                continue
            
            # Check if alert condition is met
            if alert.alert_type == "ABOVE" and current_price >= alert.target_price:
                alert.triggered = True
                triggered.append(alert)
            elif alert.alert_type == "BELOW" and current_price <= alert.target_price:
                alert.triggered = True
                triggered.append(alert)
        
        return triggered
    
    def get_active_alerts(self) -> List[PriceAlert]:
        """Get all active (non-triggered) alerts"""
        return [a for a in self.alerts if not a.triggered]
    
    def remove_alert(self, alert_id: str):
        """Remove an alert"""
        self.alerts = [a for a in self.alerts if a.alert_id != alert_id]
    
    def get_sell_recommendations(self, current_prices: Dict[str, int], min_profit_pct: float = 2.0) -> List[Dict]:
        """Get positions that should be sold"""
        recommendations = []
        
        for slot in self.get_active_positions():
            current_price = current_prices.get(slot.item_name, slot.buy_price)
            profit = (current_price - slot.buy_price) * slot.quantity
            roi = (profit / slot.total_cost * 100) if slot.total_cost > 0 else 0
            
            if roi >= min_profit_pct:
                recommendations.append({
                    'slot': slot.slot_number,
                    'item': slot.item_name,
                    'sell_now_price': current_price,
                    'profit': profit,
                    'roi': roi,
                    'reason': f'Hit {roi:.1f}% profit target'
                })
        
        return sorted(recommendations, key=lambda x: x['roi'], reverse=True)
