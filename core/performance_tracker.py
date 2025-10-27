"""
Historical Performance Tracker
Logs trades and tracks P&L over time
"""

import polars as pl
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st


class PerformanceTracker:
    """Track trading performance over time"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.trades_file = self.data_dir / "trade_history.json"
        
    def log_trade(self, trade_data: Dict) -> bool:
        """
        Log a new trade
        
        Args:
            trade_data: Dictionary with trade info
                - item_name: str
                - item_id: int
                - buy_price: int
                - sell_price: int (actual, not predicted)
                - quantity: int
                - profit_per_item: int
                - total_profit: int
                - strategy: str
                - predicted_opportunity_score: int
                - predicted_risk_score: int
                - entry_time: datetime
                - exit_time: datetime (optional, for open trades)
                - status: 'OPEN' or 'CLOSED'
        
        Returns:
            True if successful
        """
        try:
            # Load existing trades
            trades = self._load_trades()
            
            # Add new trade
            trade_entry = {
                **trade_data,
                "trade_id": len(trades) + 1,
                "logged_at": datetime.now().isoformat(),
            }
            
            trades.append(trade_entry)
            
            # Save
            self._save_trades(trades)
            return True
            
        except Exception as e:
            st.error(f"Error logging trade: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """
        Calculate overall performance statistics
        
        Returns:
            Dictionary with stats:
            - total_trades: int
            - closed_trades: int
            - open_trades: int
            - total_pnl: int
            - win_rate: float (0-1)
            - avg_profit_per_trade: int
            - best_trade: int
            - worst_trade: int
            - total_roi: float
        """
        trades = self._load_trades()
        
        if not trades:
            return self._empty_stats()
        
        # Filter closed trades for P&L calculations
        closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {
                **self._empty_stats(),
                "total_trades": len(trades),
                "open_trades": len(trades),
            }
        
        # Calculate stats
        profits = [t['total_profit'] for t in closed_trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_capital_used = sum(t['buy_price'] * t['quantity'] for t in closed_trades)
        
        return {
            "total_trades": len(trades),
            "closed_trades": len(closed_trades),
            "open_trades": len(trades) - len(closed_trades),
            "total_pnl": sum(profits),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "avg_profit_per_trade": sum(profits) // len(closed_trades) if closed_trades else 0,
            "best_trade": max(profits) if profits else 0,
            "worst_trade": min(profits) if profits else 0,
            "total_roi": (sum(profits) / total_capital_used * 100) if total_capital_used > 0 else 0,
            "total_wins": len(wins),
            "total_losses": len(losses),
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get most recent trades"""
        trades = self._load_trades()
        return sorted(trades, key=lambda x: x.get('logged_at', ''), reverse=True)[:limit]
    
    def get_best_strategies(self) -> Dict[str, Dict]:
        """
        Analyze which strategies perform best
        
        Returns:
            Dictionary mapping strategy_type to performance stats
        """
        trades = self._load_trades()
        closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
        
        if not closed_trades:
            return {}
        
        # Group by strategy
        strategy_stats = {}
        
        for trade in closed_trades:
            strategy = trade.get('strategy', 'UNKNOWN')
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': [],
                    'total_profit': 0,
                    'win_rate': 0,
                }
            
            strategy_stats[strategy]['trades'].append(trade['total_profit'])
            strategy_stats[strategy]['total_profit'] += trade['total_profit']
        
        # Calculate win rates
        for strategy, stats in strategy_stats.items():
            wins = sum(1 for p in stats['trades'] if p > 0)
            stats['win_rate'] = wins / len(stats['trades']) if stats['trades'] else 0
            stats['avg_profit'] = stats['total_profit'] // len(stats['trades']) if stats['trades'] else 0
            stats['trade_count'] = len(stats['trades'])
        
        return strategy_stats
    
    def _load_trades(self) -> List[Dict]:
        """Load trades from file"""
        if not self.trades_file.exists():
            return []
        
        try:
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_trades(self, trades: List[Dict]):
        """Save trades to file"""
        with open(self.trades_file, 'w') as f:
            json.dump(trades, f, indent=2)
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure"""
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "open_trades": 0,
            "total_pnl": 0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "total_roi": 0.0,
            "total_wins": 0,
            "total_losses": 0,
        }
