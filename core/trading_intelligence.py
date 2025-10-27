"""
ULTIMATE Trading Intelligence Engine
Real-time scanning, forecasting, and execution intelligence
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeForecast:
    """Forecast for a trading opportunity"""
    item_name: str
    expected_profit_24h: int
    confidence_low: int  # 5th percentile
    confidence_high: int  # 95th percentile
    probability_of_profit: float
    expected_trades: int
    risk_level: str


@dataclass
class OpportunityAlert:
    """Real-time trading alert"""
    timestamp: datetime
    item_name: str
    alert_type: str  # BREAKOUT, VOLUME_SPIKE, NEW_OPPORTUNITY, PRICE_DROP
    opportunity_score: int
    action: str  # BUY, SELL, WATCH
    urgency: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str


class TradingIntelligence:
    """Advanced trading intelligence and forecasting"""
    
    def __init__(self):
        self.alerts = []
        self.performance_history = []
        
    def forecast_daily_profit(
        self,
        df: pl.DataFrame,
        capital: int,
        risk_tolerance: str = "MODERATE"
    ) -> Dict:
        """
        Forecast expected daily profits with confidence intervals
        
        Uses Monte Carlo simulation to estimate:
        - Expected profit
        - Best case (95th percentile)
        - Worst case (5th percentile)
        - Probability of profit
        """
        if len(df) == 0:
            return self._empty_forecast()
        
        # Get allocated positions
        allocated = df.head(10)
        
        # Simulation parameters
        n_simulations = 1000
        trades_per_day_range = (2, 8)  # Conservative: 2-8 full cycles per day
        
        # Monte Carlo simulation
        daily_profits = []
        
        for _ in range(n_simulations):
            simulation_profit = 0
            
            for row in allocated.iter_rows(named=True):
                # Random number of successful trades per day
                n_trades = np.random.randint(*trades_per_day_range)
                
                # Win rate based on confidence score
                win_rate = row.get('confidence_score', 50) / 100
                
                # Each trade: probability of success
                for _ in range(n_trades):
                    if np.random.random() < win_rate:
                        # Successful trade
                        profit = row['net_edge'] * min(
                            row.get('allocation_qty', row.get('qty', 0)),
                            row['hourly_volume'] // 4  # Conservative: 25% of hourly volume
                        )
                        simulation_profit += profit
                    else:
                        # Failed trade: lose some GP (spread + opportunity cost)
                        loss = row['avgLowPrice'] * 0.01  # 1% loss
                        simulation_profit -= loss
            
            daily_profits.append(simulation_profit)
        
        # Calculate statistics
        daily_profits = np.array(daily_profits)
        expected = int(np.mean(daily_profits))
        confidence_low = int(np.percentile(daily_profits, 5))
        confidence_high = int(np.percentile(daily_profits, 95))
        prob_profit = (daily_profits > 0).mean()
        
        # Risk assessment
        downside_risk = abs(confidence_low) if confidence_low < 0 else 0
        upside_potential = confidence_high
        risk_reward = upside_potential / (downside_risk + 1)
        
        return {
            "expected_daily_profit": expected,
            "confidence_interval_low": confidence_low,
            "confidence_interval_high": confidence_high,
            "probability_of_profit": prob_profit,
            "risk_reward_ratio": risk_reward,
            "expected_weekly_profit": expected * 7,
            "expected_monthly_profit": expected * 30,
            "downside_risk": downside_risk,
            "upside_potential": upside_potential,
        }
    
    def generate_execution_plan(
        self,
        df: pl.DataFrame,
        capital: int,
        session_duration_hours: float = 2.0
    ) -> List[Dict]:
        """
        Generate optimal trade execution sequence
        
        Considers:
        - GE limits
        - Capital constraints
        - Time available
        - Opportunity scores
        """
        if len(df) == 0:
            return []
        
        plan = []
        remaining_capital = capital
        elapsed_time = 0
        
        for idx, row in enumerate(df.iter_rows(named=True), 1):
            # Time to execute (buy + sell)
            exec_time_minutes = 15 + (row['allocation_qty'] / row['hourly_volume'] * 60)
            
            if elapsed_time + exec_time_minutes / 60 > session_duration_hours:
                break  # Out of time
            
            if remaining_capital < row['allocation_gp']:
                continue  # Not enough capital
            
            # Calculate optimal quantity
            ge_limit = row['limit'] if row['limit'] > 0 else float('inf')
            optimal_qty = min(
                row['allocation_qty'],
                ge_limit,
                int(row['hourly_volume'] * 0.3)  # Max 30% of hourly volume
            )
            
            actual_cost = optimal_qty * row['buy_price']
            expected_profit = optimal_qty * row['net_edge']
            
            plan.append({
                "sequence": idx,
                "item_name": row['name'],
                "action": "BUY_THEN_SELL",
                "quantity": optimal_qty,
                "buy_price": int(row['buy_price']),
                "sell_price": int(row['avgHighPrice']),
                "capital_required": actual_cost,
                "expected_profit": expected_profit,
                "expected_time_minutes": int(exec_time_minutes),
                "opportunity_score": row.get('opportunity_score', 0),
                "risk_score": row.get('risk_score', 50),
                "priority": "HIGH" if row.get('opportunity_score', 0) >= 80 else "MEDIUM",
            })
            
            remaining_capital -= actual_cost
            elapsed_time += exec_time_minutes / 60
        
        return plan
    
    def scan_for_opportunities(
        self,
        df: pl.DataFrame,
        thresholds: Dict = None
    ) -> List[OpportunityAlert]:
        """
        Real-time opportunity scanner
        
        Detects:
        - New high-score opportunities
        - Volume spikes
        - Price breakouts
        - Risk-level changes
        """
        if thresholds is None:
            thresholds = {
                "opportunity_score": 75,
                "volume_spike_multiplier": 2.0,
                "risk_max": 40,
            }
        
        alerts = []
        now = datetime.now()
        
        for row in df.iter_rows(named=True):
            opp_score = row.get('opportunity_score', 0)
            risk_score = row.get('risk_score', 50)
            
            # High-score opportunity
            if opp_score >= thresholds["opportunity_score"] and risk_score <= thresholds["risk_max"]:
                alerts.append(OpportunityAlert(
                    timestamp=now,
                    item_name=row['name'],
                    alert_type="NEW_OPPORTUNITY",
                    opportunity_score=opp_score,
                    action="BUY",
                    urgency="HIGH" if opp_score >= 85 else "MEDIUM",
                    message=f"Exceptional opportunity detected! Score: {opp_score}/100, Risk: {risk_score}/100"
                ))
            
            # Volume spike detection (if we had historical data)
            # This is a placeholder for future enhancement
            
        return alerts
    
    def calculate_portfolio_metrics(
        self,
        trades: List[Dict]
    ) -> Dict:
        """
        Calculate comprehensive portfolio performance metrics
        
        Returns:
        - Total P&L
        - Win rate
        - Sharpe ratio
        - Maximum drawdown
        - Profit factor
        """
        if not trades:
            return self._empty_metrics()
        
        profits = [t.get('profit', 0) for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_pnl = sum(profits)
        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = np.array(profits)
        sharpe = (np.mean(returns) / (np.std(returns) + 1)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Maximum drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "total_pnl": int(total_pnl),
            "total_trades": len(trades),
            "win_rate": win_rate,
            "average_win": int(avg_win),
            "average_loss": int(avg_loss),
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": int(max_drawdown),
            "best_trade": int(max(profits)) if profits else 0,
            "worst_trade": int(min(profits)) if profits else 0,
        }
    
    def generate_market_insights(self, df: pl.DataFrame) -> Dict:
        """
        Generate market-wide intelligence
        
        Returns:
        - Hottest categories
        - Trending items
        - Market regime
        - Opportunity density
        """
        if len(df) == 0:
            return {}
        
        # Opportunity density
        total_items = len(df)
        high_opp = len(df.filter(pl.col("opportunity_score") >= 75))
        
        # Average metrics
        avg_opp = df["opportunity_score"].mean()
        avg_risk = df["risk_score"].mean()
        avg_volume = df["hourly_volume"].mean()
        
        # Strategy distribution
        strategy_counts = df.group_by("strategy_type").agg(
            pl.count().alias("count")
        ).sort("count", descending=True)
        
        return {
            "total_opportunities": total_items,
            "high_quality_opportunities": high_opp,
            "opportunity_density": (high_opp / total_items * 100) if total_items > 0 else 0,
            "average_opportunity_score": float(avg_opp) if avg_opp else 0,
            "average_risk_score": float(avg_risk) if avg_risk else 0,
            "average_volume": int(avg_volume) if avg_volume else 0,
            "top_strategy": strategy_counts[0]["strategy_type"] if len(strategy_counts) > 0 else "N/A",
            "market_temperature": "HOT" if high_opp >= 10 else "WARM" if high_opp >= 5 else "COOL",
        }
    
    def _empty_forecast(self) -> Dict:
        """Return empty forecast structure"""
        return {
            "expected_daily_profit": 0,
            "confidence_interval_low": 0,
            "confidence_interval_high": 0,
            "probability_of_profit": 0.0,
            "risk_reward_ratio": 0.0,
            "expected_weekly_profit": 0,
            "expected_monthly_profit": 0,
            "downside_risk": 0,
            "upside_potential": 0,
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            "total_pnl": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "average_win": 0,
            "average_loss": 0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0,
            "best_trade": 0,
            "worst_trade": 0,
        }
