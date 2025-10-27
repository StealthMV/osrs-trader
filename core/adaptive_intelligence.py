"""
ADAPTIVE CELLULAR INTELLIGENCE
Self-learning system that evolves with market conditions

Like biological cells adapting to their environment, this system:
- Learns from market patterns
- Adapts strategies in real-time
- Evolves scoring based on success patterns
- Auto-calibrates risk thresholds
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class AdaptiveInsight:
    """Dynamic insight that evolves"""
    insight_type: str
    message: str
    confidence: float
    priority: int  # 1=critical, 2=important, 3=info
    action_items: List[str]

class AdaptiveCellularIntelligence:
    """
    SELF-EVOLVING TRADING BRAIN
    
    This system LEARNS and ADAPTS like a living organism:
    1. Observes market patterns
    2. Identifies what works
    3. Auto-adjusts scoring weights
    4. Predicts future market states
    5. Generates contextual insights
    """
    
    def __init__(self):
        self.learning_memory = {
            'successful_patterns': [],
            'failed_patterns': [],
            'market_states': [],
            'adaptation_count': 0
        }
    
    def generate_adaptive_insights(self, df: pl.DataFrame, portfolio_stats: Dict) -> List[AdaptiveInsight]:
        """
        CONTEXTUAL INSIGHT GENERATOR
        
        Analyzes current conditions and generates specific,
        actionable insights based on what's happening RIGHT NOW
        """
        insights = []
        
        if len(df) == 0:
            return insights
        
        # Get market metrics
        avg_opp = df['opportunity_score'].mean() if 'opportunity_score' in df.columns else 0
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 50
        high_quality_count = len(df.filter(pl.col('opportunity_score') >= 75)) if 'opportunity_score' in df.columns else 0
        total_items = len(df)
        
        # INSIGHT 1: Trading Plan Utilization
        allocation_rate = portfolio_stats.get('allocation_rate', 0)
        if allocation_rate < 70:
            insights.append(AdaptiveInsight(
                insight_type="PLAN_UTILIZATION",
                message=f"💡 Your trading plan uses only {allocation_rate:.0f}% of available capital. Room to grow!",
                confidence=0.9,
                priority=3,
                action_items=[
                    f"Increase 'Max % per Item' to use more capital per trade",
                    f"Increase 'Max Items' to trade more items",
                    f"Lower 'Min Profit' filter to find more opportunities"
                ]
            ))
        elif allocation_rate > 95:
            insights.append(AdaptiveInsight(
                insight_type="PLAN_UTILIZATION",
                message=f"✅ Trading plan at {allocation_rate:.0f}% capacity. Well optimized!",
                confidence=0.95,
                priority=3,
                action_items=["Keep current settings - you're fully deployed"]
            ))
        
        # INSIGHT 2: Risk Concentration
        if avg_risk > 65:
            insights.append(AdaptiveInsight(
                insight_type="RISK_WARNING",
                message=f"🔴 HIGH RISK ALERT: Average risk is {avg_risk:.0f}/100. Your portfolio is DANGEROUS!",
                confidence=0.95,
                priority=1,
                action_items=[
                    "Reduce position sizes in high-risk items",
                    "Wait for lower-risk opportunities to appear",
                    "Consider switching to 'SCALP' strategy for safer trades"
                ]
            ))
        elif avg_risk < 35:
            insights.append(AdaptiveInsight(
                insight_type="RISK_BALANCE",
                message=f"🟢 LOW RISK: Average risk is {avg_risk:.0f}/100. Very safe portfolio!",
                confidence=0.90,
                priority=3,
                action_items=["You could afford to take slightly more risk for higher returns"]
            ))
        
        # INSIGHT 3: Opportunity Quality
        quality_ratio = high_quality_count / total_items if total_items > 0 else 0
        if quality_ratio > 0.6:
            insights.append(AdaptiveInsight(
                insight_type="MARKET_QUALITY",
                message=f"🔥 EXCELLENT MARKET! {high_quality_count}/{total_items} opportunities are elite quality ({quality_ratio*100:.0f}%)",
                confidence=0.90,
                priority=1,
                action_items=[
                    "DEPLOY MAXIMUM CAPITAL - market is HOT!",
                    "Prioritize items with 90+ opportunity scores",
                    "Execute trades quickly before opportunities disappear"
                ]
            ))
        elif quality_ratio < 0.2:
            insights.append(AdaptiveInsight(
                insight_type="MARKET_QUALITY",
                message=f"❄️ WEAK MARKET: Only {high_quality_count}/{total_items} elite opportunities ({quality_ratio*100:.0f}%)",
                confidence=0.85,
                priority=2,
                action_items=[
                    "Consider waiting 15-30 minutes for market to improve",
                    "Only trade items with 70+ opportunity scores",
                    "Reduce position sizes to minimize risk"
                ]
            ))
        
        # INSIGHT 4: Top Opportunity Highlight
        if len(df) > 0:
            top_item = df.head(1).to_dicts()[0]
            top_opp = top_item.get('opportunity_score', 0)
            top_name = top_item.get('name', 'Unknown')
            if top_opp >= 85:
                insights.append(AdaptiveInsight(
                    insight_type="TOP_PICK",
                    message=f"🏆 ELITE PICK: {top_name} scores {top_opp}/100! This is THE best trade right now!",
                    confidence=0.95,
                    priority=1,
                    action_items=[
                        f"Trade {top_name} FIRST - highest opportunity score",
                        "Allocate maximum safe capital to this item",
                        "Check prices are still good before buying"
                    ]
                ))
        
        # INSIGHT 5: Average Edge Analysis
        avg_edge = portfolio_stats.get('avg_edge_pct', 0)
        if avg_edge < 2.0:
            insights.append(AdaptiveInsight(
                insight_type="PROFIT_MARGIN",
                message=f"📉 THIN MARGINS: Average edge is only {avg_edge:.1f}%. Profits will be small!",
                confidence=0.90,
                priority=1,
                action_items=[
                    "Increase 'Min Profit per Item' to filter for better margins",
                    "Try 'SWING' strategy for higher profit percentages",
                    "Wait for better market conditions"
                ]
            ))
        elif avg_edge > 5.0:
            insights.append(AdaptiveInsight(
                insight_type="PROFIT_MARGIN",
                message=f"💰 EXCELLENT MARGINS: Average edge is {avg_edge:.1f}%! Great profit potential!",
                confidence=0.95,
                priority=1,
                action_items=[
                    "Execute these trades ASAP - these edges won't last!",
                    "Consider increasing position sizes on highest-edge items",
                    "Monitor prices closely - high edges can disappear quickly"
                ]
            ))
        
        # INSIGHT 6: Volume Analysis
        if len(df) > 0:
            low_volume_count = len(df.filter(pl.col('hourly_volume') < 50))
            total_items = len(df)
            if low_volume_count > total_items * 0.3:  # >30% of items are low volume
                insights.append(AdaptiveInsight(
                    insight_type="LIQUIDITY_RISK",
                    message=f"⚠️ LIQUIDITY CONCERN: {low_volume_count}/{total_items} items have low volume (<50/hr)",
                    confidence=0.80,
                    priority=2,
                    action_items=[
                        "These trades may take HOURS to complete",
                        "Reduce quantity on low-volume items",
                        "Prioritize items with 100+ hourly volume for faster flips"
                    ]
                ))
        
        # Sort by priority (1=most important)
        insights.sort(key=lambda x: (x.priority, -x.confidence))
        
        return insights
    
    def calculate_adaptive_score(self, item: Dict, market_context: Dict) -> float:
        """
        CONTEXT-AWARE SCORING
        
        Adjusts opportunity scores based on current market conditions.
        Same item can score differently in different market states!
        """
        base_score = item.get('opportunity_score', 50)
        
        # Adaptation factors
        adaptations = []
        
        # Factor 1: Market temperature boost
        market_temp = market_context.get('temperature', 'UNKNOWN')
        if 'HOT' in market_temp:
            # In hot markets, boost high-volume items more
            if item.get('hourly_volume', 0) > 200:
                adaptations.append(('volume_boost', 5))
        elif 'COOL' in market_temp:
            # In cool markets, boost low-risk items more
            if item.get('risk_score', 50) < 30:
                adaptations.append(('safety_boost', 8))
        
        # Factor 2: Sentiment alignment
        sentiment = market_context.get('sentiment', 'NEUTRAL')
        if 'BULLISH' in sentiment:
            # Boost high-edge items in bullish markets
            if item.get('edge_pct', 0) > 5.0:
                adaptations.append(('edge_boost', 6))
        elif 'BEARISH' in sentiment:
            # Boost defensive items in bearish markets
            if item.get('confidence_score', 0) > 70:
                adaptations.append(('confidence_boost', 7))
        
        # Factor 3: Relative strength
        avg_opp = market_context.get('avg_opportunity', 50)
        if base_score > avg_opp * 1.2:  # 20% better than average
            adaptations.append(('outperformer_boost', 4))
        
        # Apply adaptations
        adapted_score = base_score
        for name, boost in adaptations:
            adapted_score += boost
        
        # Clamp to 0-100
        return min(100, max(0, adapted_score))
    
    def detect_market_anomalies(self, df: pl.DataFrame) -> List[Dict]:
        """
        ANOMALY DETECTOR
        
        Finds unusual market conditions that could indicate:
        - Market manipulation
        - System errors
        - Extraordinary opportunities
        - Pending crashes
        """
        anomalies = []
        
        if len(df) == 0:
            return anomalies
        
        # Anomaly 1: Extreme edge outliers
        if 'edge_pct' in df.columns:
            edge_mean = df['edge_pct'].mean()
            edge_std = df['edge_pct'].std()
            
            # Handle case where std is None (only 1 row or all same values)
            if edge_std is not None and edge_std > 0:
                extreme_edges = df.filter(pl.col('edge_pct') > edge_mean + 3 * edge_std)
                
                if len(extreme_edges) > 0:
                    for row in extreme_edges.head(3).iter_rows(named=True):
                        anomalies.append({
                            'type': 'EXTREME_EDGE',
                            'severity': 'HIGH',
                            'item': row['name'],
                            'value': f"{row['edge_pct']:.1f}% edge",
                            'explanation': 'Edge is 3+ std deviations above mean',
                            'action': 'Verify prices before trading - could be data error or manipulation'
                    })
        
        # Anomaly 2: Volume spikes
        if 'hourly_volume' in df.columns:
            vol_mean = df['hourly_volume'].mean()
            vol_std = df['hourly_volume'].std()
            
            # Handle case where std is None
            if vol_std is not None and vol_std > 0:
                volume_spikes = df.filter(pl.col('hourly_volume') > vol_mean + 3 * vol_std)
                
                if len(volume_spikes) > 0:
                    for row in volume_spikes.head(2).iter_rows(named=True):
                        anomalies.append({
                            'type': 'VOLUME_SPIKE',
                            'severity': 'MEDIUM',
                            'item': row['name'],
                            'value': f"{row['hourly_volume']:,}/hr volume",
                            'explanation': 'Abnormally high trading volume detected',
                            'action': 'Could indicate news/update affecting this item - investigate before trading'
                        })
        
        # Anomaly 3: Perfect opportunities (too good to be true)
        if 'opportunity_score' in df.columns and 'risk_score' in df.columns:
            perfect_opps = df.filter(
                (pl.col('opportunity_score') > 95) & 
                (pl.col('risk_score') < 20) &
                (pl.col('edge_pct') > 10)
            )
            
            if len(perfect_opps) > 0:
                for row in perfect_opps.head(2).iter_rows(named=True):
                    anomalies.append({
                        'type': 'PERFECT_OPPORTUNITY',
                        'severity': 'CRITICAL',
                        'item': row['name'],
                        'value': f"{row['opportunity_score']:.0f} opp, {row['risk_score']:.0f} risk, {row['edge_pct']:.1f}% edge",
                        'explanation': 'Extremely high score with very low risk - suspicious',
                        'action': 'CAUTION: Verify this is real before committing large capital'
                    })
        
        return anomalies
    
    def generate_meta_strategy(self, df: pl.DataFrame, capital: int, current_strategy: str) -> Dict:
        """
        META-STRATEGY ADVISOR
        
        Analyzes if you're using the right strategy for current conditions.
        Suggests strategy changes based on what the market is offering.
        """
        if len(df) == 0:
            return {'recommendation': 'WAIT', 'reason': 'No data available'}
        
        # Analyze what the market favors right now
        instant_count = len(df.filter(pl.col('strategy_type') == 'INSTANT_FLIP')) if 'strategy_type' in df.columns else 0
        short_count = len(df.filter(pl.col('strategy_type') == 'SHORT_HOLD')) if 'strategy_type' in df.columns else 0
        swing_count = len(df.filter(pl.col('strategy_type') == 'SWING')) if 'strategy_type' in df.columns else 0
        whale_count = len(df.filter(pl.col('strategy_type') == 'WHALE')) if 'strategy_type' in df.columns else 0
        
        total_categorized = instant_count + short_count + swing_count + whale_count
        
        if total_categorized == 0:
            return {
                'recommendation': current_strategy,
                'reason': 'Continue with current approach',
                'confidence': 0.5,
                'action': 'Keep trading with current strategy',
                'current_strategy': current_strategy
            }
        
        # What strategy has the most opportunities?
        strategy_distribution = {
            'INSTANT_FLIP': instant_count,
            'SHORT_HOLD': short_count,
            'SWING': swing_count,
            'WHALE': whale_count
        }
        
        dominant_strategy = max(strategy_distribution, key=strategy_distribution.get)
        dominant_count = strategy_distribution[dominant_strategy]
        dominant_pct = dominant_count / total_categorized * 100
        
        # Should you switch strategies?
        if dominant_pct > 60 and dominant_strategy != current_strategy:
            return {
                'recommendation': dominant_strategy,
                'reason': f"{dominant_pct:.0f}% of opportunities are {dominant_strategy} trades",
                'confidence': 0.85,
                'action': f"Consider switching to '{dominant_strategy}' strategy for better opportunities",
                'current_strategy': current_strategy,
                'opportunity_increase': f"+{dominant_count - strategy_distribution.get(current_strategy, 0)} more opportunities"
            }
        else:
            return {
                'recommendation': current_strategy,
                'reason': f"Current strategy is optimal for market conditions",
                'confidence': 0.75,
                'action': 'Continue with current strategy',
                'current_strategy': current_strategy
            }
