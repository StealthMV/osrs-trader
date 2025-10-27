"""
COMPREHENSIVE MARKET DASHBOARD
Real-time market intelligence and analytics

Market Temperature Explained:
- HOT (🔥): 10+ high-quality opportunities (score 75+) = AGGRESSIVE TRADING
- WARM (🌡️): 5-9 high-quality opportunities = NORMAL TRADING  
- COOL (❄️): <5 high-quality opportunities = SELECTIVE/DEFENSIVE

This tells you how many GREAT trades are available RIGHT NOW!
"""

import polars as pl
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketSnapshot:
    """Complete market state"""
    timestamp: datetime
    temperature: str  # HOT, WARM, COOL
    total_opportunities: int
    high_quality_count: int
    average_profit_pct: float
    average_risk: float
    total_volume: int
    market_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    recommended_strategy: str
    confidence: float

class MarketDashboard:
    """
    ULTIMATE MARKET INTELLIGENCE SYSTEM
    
    Provides real-time market analysis including:
    - Market temperature (how many good trades exist)
    - Sentiment analysis (bullish vs bearish)
    - Volume analysis (liquidity and activity)
    - Opportunity density (quality of available trades)
    - Recommended actions based on current conditions
    """
    
    def __init__(self):
        self.snapshots = []
        
    def get_comprehensive_market_state(self, df: pl.DataFrame) -> MarketSnapshot:
        """
        MASTER MARKET ANALYZER
        
        Returns complete market snapshot with:
        - Temperature (HOT/WARM/COOL based on opportunity count)
        - Sentiment (BULLISH/BEARISH/NEUTRAL based on edges and risks)
        - Recommended strategy
        - Confidence in analysis
        """
        if len(df) == 0:
            return self._empty_snapshot()
        
        # Calculate key metrics
        total_opps = len(df)
        high_quality = len(df.filter(pl.col('opportunity_score') >= 75))
        
        avg_profit = df['edge_pct'].mean() if 'edge_pct' in df.columns else 0
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 50
        total_volume = df['hourly_volume'].sum() if 'hourly_volume' in df.columns else 0
        
        # TEMPERATURE = How many HIGH-QUALITY trades exist
        if high_quality >= 10:
            temperature = "HOT 🔥"
            temp_desc = "EXCELLENT! Many high-quality opportunities available!"
        elif high_quality >= 5:
            temperature = "WARM 🌡️"
            temp_desc = "Good market conditions. Normal trading recommended."
        else:
            temperature = "COOL ❄️"
            temp_desc = "Limited opportunities. Be selective or wait for better conditions."
        
        # SENTIMENT = Are opportunities good or risky?
        if avg_profit > 4.0 and avg_risk < 40:
            sentiment = "BULLISH 📈"
            sentiment_desc = "Strong profits with low risk. Deploy capital aggressively."
        elif avg_profit < 2.0 or avg_risk > 60:
            sentiment = "BEARISH 📉"
            sentiment_desc = "Weak opportunities or high risk. Trade defensively."
        else:
            sentiment = "NEUTRAL ➡️"
            sentiment_desc = "Balanced market. Stick to proven strategies."
        
        # RECOMMENDED STRATEGY based on temperature + sentiment
        if temperature == "HOT 🔥" and "BULLISH" in sentiment:
            recommended_strategy = "🚀 AGGRESSIVE: Deploy full capital, prioritize highest opportunity scores"
            confidence = 0.90
        elif temperature == "HOT 🔥":
            recommended_strategy = "⚡ ACTIVE: Many trades available, but be selective on risk"
            confidence = 0.80
        elif temperature == "WARM 🌡️" and "BULLISH" in sentiment:
            recommended_strategy = "✅ NORMAL: Good conditions, trade your usual strategy"
            confidence = 0.75
        elif temperature == "COOL ❄️" or "BEARISH" in sentiment:
            recommended_strategy = "🛡️ DEFENSIVE: Only take highest-confidence, lowest-risk trades"
            confidence = 0.70
        else:
            recommended_strategy = "⏸️ WAIT: Consider waiting for better market conditions"
            confidence = 0.60
        
        return MarketSnapshot(
            timestamp=datetime.now(),
            temperature=temperature,
            total_opportunities=total_opps,
            high_quality_count=high_quality,
            average_profit_pct=float(avg_profit),
            average_risk=float(avg_risk),
            total_volume=int(total_volume),
            market_sentiment=sentiment,
            recommended_strategy=recommended_strategy,
            confidence=confidence
        )
    
    def calculate_opportunity_heatmap(self, df: pl.DataFrame) -> Dict[str, int]:
        """
        OPPORTUNITY HEATMAP
        
        Shows where the best trades are by category:
        - ELITE (90-100): Absolute best
        - EXCELLENT (75-89): Very strong
        - GOOD (60-74): Solid opportunities
        - MEDIOCRE (40-59): Marginal
        - POOR (<40): Avoid
        """
        if len(df) == 0:
            return {}
        
        heatmap = {
            "ELITE (90-100)": len(df.filter(pl.col('opportunity_score') >= 90)),
            "EXCELLENT (75-89)": len(df.filter((pl.col('opportunity_score') >= 75) & (pl.col('opportunity_score') < 90))),
            "GOOD (60-74)": len(df.filter((pl.col('opportunity_score') >= 60) & (pl.col('opportunity_score') < 75))),
            "MEDIOCRE (40-59)": len(df.filter((pl.col('opportunity_score') >= 40) & (pl.col('opportunity_score') < 60))),
            "POOR (<40)": len(df.filter(pl.col('opportunity_score') < 40)),
        }
        
        return heatmap
    
    def calculate_risk_distribution(self, df: pl.DataFrame) -> Dict[str, int]:
        """
        RISK BREAKDOWN
        
        Shows risk levels across all opportunities:
        - LOW (0-30): Safest trades
        - MEDIUM (31-60): Moderate risk
        - HIGH (61-80): Risky
        - EXTREME (81-100): Very dangerous
        """
        if len(df) == 0:
            return {}
        
        risk_dist = {
            "LOW (0-30) 🟢": len(df.filter(pl.col('risk_score') <= 30)),
            "MEDIUM (31-60) 🟡": len(df.filter((pl.col('risk_score') > 30) & (pl.col('risk_score') <= 60))),
            "HIGH (61-80) 🟠": len(df.filter((pl.col('risk_score') > 60) & (pl.col('risk_score') <= 80))),
            "EXTREME (81-100) 🔴": len(df.filter(pl.col('risk_score') > 80)),
        }
        
        return risk_dist
    
    def calculate_volume_analysis(self, df: pl.DataFrame) -> Dict:
        """
        VOLUME INTELLIGENCE
        
        Analyzes market liquidity and activity:
        - Total hourly volume across all items
        - High-volume items (>200/hr)
        - Average volume per opportunity
        - Liquidity score (how easy to execute trades)
        """
        if len(df) == 0:
            return {}
        
        total_volume = df['hourly_volume'].sum()
        avg_volume = df['hourly_volume'].mean()
        high_volume_items = len(df.filter(pl.col('hourly_volume') >= 200))
        
        # Liquidity score: how much of total volume is in good opportunities?
        good_opps = df.filter(pl.col('opportunity_score') >= 60)
        good_volume = good_opps['hourly_volume'].sum() if len(good_opps) > 0 else 0
        liquidity_score = (good_volume / total_volume * 100) if total_volume > 0 else 0
        
        return {
            "total_hourly_volume": int(total_volume),
            "average_volume": int(avg_volume),
            "high_volume_items": high_volume_items,
            "liquidity_score": float(liquidity_score),
            "liquidity_rating": "EXCELLENT" if liquidity_score > 60 else "GOOD" if liquidity_score > 40 else "FAIR" if liquidity_score > 20 else "POOR"
        }
    
    def get_top_movers(self, df: pl.DataFrame) -> List[Dict]:
        """
        TOP MOVERS
        
        Finds items with the most momentum:
        - Highest volume
        - Highest edges
        - Best opportunity scores
        - Biggest potential profits
        """
        if len(df) == 0:
            return []
        
        movers = []
        
        # Top by opportunity score
        top_opp = df.sort('opportunity_score', descending=True).head(3)
        for row in top_opp.iter_rows(named=True):
            movers.append({
                "category": "🏆 BEST OPPORTUNITY",
                "item": row['name'],
                "value": f"{row['opportunity_score']:.0f}/100",
                "detail": f"{row['edge_pct']:.1f}% edge, {row.get('risk_score', 50):.0f} risk"
            })
        
        # Top by volume
        top_vol = df.sort('hourly_volume', descending=True).head(2)
        for row in top_vol.iter_rows(named=True):
            movers.append({
                "category": "💧 HIGHEST LIQUIDITY",
                "item": row['name'],
                "value": f"{row['hourly_volume']:,}/hr",
                "detail": f"{row['edge_pct']:.1f}% edge"
            })
        
        # Top by edge %
        top_edge = df.sort('edge_pct', descending=True).head(2)
        for row in top_edge.iter_rows(named=True):
            movers.append({
                "category": "💰 HIGHEST PROFIT %",
                "item": row['name'],
                "value": f"{row['edge_pct']:.1f}%",
                "detail": f"{row['hourly_volume']:,}/hr volume"
            })
        
        return movers
    
    def generate_executive_summary(self, df: pl.DataFrame, capital: int) -> str:
        """
        EXECUTIVE SUMMARY
        
        One-paragraph summary of market conditions for quick decision making
        """
        snapshot = self.get_comprehensive_market_state(df)
        
        summary = f"""
        **MARKET STATUS ({snapshot.timestamp.strftime('%H:%M:%S')})**
        
        Temperature is **{snapshot.temperature}** with **{snapshot.high_quality_count}** elite opportunities out of {snapshot.total_opportunities} total items. 
        Market sentiment is **{snapshot.market_sentiment}** (Avg Profit: {snapshot.average_profit_pct:.1f}%, Avg Risk: {snapshot.average_risk:.0f}/100).
        
        **{snapshot.recommended_strategy}**
        
        Analysis Confidence: {snapshot.confidence*100:.0f}%
        """
        
        return summary.strip()
    
    def _empty_snapshot(self) -> MarketSnapshot:
        """Return empty market snapshot"""
        return MarketSnapshot(
            timestamp=datetime.now(),
            temperature="UNKNOWN",
            total_opportunities=0,
            high_quality_count=0,
            average_profit_pct=0.0,
            average_risk=0.0,
            total_volume=0,
            market_sentiment="UNKNOWN",
            recommended_strategy="No data available",
            confidence=0.0
        )
