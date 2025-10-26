"""
Price History Analyzer - Detect True Bargains vs Falling Knives
Analyzes historical price trends to find items at genuinely low prices
"""

import polars as pl
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PriceAnalysis:
    """Analysis of item's price history"""
    item_name: str
    current_price: int
    avg_30d: Optional[int]
    high_30d: Optional[int]
    low_30d: Optional[int]
    
    # Key metrics
    vs_30d_avg_pct: float  # % difference from 30-day average
    vs_30d_low_pct: float  # % above 30-day low
    trend: str  # "UPTREND", "DOWNTREND", "STABLE", "VOLATILE"
    value_rating: str  # "BARGAIN", "FAIR", "OVERPRICED", "FALLING_KNIFE"
    
    # Confidence
    confidence: float  # 0-1
    recommendation: str

class PriceHistoryAnalyzer:
    """Analyzes price history to find true bargains"""
    
    def __init__(self, api_client):
        self.api = api_client
        self._cache = {}
    
    def analyze_item(self, item_id: int, item_name: str, current_price: int) -> Optional[PriceAnalysis]:
        """
        Analyze if an item is a true bargain or a falling knife
        
        Returns:
            PriceAnalysis with buy recommendation
        """
        # Get 30-day price history
        try:
            history_df = self.api.get_timeseries(item_id, timestep="5m")
            
            if history_df is None or len(history_df) == 0:
                return None
            
            # Calculate 30-day metrics
            prices = history_df['avgHighPrice'].to_list()
            prices = [p for p in prices if p is not None and p > 0]
            
            if len(prices) < 10:  # Need at least 10 data points
                return None
            
            avg_30d = int(np.mean(prices))
            high_30d = int(np.max(prices))
            low_30d = int(np.min(prices))
            
            # Calculate position vs historical
            vs_avg_pct = ((current_price - avg_30d) / avg_30d * 100)
            vs_low_pct = ((current_price - low_30d) / low_30d * 100)
            
            # Detect trend (last 7 days vs previous 7 days)
            recent = prices[-20:]  # Recent ~7 days (5m intervals)
            older = prices[-40:-20]  # Previous ~7 days
            
            if len(recent) >= 5 and len(older) >= 5:
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                trend_pct = ((recent_avg - older_avg) / older_avg * 100)
                
                if trend_pct > 5:
                    trend = "UPTREND"
                elif trend_pct < -5:
                    trend = "DOWNTREND"
                else:
                    # Check volatility
                    std_dev = np.std(prices)
                    cv = (std_dev / avg_30d * 100) if avg_30d > 0 else 0
                    trend = "VOLATILE" if cv > 10 else "STABLE"
            else:
                trend = "UNKNOWN"
            
            # Determine value rating
            value_rating, recommendation, confidence = self._calculate_value_rating(
                current_price, avg_30d, low_30d, high_30d, 
                vs_avg_pct, vs_low_pct, trend
            )
            
            return PriceAnalysis(
                item_name=item_name,
                current_price=current_price,
                avg_30d=avg_30d,
                high_30d=high_30d,
                low_30d=low_30d,
                vs_30d_avg_pct=vs_avg_pct,
                vs_30d_low_pct=vs_low_pct,
                trend=trend,
                value_rating=value_rating,
                confidence=confidence,
                recommendation=recommendation
            )
            
        except Exception as e:
            print(f"Error analyzing {item_name}: {e}")
            return None
    
    def _calculate_value_rating(
        self, current: int, avg: int, low: int, high: int,
        vs_avg_pct: float, vs_low_pct: float, trend: str
    ) -> tuple[str, str, float]:
        """
        Determine if item is a bargain or falling knife
        
        Returns:
            (value_rating, recommendation, confidence)
        """
        # BARGAIN: At/near historical low + stable/uptrend
        if vs_avg_pct <= -10 and vs_low_pct <= 5 and trend in ["UPTREND", "STABLE"]:
            return (
                "BARGAIN",
                "🔥 TRUE BARGAIN! Item is 10%+ below avg and at/near historical low. BUY MASS QUANTITY!",
                0.9
            )
        
        # FALLING KNIFE: Below average but still dropping
        elif vs_avg_pct <= -10 and trend == "DOWNTREND":
            return (
                "FALLING_KNIFE",
                "⚠️ FALLING KNIFE! Item is cheap but still dropping. AVOID for long holds!",
                0.85
            )
        
        # GOOD VALUE: Below average, stable
        elif vs_avg_pct <= -5 and trend in ["STABLE", "UPTREND"]:
            return (
                "FAIR",
                "✅ Good value. Below average and stable/rising. Good for medium-term hold.",
                0.75
            )
        
        # OVERPRICED: Above average
        elif vs_avg_pct >= 10:
            return (
                "OVERPRICED",
                "❌ OVERPRICED! Item is 10%+ above average. Wait for better entry.",
                0.8
            )
        
        # AT PEAK: At or near historical high
        elif vs_low_pct >= 80:  # Within 20% of historical high
            return (
                "PEAK",
                "⚠️ AT PEAK! Near historical high. Risky entry point.",
                0.7
            )
        
        # FAIR: Near average
        else:
            return (
                "FAIR",
                "➡️ Fair price. Near 30-day average. Standard opportunity.",
                0.6
            )
    
    def get_bargain_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add price history analysis to trading dataframe
        
        Adds columns:
        - value_rating: BARGAIN, FAIR, OVERPRICED, FALLING_KNIFE
        - vs_avg_pct: % vs 30-day average
        - trend: UPTREND, DOWNTREND, STABLE
        """
        analyses = []
        
        for row in df.iter_rows(named=True):
            analysis = self.analyze_item(
                row['item_id'],
                row['name'],
                int(row['avgHighPrice'])
            )
            
            if analysis:
                analyses.append({
                    'item_id': row['item_id'],
                    'value_rating': analysis.value_rating,
                    'vs_avg_pct': analysis.vs_30d_avg_pct,
                    'trend': analysis.trend,
                    'price_recommendation': analysis.recommendation,
                    'price_confidence': analysis.confidence
                })
        
        if not analyses:
            # Add empty columns if no analysis available
            return df.with_columns([
                pl.lit("UNKNOWN").alias("value_rating"),
                pl.lit(0.0).alias("vs_avg_pct"),
                pl.lit("UNKNOWN").alias("trend"),
                pl.lit("No historical data").alias("price_recommendation"),
                pl.lit(0.0).alias("price_confidence")
            ])
        
        analysis_df = pl.DataFrame(analyses)
        return df.join(analysis_df, on="item_id", how="left")
