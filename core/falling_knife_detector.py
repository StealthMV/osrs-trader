"""
Falling Knife Detection for OSRS Trading
Detects items with declining prices to avoid catching a falling knife
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple
from .api import OSRSPricesAPI


def detect_falling_knife(item_id: int, item_name: str) -> Dict[str, any]:
    """
    Detect if an item is a falling knife (price is crashing)
    
    Args:
        item_id: Item ID to check
        item_name: Item name (for logging)
    
    Returns:
        Dict with:
            - is_falling_knife: bool
            - momentum_1h: float (% change over 1 hour)
            - momentum_6h: float (% change over 6 hours) 
            - momentum_24h: float (% change over 24 hours)
            - trend: str ("STRONG_DOWN", "DOWN", "NEUTRAL", "UP", "STRONG_UP")
            - warning: str (warning message if falling knife)
            - confidence: float (0-1, how confident we are in the analysis)
    """
    
    try:
        with OSRSPricesAPI() as api:
            # Fetch 6h timeseries (gives us ~30 hours of data)
            data_6h = api.get_timeseries(item_id, timestep="6h")
            
            if not data_6h or "data" not in data_6h or len(data_6h["data"]) < 3:
                # Not enough data - can't determine trend
                return {
                    "is_falling_knife": False,
                    "momentum_1h": 0.0,
                    "momentum_6h": 0.0,
                    "momentum_24h": 0.0,
                    "trend": "UNKNOWN",
                    "warning": None,
                    "confidence": 0.0,
                }
            
            # Convert to DataFrame
            df = pl.DataFrame(data_6h["data"])
            
            # Sort by timestamp
            df = df.sort("timestamp")
            
            # Calculate average mid prices for each period
            df = df.with_columns([
                ((pl.col("avgHighPrice") + pl.col("avgLowPrice")) / 2).alias("mid_price")
            ])
            
            prices = df["mid_price"].to_list()
            
            # Need at least 3 data points
            if len(prices) < 3:
                return {
                    "is_falling_knife": False,
                    "momentum_1h": 0.0,
                    "momentum_6h": 0.0,
                    "momentum_24h": 0.0,
                    "trend": "UNKNOWN",
                    "warning": None,
                    "confidence": 0.0,
                }
            
            # Calculate momentum at different timeframes
            # Most recent price vs previous periods
            current_price = prices[-1]
            
            # 6h momentum (current vs 1 period ago)
            momentum_6h = ((current_price - prices[-2]) / prices[-2] * 100) if len(prices) >= 2 else 0.0
            
            # 12h momentum (current vs 2 periods ago) 
            momentum_12h = ((current_price - prices[-3]) / prices[-3] * 100) if len(prices) >= 3 else 0.0
            
            # 24h momentum (current vs 4 periods ago)
            momentum_24h = ((current_price - prices[-5]) / prices[-5] * 100) if len(prices) >= 5 else momentum_12h
            
            # Calculate trend strength
            # Check if consecutive periods show decline
            consecutive_declines = 0
            for i in range(len(prices) - 1, 0, -1):
                if prices[i] < prices[i-1]:
                    consecutive_declines += 1
                else:
                    break
            
            # Determine if falling knife
            is_falling_knife = False
            trend = "NEUTRAL"
            warning = None
            
            # STRONG FALLING KNIFE: 
            # - Down 5%+ in 24h AND down 2%+ in 6h
            # - OR 3+ consecutive declines
            if (momentum_24h <= -5.0 and momentum_6h <= -2.0) or consecutive_declines >= 3:
                is_falling_knife = True
                trend = "STRONG_DOWN"
                warning = f"🚨 FALLING KNIFE! Price down {momentum_24h:.1f}% in 24h. AVOID!"
            
            # MODERATE FALLING KNIFE:
            # - Down 3%+ in 24h
            elif momentum_24h <= -3.0:
                is_falling_knife = True
                trend = "DOWN"
                warning = f"⚠️ Declining price: {momentum_24h:.1f}% in 24h. Be cautious!"
            
            # MILD DECLINE
            elif momentum_24h <= -1.5:
                trend = "DOWN"
                warning = f"📉 Slight decline: {momentum_24h:.1f}% in 24h"
            
            # RISING
            elif momentum_24h >= 3.0:
                trend = "STRONG_UP"
            elif momentum_24h >= 1.0:
                trend = "UP"
            
            # Confidence based on data availability
            confidence = min(len(prices) / 5.0, 1.0)
            
            return {
                "is_falling_knife": is_falling_knife,
                "momentum_1h": 0.0,  # Would need 1h data for this
                "momentum_6h": momentum_6h,
                "momentum_24h": momentum_24h,
                "trend": trend,
                "warning": warning,
                "confidence": confidence,
                "consecutive_declines": consecutive_declines,
            }
            
    except Exception as e:
        # If API fails, don't block trading - just return unknown
        return {
            "is_falling_knife": False,
            "momentum_1h": 0.0,
            "momentum_6h": 0.0,
            "momentum_24h": 0.0,
            "trend": "UNKNOWN",
            "warning": None,
            "confidence": 0.0,
        }


def batch_detect_falling_knives(items: list) -> Dict[int, Dict]:
    """
    Check multiple items for falling knife patterns
    
    Args:
        items: List of dicts with 'item_id' and 'name'
    
    Returns:
        Dict mapping item_id -> falling knife analysis
    """
    results = {}
    
    for item in items:
        item_id = item['item_id']
        item_name = item['name']
        
        analysis = detect_falling_knife(item_id, item_name)
        results[item_id] = analysis
    
    return results
