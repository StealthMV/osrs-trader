"""
Market Microstructure Analysis - The Missing Piece
Analyzes order flow, time-to-fill, and market pressure
"""

import polars as pl
import numpy as np
from typing import Dict, Tuple


def calculate_time_to_fill(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estimate how long it takes to fill your orders
    
    Logic:
    - If hourly_volume = 100 and you want to buy 500, that's 5 hours
    - But we're more conservative: assume you capture 30% of market volume
    - So 500 units at 100/hr with 30% capture = 16.7 hours (ouch!)
    
    Returns:
    - time_to_fill_hours: Hours to fill your suggested quantity
    - fill_difficulty: INSTANT, QUICK, MODERATE, SLOW, VERY_SLOW
    """
    
    return df.with_columns([
        # Assume we can capture 30% of hourly volume (conservative)
        # Example: 100/hr market volume, we can buy/sell 30 per hour
        (
            pl.col("allocation_qty") / (pl.col("hourly_volume") * 0.3 + 1)
        ).alias("time_to_fill_hours"),
    ]).with_columns([
        # Classify fill difficulty
        pl.when(pl.col("time_to_fill_hours") <= 0.5)
          .then(pl.lit("⚡ INSTANT"))
        .when(pl.col("time_to_fill_hours") <= 2.0)
          .then(pl.lit("🟢 QUICK"))
        .when(pl.col("time_to_fill_hours") <= 8.0)
          .then(pl.lit("🟡 MODERATE"))
        .when(pl.col("time_to_fill_hours") <= 24.0)
          .then(pl.lit("🟠 SLOW"))
        .otherwise(pl.lit("🔴 VERY SLOW"))
        .alias("fill_difficulty"),
    ])


def calculate_market_pressure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze buy vs sell pressure to predict price direction
    
    Logic:
    - highPriceVolume = people selling to buyers (sell pressure)
    - lowPriceVolume = people buying from sellers (buy pressure)
    - If sell pressure >> buy pressure = price will DROP
    - If buy pressure >> sell pressure = price will RISE
    
    Returns:
    - pressure_ratio: sell_volume / buy_volume (>1.5 = bearish, <0.67 = bullish)
    - pressure_direction: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    - price_forecast: UP, FLAT, DOWN based on pressure
    """
    
    return df.with_columns([
        # Pressure ratio: sell volume / buy volume
        (
            pl.col("highPriceVolume") / (pl.col("lowPriceVolume") + 1)
        ).alias("pressure_ratio"),
        
        # Net pressure (positive = buying, negative = selling)
        (
            pl.col("lowPriceVolume") - pl.col("highPriceVolume")
        ).alias("net_pressure"),
    ]).with_columns([
        # Classify pressure direction
        pl.when(pl.col("pressure_ratio") >= 2.0)
          .then(pl.lit("🔴 STRONG SELL PRESSURE"))
        .when(pl.col("pressure_ratio") >= 1.3)
          .then(pl.lit("🟠 SELL PRESSURE"))
        .when(pl.col("pressure_ratio") <= 0.5)
          .then(pl.lit("🟢 STRONG BUY PRESSURE"))
        .when(pl.col("pressure_ratio") <= 0.77)
          .then(pl.lit("🟡 BUY PRESSURE"))
        .otherwise(pl.lit("➡️ BALANCED"))
        .alias("pressure_direction"),
        
        # Price forecast based on pressure
        pl.when(pl.col("pressure_ratio") >= 1.5)
          .then(pl.lit("DOWN"))  # Heavy selling = price drops
        .when(pl.col("pressure_ratio") <= 0.67)
          .then(pl.lit("UP"))    # Heavy buying = price rises
        .otherwise(pl.lit("FLAT"))
        .alias("price_forecast"),
    ])


def calculate_market_depth_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Measure how deep the market is (can it absorb your order?)
    
    Logic:
    - Deep market = high volume, tight spread
    - Shallow market = low volume, wide spread
    - Your order size vs market size = impact
    
    Returns:
    - market_depth_score: 0-100 (higher = deeper market)
    - order_impact: LOW, MEDIUM, HIGH (will your order move the price?)
    """
    
    return df.with_columns([
        # Market depth = volume × (1 - spread_pct/100)
        # High volume + tight spread = deep market
        (
            (pl.col("hourly_volume") / 100).log1p() * 20 * (1 - pl.col("spread_pct") / 100)
        ).clip(0, 100).alias("market_depth_score"),
        
        # Order impact: your qty vs hourly volume
        (
            pl.col("allocation_qty") / (pl.col("hourly_volume") + 1) * 100
        ).alias("order_impact_pct"),
    ]).with_columns([
        # Classify order impact
        pl.when(pl.col("order_impact_pct") <= 5)
          .then(pl.lit("🟢 LOW IMPACT"))
        .when(pl.col("order_impact_pct") <= 20)
          .then(pl.lit("🟡 MEDIUM IMPACT"))
        .otherwise(pl.lit("🔴 HIGH IMPACT"))
        .alias("order_impact"),
    ])


def calculate_competitive_pressure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estimate how many other traders are competing for this item
    
    Logic:
    - Popular items (high volume, low spread) = many merchers
    - Niche items (low volume, high spread) = few competitors
    - More competition = harder to get optimal fills
    
    Returns:
    - competition_score: 0-100 (higher = more crowded)
    - competition_level: LOW, MEDIUM, HIGH, EXTREME
    """
    
    return df.with_columns([
        # Competition = high volume with tight spreads (means many active traders)
        # Formula: volume_score * (1 / spread_score)
        (
            (pl.col("hourly_volume") / 1000).log1p() * 15 / (pl.col("spread_pct") + 1)
        ).clip(0, 100).alias("competition_score"),
    ]).with_columns([
        # Classify competition level
        pl.when(pl.col("competition_score") >= 70)
          .then(pl.lit("🔴 EXTREME"))
        .when(pl.col("competition_score") >= 50)
          .then(pl.lit("🟠 HIGH"))
        .when(pl.col("competition_score") >= 30)
          .then(pl.lit("🟡 MEDIUM"))
        .otherwise(pl.lit("🟢 LOW"))
        .alias("competition_level"),
    ])


def adjust_opportunity_for_microstructure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adjust opportunity score based on market microstructure realities
    
    Penalizes:
    - Long time-to-fill
    - Adverse price pressure (selling into sell pressure = bad)
    - High competition
    - Large order impact
    
    Returns:
    - microstructure_adjusted_score: Realistic opportunity score
    """
    
    return df.with_columns([
        # Start with base opportunity score
        pl.col("opportunity_score").alias("base_opportunity_score"),
    ]).with_columns([
        (
            pl.col("base_opportunity_score")
            # Penalize slow fills (>8 hours = -20%)
            * pl.when(pl.col("time_to_fill_hours") > 8).then(0.8).otherwise(1.0)
            # Penalize bad price pressure (selling into sell pressure = -15%)
            * pl.when(
                (pl.col("pressure_direction").str.contains("SELL")) 
            ).then(0.85).otherwise(1.0)
            # Penalize high competition (-10%)
            * pl.when(pl.col("competition_score") >= 70).then(0.9).otherwise(1.0)
            # Penalize high order impact (-15%)
            * pl.when(pl.col("order_impact_pct") > 20).then(0.85).otherwise(1.0)
        ).clip(0, 100).alias("microstructure_adjusted_score")
    ])


def add_fill_warnings(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate warnings for problematic fills
    
    Returns:
    - fill_warning: Text warning about execution risk
    - warning_severity: NONE, LOW, MEDIUM, HIGH, CRITICAL
    """
    
    warnings = []
    
    for row in df.iter_rows(named=True):
        warning_msgs = []
        severity = "NONE"
        
        # Time to fill warning
        if row.get("time_to_fill_hours", 0) > 24:
            warning_msgs.append("⏰ Will take 24+ hours to fill")
            severity = "HIGH"
        elif row.get("time_to_fill_hours", 0) > 8:
            warning_msgs.append("⏰ Will take 8+ hours to fill")
            severity = "MEDIUM" if severity == "NONE" else severity
        
        # Pressure warning
        pressure = row.get("pressure_direction", "")
        if "STRONG SELL" in pressure:
            warning_msgs.append("📉 Heavy selling pressure - price may drop")
            severity = "HIGH"
        elif "SELL" in pressure:
            warning_msgs.append("📉 Selling pressure detected")
            severity = "MEDIUM" if severity == "NONE" else severity
        
        # Competition warning
        if row.get("competition_score", 0) >= 70:
            warning_msgs.append("🏁 EXTREME competition - hard to get fills")
            severity = "MEDIUM" if severity == "NONE" else severity
        
        # Order impact warning
        if row.get("order_impact_pct", 0) > 20:
            warning_msgs.append("📊 Your order is 20%+ of market - will move price")
            severity = "HIGH"
        
        warnings.append({
            "fill_warning": " | ".join(warning_msgs) if warning_msgs else "✅ No major concerns",
            "warning_severity": severity
        })
    
    # Add warnings to dataframe
    warnings_df = pl.DataFrame(warnings)
    return pl.concat([df, warnings_df], how="horizontal")


def analyze_market_microstructure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complete market microstructure analysis pipeline
    
    Adds:
    1. Time-to-fill estimates
    2. Buy/sell pressure analysis
    3. Market depth scoring
    4. Competition analysis
    5. Adjusted opportunity scores
    6. Fill warnings
    """
    
    df = calculate_time_to_fill(df)
    df = calculate_market_pressure(df)
    df = calculate_market_depth_score(df)
    df = calculate_competitive_pressure(df)
    df = adjust_opportunity_for_microstructure(df)
    df = add_fill_warnings(df)
    
    return df
