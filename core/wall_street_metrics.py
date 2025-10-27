"""
Advanced Wall Street Trading Metrics for OSRS
Missing features identified by seasoned traders
"""

import polars as pl
import numpy as np
from datetime import datetime, time
from typing import Dict


def calculate_slot_efficiency(df: pl.DataFrame, ge_slots: int = 8) -> pl.DataFrame:
    """
    Calculate GP/Slot/Hour - the REAL profitability metric
    
    A 10M profit item that ties up a slot for 7 days earns:
    10M / (7 days × 24 hours) = 59.5K GP/hour per slot
    
    A 50K profit item that flips in 1 hour earns:
    50K / 1 hour = 50K GP/hour per slot
    
    But you can flip it 24 times a day = 1.2M/day = better than the 10M item!
    """
    
    df = df.with_columns([
        # Estimate hold time based on strategy
        pl.when(pl.col("strategy_type") == "INSTANT_FLIP")
        .then(pl.lit(2.0))  # 2 hours
        .when(pl.col("strategy_type") == "SHORT_HOLD")
        .then(pl.lit(36.0))  # 1.5 days
        .when(pl.col("strategy_type") == "SWING")
        .then(pl.lit(120.0))  # 5 days
        .when(pl.col("strategy_type") == "WHALE")
        .then(pl.lit(168.0))  # 7 days
        .otherwise(pl.lit(24.0))  # 1 day default
        .alias("expected_hold_hours")
    ])
    
    # Calculate profitability per slot per hour
    df = df.with_columns([
        (pl.col("net_edge") / pl.col("expected_hold_hours")).alias("gp_per_slot_per_hour")
    ])
    
    # Flips per day (how many times can you flip this in 24h)
    df = df.with_columns([
        (24.0 / pl.col("expected_hold_hours")).alias("flips_per_day")
    ])
    
    # Daily profit per slot
    df = df.with_columns([
        (pl.col("net_edge") * pl.col("flips_per_day")).alias("daily_profit_per_slot")
    ])
    
    return df


def calculate_execution_risk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estimate slippage and execution difficulty
    
    High volume = easy to fill at quoted price
    Low volume = might not fill, or worse price
    """
    
    df = df.with_columns([
        # Execution confidence (0-100): Will you fill at these prices?
        pl.when(pl.col("hourly_volume") >= 10000)
        .then(pl.lit(95))  # Very confident
        .when(pl.col("hourly_volume") >= 1000)
        .then(pl.lit(80))  # Confident
        .when(pl.col("hourly_volume") >= 100)
        .then(pl.lit(60))  # Moderate
        .when(pl.col("hourly_volume") >= 10)
        .then(pl.lit(30))  # Low
        .otherwise(pl.lit(10))  # Very low
        .alias("execution_confidence")
    ])
    
    # Expected slippage (% worse than quoted price)
    df = df.with_columns([
        pl.when(pl.col("hourly_volume") >= 5000)
        .then(pl.lit(0.5))  # Minimal slippage
        .when(pl.col("hourly_volume") >= 1000)
        .then(pl.lit(1.0))
        .when(pl.col("hourly_volume") >= 100)
        .then(pl.lit(2.0))
        .when(pl.col("hourly_volume") >= 10)
        .then(pl.lit(5.0))
        .otherwise(pl.lit(10.0))  # High slippage
        .alias("expected_slippage_pct")
    ])
    
    # Adjusted profit after slippage
    df = df.with_columns([
        (pl.col("net_edge") * (1 - pl.col("expected_slippage_pct") / 100)).alias("slippage_adjusted_profit")
    ])
    
    return df


def detect_manipulation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect potential market manipulation
    
    Red flags:
    - Huge volume but tiny spread = wash trading
    - Volume imbalance >90% one-sided = fake volume
    - Spread >30% with high volume = pump & dump setup
    """
    
    df = df.with_columns([
        # Volume imbalance ratio (0 = perfect balance, 1 = totally one-sided)
        (
            abs(pl.col("highPriceVolume") - pl.col("lowPriceVolume")) /
            (pl.col("hourly_volume") + 1)
        ).alias("volume_imbalance_ratio")
    ])
    
    # Detect manipulation flags
    df = df.with_columns([
        pl.when(
            (pl.col("volume_imbalance_ratio") > 0.9) &
            (pl.col("hourly_volume") > 1000)
        )
        .then(pl.lit("WASH_TRADING"))
        .when(
            (pl.col("spread_pct") > 30) &
            (pl.col("hourly_volume") > 500)
        )
        .then(pl.lit("PUMP_DUMP_RISK"))
        .when(
            (pl.col("spread_pct") < 1.0) &
            (pl.col("hourly_volume") > 10000) &
            (pl.col("volume_imbalance_ratio") > 0.8)
        )
        .then(pl.lit("SUSPICIOUS_VOLUME"))
        .otherwise(pl.lit("CLEAN"))
        .alias("manipulation_flag")
    ])
    
    return df


def calculate_downside_risk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate downside-specific risk (Sortino ratio concept)
    
    We care more about LOSSES than gains
    """
    
    # Estimate maximum realistic loss if trade goes bad
    df = df.with_columns([
        # If you can't sell, you lose the buy price + tax if you panic sell
        # Assume worst case: -5% to -20% depending on liquidity
        pl.when(pl.col("hourly_volume") >= 5000)
        .then(pl.lit(5.0))  # Very liquid, minimal loss
        .when(pl.col("hourly_volume") >= 1000)
        .then(pl.lit(8.0))
        .when(pl.col("hourly_volume") >= 100)
        .then(pl.lit(12.0))
        .when(pl.col("hourly_volume") >= 10)
        .then(pl.lit(15.0))
        .otherwise(pl.lit(20.0))  # Illiquid, big potential loss
        .alias("max_downside_pct")
    ])
    
    # Downside risk in GP
    df = df.with_columns([
        (pl.col("avgLowPrice") * pl.col("max_downside_pct") / 100).alias("max_loss_per_item")
    ])
    
    # Risk-adjusted return (profit / max_loss ratio)
    df = df.with_columns([
        (pl.col("net_edge") / (pl.col("max_loss_per_item") + 1)).alias("profit_to_risk_ratio")
    ])
    
    return df


def get_time_of_day_factor() -> float:
    """
    Return trading volume multiplier based on time of day
    
    OSRS is international but peaks during:
    - US evenings (8pm-11pm EST)
    - UK evenings (7pm-10pm GMT)
    - Weekends
    """
    now = datetime.now()
    hour = now.hour
    is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6
    
    # Peak hours: 7pm-11pm (19-23)
    if 19 <= hour <= 23:
        volume_multiplier = 1.3  # 30% more volume
    # Good hours: 2pm-7pm and 11pm-1am
    elif (14 <= hour <= 18) or (23 <= hour <= 1):
        volume_multiplier = 1.1  # 10% more
    # Off-peak: 2am-10am
    elif 2 <= hour <= 10:
        volume_multiplier = 0.7  # 30% less
    else:
        volume_multiplier = 1.0  # Normal
    
    # Weekend bonus
    if is_weekend:
        volume_multiplier *= 1.2  # 20% more on weekends
    
    return volume_multiplier


def calculate_true_opportunity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Wall Street-grade opportunity score considering ALL factors:
    - Slot efficiency (GP/slot/hour)
    - Execution risk
    - Manipulation flags
    - Downside protection
    - Time of day
    """
    
    # Add all the new metrics
    df = calculate_slot_efficiency(df)
    df = calculate_execution_risk(df)
    df = detect_manipulation(df)
    df = calculate_downside_risk(df)
    
    # Get time adjustment
    time_factor = get_time_of_day_factor()
    
    # Recalculate opportunity score with new factors
    df = df.with_columns([
        (
            # Slot efficiency (30%): GP per slot per hour
            (pl.col("gp_per_slot_per_hour") / 10000).clip(0, 30) +
            
            # Execution confidence (20%): Can you actually fill this?
            (pl.col("execution_confidence") / 100 * 20) +
            
            # Profit/Risk ratio (25%): Upside vs downside
            (pl.col("profit_to_risk_ratio").clip(0, 5) / 5 * 25) +
            
            # Original profit quality (15%)
            (pl.col("profit_quality") / 100 * 15) +
            
            # Inverse risk (10%)
            ((100 - pl.col("risk_score")) / 100 * 10)
        ).cast(pl.Int32).alias("wall_street_score")
    ])
    
    # Penalize manipulation
    df = df.with_columns([
        pl.when(pl.col("manipulation_flag") != "CLEAN")
        .then(pl.col("wall_street_score") * 0.5)  # 50% penalty
        .otherwise(pl.col("wall_street_score"))
        .cast(pl.Int32)
        .alias("wall_street_score")
    ])
    
    return df


def get_portfolio_correlation_warning(items: list) -> Dict:
    """
    Check if portfolio is too correlated (all same item type)
    
    Example: Don't buy only dragonhide - if dragonhide market crashes, you're screwed
    """
    
    item_names = [item['name'].lower() for item in items]
    
    # Common correlated groups
    correlations = {
        'dragonhide': ['green d\'hide', 'blue d\'hide', 'red d\'hide', 'black d\'hide'],
        'dragon_bones': ['dragon bones', 'superior dragon bones', 'wyvern bones'],
        'runes': ['blood rune', 'death rune', 'soul rune', 'wrath rune'],
        'logs': ['magic logs', 'yew logs', 'redwood logs'],
        'ore': ['adamantite ore', 'runite ore', 'coal'],
    }
    
    for category, related_items in correlations.items():
        count = sum(1 for name in item_names if any(r in name for r in related_items))
        if count >= 3:
            return {
                'warning': True,
                'category': category.replace('_', ' ').title(),
                'count': count,
                'message': f"⚠️ {count} items are {category.replace('_', ' ')}! Diversify across categories."
            }
    
    return {'warning': False}
