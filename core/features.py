"""
Feature engineering for OSRS trading analytics
Computes momentum, liquidity, volatility, edge, and ranking scores
"""

import polars as pl
import numpy as np
from typing import Dict, Any

from .config import (
    GE_TAX_RATE,
    MIN_PRICE,
    MIN_HOURLY_VOLUME,
    MIN_SPREAD_PCT,
    MAX_SPREAD_PCT,
    MIN_PROFIT_PER_FLIP,
    RANK_WEIGHTS,
)


def parse_mapping_data(mapping_data) -> pl.DataFrame:
    """
    Convert mapping API response to Polars DataFrame
    
    Args:
        mapping_data: Response from /mapping endpoint (list or dict)
    
    Returns:
        DataFrame with columns: item_id, name, limit, highalch, icon, members
    """
    items = []
    
    # Handle list format (direct API response)
    if isinstance(mapping_data, list):
        for item_info in mapping_data:
            items.append({
                "item_id": item_info.get("id"),
                "name": item_info.get("name", "Unknown"),
                "limit": item_info.get("limit", 0),
                "highalch": item_info.get("highalch", 0),
                "icon": item_info.get("icon", ""),
                "members": item_info.get("members", False),
            })
    # Handle dict format (keyed by ID)
    else:
        for item_id_str, item_info in mapping_data.items():
            items.append({
                "item_id": int(item_id_str),
                "name": item_info.get("name", "Unknown"),
                "limit": item_info.get("limit", 0),
                "highalch": item_info.get("highalch", 0),
                "icon": item_info.get("icon", ""),
                "members": item_info.get("members", False),
            })
    
    return pl.DataFrame(items)


def parse_hourly_data(hourly_data: Dict[str, Any]) -> pl.DataFrame:
    """
    Convert 1h API response to Polars DataFrame
    
    Args:
        hourly_data: Response from /1h endpoint
    
    Returns:
        DataFrame with columns: item_id, avgHighPrice, highPriceVolume, avgLowPrice, lowPriceVolume
    """
    items = []
    data_dict = hourly_data.get("data", {})
    
    for item_id_str, price_info in data_dict.items():
        # Skip items with missing price data
        if not price_info:
            continue
            
        avg_high = price_info.get("avgHighPrice")
        avg_low = price_info.get("avgLowPrice")
        
        # Skip if either price is missing
        if avg_high is None or avg_low is None:
            continue
        
        items.append({
            "item_id": int(item_id_str),
            "avgHighPrice": avg_high,
            "highPriceVolume": price_info.get("highPriceVolume", 0),
            "avgLowPrice": avg_low,
            "lowPriceVolume": price_info.get("lowPriceVolume", 0),
        })
    
    return pl.DataFrame(items)


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute all trading features from price data
    
    Expected input columns:
        - item_id, name, limit, avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume
    
    Adds columns:
        - mid_price: average of high and low
        - spread: difference between high and low
        - spread_pct: spread as percentage of mid
        - hourly_volume: total traded volume
        - ge_tax: GE tax on selling at avgHigh
        - net_edge: profit after tax
        - edge_pct: edge as percentage of buy price
    """
    return df.with_columns([
        # Mid price
        ((pl.col("avgHighPrice") + pl.col("avgLowPrice")) / 2).alias("mid_price"),
        
        # Spread
        (pl.col("avgHighPrice") - pl.col("avgLowPrice")).alias("spread"),
        
        # Spread percentage
        (
            (pl.col("avgHighPrice") - pl.col("avgLowPrice")) / 
            ((pl.col("avgHighPrice") + pl.col("avgLowPrice")) / 2) * 100
        ).alias("spread_pct"),
        
        # Total hourly volume
        (pl.col("highPriceVolume") + pl.col("lowPriceVolume")).alias("hourly_volume"),
        
        # Calculate GE tax - IMPORTANT: Items under 100 GP have NO TAX!
        pl.when(pl.col("avgHighPrice") >= 100)
          .then(pl.col("avgHighPrice") * GE_TAX_RATE)
          .otherwise(0)
          .alias("ge_tax"),
        
        # Net edge after tax
        pl.when(pl.col("avgHighPrice") >= 100)
          .then((pl.col("avgHighPrice") * (1 - GE_TAX_RATE)) - pl.col("avgLowPrice"))
          .otherwise(pl.col("avgHighPrice") - pl.col("avgLowPrice"))
          .alias("net_edge"),
        
        # Edge as percentage of buy price
        pl.when(pl.col("avgHighPrice") >= 100)
          .then(((pl.col("avgHighPrice") * (1 - GE_TAX_RATE)) - pl.col("avgLowPrice")) / pl.col("avgLowPrice") * 100)
          .otherwise((pl.col("avgHighPrice") - pl.col("avgLowPrice")) / pl.col("avgLowPrice") * 100)
          .alias("edge_pct"),
        
        # Buy and sell prices (for actionable trading)
        pl.col("avgLowPrice").alias("buy_price"),
        pl.when(pl.col("avgHighPrice") >= 100)
          .then((pl.col("avgHighPrice") * (1 - GE_TAX_RATE)).cast(pl.Int64))
          .otherwise(pl.col("avgHighPrice").cast(pl.Int64))
          .alias("sell_price"),
    ])


def compute_momentum(
    df: pl.DataFrame,
    window_hours: int = 6,
) -> pl.DataFrame:
    """
    Compute price momentum (% change over window)
    
    Note: This requires historical data. For now, we'll use a placeholder.
    In production, you'd fetch /timeseries data for each item.
    """
    # Placeholder: set momentum to 0 for now
    # Real implementation would compare current vs N hours ago
    return df.with_columns([
        pl.lit(0.0).alias("momentum_pct")
    ])


def compute_volatility(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute volatility (coefficient of variation)
    
    Note: This requires historical data. For now, we'll use spread as proxy.
    """
    # Use spread_pct as a simple volatility proxy
    return df.with_columns([
        (pl.col("spread_pct") / 100).alias("volatility")
    ])


def compute_rank_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute composite ranking score based on weighted features
    
    Score = 0.45*edge_pct + 0.25*log(liquidity) + 0.20*momentum - 0.10*volatility
    """
    # Handle empty dataframe
    if len(df) == 0:
        return df.with_columns([
            pl.lit(0.0).alias("rank_score")
        ])
    
    # Normalize log liquidity to 0-100 scale for better weighting
    max_vol = df["hourly_volume"].max()
    max_log_liq = np.log1p(max_vol if max_vol is not None else 1)
    
    return df.with_columns([
        (
            RANK_WEIGHTS["edge_pct"] * pl.col("edge_pct") +
            RANK_WEIGHTS["log_liquidity"] * (pl.col("hourly_volume").log1p() / max_log_liq * 100) +
            RANK_WEIGHTS["momentum"] * pl.col("momentum_pct") +
            RANK_WEIGHTS["volatility"] * pl.col("volatility") * 100
        ).alias("rank_score")
    ])


def filter_tradeable_items(df: pl.DataFrame, min_profit_per_flip: int = None) -> pl.DataFrame:
    """
    Filter out non-tradeable, illiquid, or low-value items
    
    Args:
        df: Trading dataframe
        min_profit_per_flip: Minimum GP profit per single item flip (overrides default)
    """
    from core.config import MIN_EDGE_PCT
    
    min_profit = min_profit_per_flip if min_profit_per_flip is not None else MIN_PROFIT_PER_FLIP
    
    return df.filter(
        (pl.col("mid_price") >= MIN_PRICE) &  # High-value items only
        (pl.col("hourly_volume") >= MIN_HOURLY_VOLUME) &  # Actual liquidity
        (pl.col("spread_pct") >= MIN_SPREAD_PCT) &  # Minimum spread
        (pl.col("spread_pct") <= MAX_SPREAD_PCT) &  # Reasonable spread
        (pl.col("edge_pct") >= MIN_EDGE_PCT) &  # Minimum profit margin %
        (pl.col("net_edge") > 0) &  # Only positive edges
        (pl.col("net_edge") >= min_profit) &  # Minimum profit per flip
        (pl.col("highPriceVolume") >= MIN_HOURLY_VOLUME / 4) &  # Some buy volume
        (pl.col("lowPriceVolume") >= MIN_HOURLY_VOLUME / 4)  # Some sell volume
    )


def build_trading_dataframe(
    mapping_data: Dict[str, Any],
    hourly_data: Dict[str, Any],
    min_profit_per_flip: int = None,
) -> pl.DataFrame:
    """
    Build complete trading analysis DataFrame from API data
    
    Pipeline:
        1. Parse mapping and hourly data
        2. Join on item_id
        3. Compute features
        4. Compute momentum & volatility
        5. Filter tradeable items
        6. Compute rank score
        7. Sort by rank_score descending
    
    Args:
        mapping_data: Item metadata from /mapping
        hourly_data: Price data from /1h
        min_profit_per_flip: Override minimum GP profit filter
    """
    # Parse raw data
    mapping_df = parse_mapping_data(mapping_data)
    hourly_df = parse_hourly_data(hourly_data)
    
    # Join mapping + prices
    df = mapping_df.join(hourly_df, on="item_id", how="inner")
    
    # Compute features
    df = compute_features(df)
    df = compute_momentum(df)
    df = compute_volatility(df)
    
    # Add volume confidence score (0-100)
    df = df.with_columns([
        # Volume balance: how balanced are buy/sell volumes? (closer to 50/50 = better)
        (
            100 - abs(
                (pl.col("highPriceVolume") / (pl.col("hourly_volume") + 1e-9) * 100) - 50
            ) * 2
        ).clip(0, 100).alias("volume_balance"),
        
        # Liquidity score: higher volume = more confidence
        (
            (pl.col("hourly_volume") / 1000).log1p() * 20
        ).clip(0, 100).alias("liquidity_score"),
    ]).with_columns([
        # Overall confidence: average of balance and liquidity
        ((pl.col("volume_balance") + pl.col("liquidity_score")) / 2).cast(pl.Int32).alias("confidence_score")
    ])
    
    # Filter
    df = filter_tradeable_items(df, min_profit_per_flip)
    
    # Rank
    df = compute_rank_score(df)
    
    # Add advanced analytics
    from core.advanced_analytics import build_advanced_analytics
    df = build_advanced_analytics(df)
    
    # Sort by opportunity score (advanced ranking)
    df = df.sort("opportunity_score", descending=True)
    
    return df
