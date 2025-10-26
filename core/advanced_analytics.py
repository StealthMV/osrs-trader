"""
Advanced Trading Analytics for OSRS
Multi-timeframe analysis, risk scoring, and market intelligence
"""

import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_risk_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate comprehensive risk score (0-100, lower = less risky)
    
    Factors:
    - Spread volatility (wide spread = risky)
    - Volume consistency (erratic volume = risky)
    - Price volatility (high volatility = risky)
    - Market depth imbalance (one-sided = risky)
    """
    df = df.with_columns([
        # Spread risk (0-25): normalized spread %
        (pl.col("spread_pct") / 50 * 25).clip(0, 25).alias("spread_risk"),
        
        # Volume imbalance risk (0-25): how imbalanced are buy/sell volumes
        (
            (abs(pl.col("highPriceVolume") - pl.col("lowPriceVolume")) / 
             (pl.col("hourly_volume") + 1)) * 25
        ).clip(0, 25).alias("volume_imbalance_risk"),
        
        # Liquidity risk (0-25): inverse of volume (low volume = high risk)
        (25 - (pl.col("hourly_volume") / 500).clip(0, 25)).alias("liquidity_risk"),
        
        # Volatility risk (0-25): use spread as proxy
        (pl.col("volatility") * 25).clip(0, 25).alias("volatility_risk"),
    ]).with_columns([
        # Total risk score
        (
            pl.col("spread_risk") + 
            pl.col("volume_imbalance_risk") + 
            pl.col("liquidity_risk") + 
            pl.col("volatility_risk")
        ).cast(pl.Int32).alias("risk_score")
    ])
    
    return df


def calculate_sharpe_ratio(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate risk-adjusted return (Sharpe-like ratio)
    
    Sharpe = Expected Return / Risk
    Higher is better
    """
    df = df.with_columns([
        # Risk-adjusted return: edge_pct / (risk_score + 1)
        (pl.col("edge_pct") / (pl.col("risk_score") + 1) * 10).alias("sharpe_ratio")
    ])
    
    return df


def calculate_market_cap(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate market capitalization (price × volume)"""
    df = df.with_columns([
        (pl.col("mid_price") * pl.col("hourly_volume")).alias("market_cap")
    ])
    
    return df


def calculate_velocity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate market velocity (turnover speed)
    Higher = more active trading
    """
    df = df.with_columns([
        # Velocity = volume / (price × spread_pct)
        (
            pl.col("hourly_volume") / 
            ((pl.col("mid_price") * pl.col("spread_pct") / 100) + 1)
        ).log1p().alias("velocity_score")
    ])
    
    return df


def detect_market_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect market regime for each item:
    - TRENDING: Price momentum exists
    - MEAN_REVERTING: Price oscillating
    - CHOPPY: No clear pattern
    """
    df = df.with_columns([
        pl.when(pl.col("momentum_pct").abs() > 2.0)
        .then(pl.lit("TRENDING"))
        .when(pl.col("momentum_pct").abs() < 0.5)
        .then(pl.lit("MEAN_REVERTING"))
        .otherwise(pl.lit("CHOPPY"))
        .alias("market_regime")
    ])
    
    return df


def calculate_profit_quality(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assess profit quality (0-100)
    Combines spread sustainability, volume support, and edge consistency
    """
    df = df.with_columns([
        # Quality score components
        (
            # Volume support (40%): higher volume = more sustainable
            (pl.col("hourly_volume") / 1000).log1p().clip(0, 40) +
            
            # Spread tightness (30%): tighter spread = more reliable
            (30 - pl.col("spread_pct").clip(0, 30)) +
            
            # Balance (30%): balanced buy/sell = more stable
            (pl.col("confidence_score") / 100 * 30)
        ).cast(pl.Int32).alias("profit_quality")
    ])
    
    return df


def calculate_opportunity_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Master opportunity score combining all factors
    0-100, higher = better opportunity
    """
    df = df.with_columns([
        (
            # Profit component (40%)
            (pl.col("edge_pct") * 4).clip(0, 40) +
            
            # Risk component (30%): inverse risk
            ((100 - pl.col("risk_score")) / 100 * 30) +
            
            # Quality component (30%)
            (pl.col("profit_quality") / 100 * 30)
        ).cast(pl.Int32).alias("opportunity_score")
    ])
    
    return df


def identify_strategy_fit(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identify which trading strategy fits each item best:
    - SCALP: Ultra-high volume, tight spreads, low profit per item
    - SWING: High margin, moderate volume, hold overnight
    - WHALE: Very high profit per item, lower volume, big tickets
    - ARBITRAGE: Tight spreads, guaranteed profit, high volume
    """
    df = df.with_columns([
        pl.when(
            (pl.col("hourly_volume") >= 200) & 
            (pl.col("spread_pct") <= 3.0) &
            (pl.col("net_edge") >= 100)
        )
        .then(pl.lit("⚡ SCALP"))
        .when(
            (pl.col("edge_pct") >= 5.0) &
            (pl.col("hourly_volume") >= 30)
        )
        .then(pl.lit("📈 SWING"))
        .when(
            (pl.col("net_edge") >= 100_000) &
            (pl.col("mid_price") >= 1_000_000)
        )
        .then(pl.lit("🐋 WHALE"))
        .when(
            (pl.col("spread_pct") <= 2.0) &
            (pl.col("hourly_volume") >= 100) &
            (pl.col("confidence_score") >= 70)
        )
        .then(pl.lit("🔄 ARBITRAGE"))
        .otherwise(pl.lit("📊 STANDARD"))
        .alias("strategy_type")
    ])
    
    return df


def calculate_kelly_criterion(df: pl.DataFrame, total_capital: int) -> pl.DataFrame:
    """
    Calculate optimal position size using Kelly Criterion
    Kelly % = (Edge * Win Rate - Loss Rate) / Edge
    
    Simplified: Use edge_pct and confidence_score as proxies
    """
    df = df.with_columns([
        # Estimate win rate from confidence score
        (pl.col("confidence_score") / 100).alias("win_rate"),
    ]).with_columns([
        # Kelly fraction (capped at 25% for safety)
        (
            (pl.col("edge_pct") / 100 * pl.col("win_rate") - (1 - pl.col("win_rate"))) /
            (pl.col("edge_pct") / 100)
        ).clip(0, 0.25).alias("kelly_fraction")
    ]).with_columns([
        # Recommended allocation
        (pl.col("kelly_fraction") * total_capital).cast(pl.Int64).alias("kelly_allocation")
    ])
    
    return df


def rank_by_expected_value(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate expected value considering win rate and risk
    EV = (Profit × Win Rate) - (Loss × Loss Rate)
    """
    df = df.with_columns([
        # Expected value per flip
        (
            (pl.col("net_edge") * pl.col("win_rate")) -
            (pl.col("avgLowPrice") * 0.02 * (1 - pl.col("win_rate")))  # Assume 2% loss on failed trades
        ).alias("expected_value")
    ]).with_columns([
        # Expected value per hour
        (pl.col("expected_value") * pl.col("hourly_volume")).alias("expected_value_per_hour")
    ])
    
    return df


def build_advanced_analytics(df: pl.DataFrame, total_capital: int = 50_000_000) -> pl.DataFrame:
    """
    Apply all advanced analytics to trading dataframe
    
    Pipeline:
    1. Risk scoring
    2. Sharpe ratio
    3. Market cap & velocity
    4. Market regime detection
    5. Profit quality
    6. Opportunity scoring
    7. Strategy identification
    8. Kelly Criterion sizing
    9. Expected value ranking
    """
    df = calculate_market_cap(df)
    df = calculate_velocity_score(df)
    df = calculate_risk_score(df)
    df = calculate_sharpe_ratio(df)
    df = detect_market_regime(df)
    df = calculate_profit_quality(df)
    df = calculate_opportunity_score(df)
    df = identify_strategy_fit(df)
    df = calculate_kelly_criterion(df, total_capital)
    df = rank_by_expected_value(df)
    
    return df


def get_top_opportunities_by_strategy(df: pl.DataFrame, strategy: str, n: int = 10) -> pl.DataFrame:
    """Get top N opportunities filtered by strategy type"""
    if strategy == "ALL":
        return df.sort("opportunity_score", descending=True).head(n)
    else:
        return df.filter(
            pl.col("strategy_type") == strategy
        ).sort("opportunity_score", descending=True).head(n)


def calculate_portfolio_metrics(df: pl.DataFrame) -> Dict:
    """Calculate advanced portfolio-level metrics"""
    if len(df) == 0:
        return {}
    
    total_ev = df["expected_value_per_hour"].sum()
    avg_sharpe = df["sharpe_ratio"].mean()
    avg_risk = df["risk_score"].mean()
    total_market_cap = df["market_cap"].sum()
    
    # Strategy distribution
    strategy_counts = df.group_by("strategy_type").agg(pl.count()).sort("count", descending=True)
    
    return {
        "total_expected_value_per_hour": int(total_ev) if total_ev else 0,
        "average_sharpe_ratio": float(avg_sharpe) if avg_sharpe else 0.0,
        "average_risk_score": float(avg_risk) if avg_risk else 0.0,
        "total_market_cap": int(total_market_cap) if total_market_cap else 0,
        "strategy_distribution": strategy_counts.to_dicts() if len(strategy_counts) > 0 else [],
    }
