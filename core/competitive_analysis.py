"""
Competitive Analysis - Find the Blue Ocean Trades
Identifies which items have less competition and are easier to profit from
"""

import polars as pl
from typing import Dict, List


def calculate_competition_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate how competitive/crowded each trade is
    
    High competition indicators:
    - Very high volume (lots of traders active)
    - Very tight spread (bots competing)
    - Popular items (Dragon bones, etc.)
    
    Low competition indicators:
    - Moderate volume (enough liquidity but not crowded)
    - Wider spreads (less bot competition)
    - Niche items
    
    Returns:
    - competition_score: 0-100 (higher = MORE crowded)
    - competition_level: BLUE_OCEAN, LOW, MODERATE, HIGH, EXTREME
    - ease_of_profit: VERY_EASY, EASY, MODERATE, HARD, VERY_HARD
    """
    
    # Calculate competition score in one expression
    df = df.with_columns([
        # Sum up competition indicators
        (
            (pl.col("hourly_volume") / 10000).clip(0, 1) * 40 +
            ((10 - pl.col("spread_pct").clip(0, 10)) / 10) * 30 +
            (100 - pl.col("confidence_score")) * 0.3
        ).clip(0, 100).cast(pl.Int32).alias("competition_score")
    ])
    
    # Categorize competition
    df = df.with_columns([
        pl.when(pl.col("competition_score") <= 20)
          .then(pl.lit("🏝️ BLUE OCEAN"))  # Perfect! Low competition
        .when(pl.col("competition_score") <= 40)
          .then(pl.lit("🟢 LOW"))  # Good, not crowded
        .when(pl.col("competition_score") <= 60)
          .then(pl.lit("🟡 MODERATE"))  # Some competition
        .when(pl.col("competition_score") <= 80)
          .then(pl.lit("🟠 HIGH"))  # Crowded
        .otherwise(pl.lit("🔴 EXTREME"))  # Very crowded, many bots
        .alias("competition_level")
    ])
    
    # Add ease of profit
    df = df.with_columns([
        # Ease of profit (inverse of competition)
        pl.when(pl.col("competition_score") <= 20)
          .then(pl.lit("⭐⭐⭐ VERY EASY"))
        .when(pl.col("competition_score") <= 40)
          .then(pl.lit("⭐⭐ EASY"))
        .when(pl.col("competition_score") <= 60)
          .then(pl.lit("⭐ MODERATE"))
        .when(pl.col("competition_score") <= 80)
          .then(pl.lit("💀 HARD"))
        .otherwise(pl.lit("☠️ VERY HARD"))
        .alias("ease_of_profit")
    ])
    
    return df


def detect_bot_activity(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect likely bot activity on items
    
    Bot indicators:
    - Extremely high volume (>50k/hr)
    - Perfect spread (exactly 1 GP difference)
    - Volume exactly 50/50 buy/sell
    
    Returns:
    - bot_risk: 0-100 (higher = more likely bots)
    - bot_warning: None, LOW, MEDIUM, HIGH, EXTREME
    """
    
    df = df.with_columns([
        # Bot risk calculation
        (
            # Very high volume indicator
            pl.when(pl.col("hourly_volume") >= 50000)
              .then(30)
              .otherwise(0) +
            
            # Perfect 1 GP spread (classic bot behavior)
            pl.when(pl.col("spread") == 1)
              .then(40)
              .otherwise(0) +
            
            # Perfect 50/50 volume balance (suspicious)
            pl.when(pl.col("confidence_score").is_between(95, 100))
              .then(30)
              .otherwise(0)
        ).clip(0, 100).cast(pl.Int32).alias("bot_risk_score"),
    ])
    
    # Categorize bot risk
    df = df.with_columns([
        pl.when(pl.col("bot_risk_score") == 0)
          .then(pl.lit(None))
        .when(pl.col("bot_risk_score") <= 30)
          .then(pl.lit("🟡 LOW BOT RISK"))
        .when(pl.col("bot_risk_score") <= 50)
          .then(pl.lit("🟠 MEDIUM BOT RISK"))
        .when(pl.col("bot_risk_score") <= 70)
          .then(pl.lit("🔴 HIGH BOT RISK"))
        .otherwise(pl.lit("☠️ EXTREME BOT RISK"))
        .alias("bot_warning")
    ])
    
    return df


def find_undertraded_opportunities(df: pl.DataFrame, min_opportunity: int = 60) -> List[Dict]:
    """
    Find high-quality opportunities with LOW competition
    
    The "blue ocean" trades - good profit, but not many people trading them
    
    Args:
        df: Trading dataframe with competition scores
        min_opportunity: Minimum opportunity score to consider
        
    Returns:
        List of undertraded opportunities sorted by opportunity score
    """
    
    # Filter for good opportunities with low competition
    blue_ocean = df.filter(
        (pl.col("opportunity_score") >= min_opportunity) &
        (pl.col("competition_score") <= 40)  # Low to moderate competition
    ).sort("opportunity_score", descending=True)
    
    results = []
    for row in blue_ocean.head(10).iter_rows(named=True):
        results.append({
            "name": row["name"],
            "opportunity_score": int(row["opportunity_score"]),
            "competition_score": int(row["competition_score"]),
            "competition_level": row["competition_level"],
            "ease_of_profit": row["ease_of_profit"],
            "profit_per_item": int(row["net_edge"]),
            "roi_pct": row["edge_pct"],
            "hourly_volume": int(row["hourly_volume"]),
        })
    
    return results


def analyze_competition(df: pl.DataFrame) -> pl.DataFrame:
    """
    Complete competitive analysis pipeline
    
    Adds all competition-related metrics to dataframe
    """
    df = calculate_competition_score(df)
    df = detect_bot_activity(df)
    
    return df
