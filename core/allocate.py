"""
Capital allocation engine for OSRS trading
Implements greedy allocation respecting GE limits and liquidity
"""

import polars as pl
from typing import Dict, List


def allocate_capital(
    df: pl.DataFrame,
    total_capital_gp: int,
    max_pct_per_item: float = 20.0,
    max_items: int = 10,
) -> pl.DataFrame:
    """
    Allocate capital across top-ranked items
    
    Greedy algorithm:
        1. Iterate through items sorted by rank_score (descending)
        2. For each item:
            - Max allocation = min(
                max_pct_per_item% of capital,
                GE buy limit * avgLowPrice,
                hourly_volume * avgLowPrice (don't exceed liquidity)
              )
            - Allocate up to remaining capital
        3. Stop when max_items reached or capital exhausted
    
    Args:
        df: DataFrame with trading features (must be sorted by rank_score desc)
        total_capital_gp: Total GP available to invest
        max_pct_per_item: Maximum % of capital per item (default 20%)
        max_items: Maximum number of concurrent positions (default 10)
    
    Returns:
        DataFrame with added columns:
            - allocation_gp: GP allocated to this item
            - allocation_qty: Number of items to buy
            - allocation_pct: Percentage of total capital
    """
    # Initialize allocation columns
    allocations = []
    remaining_capital = total_capital_gp
    max_capital_per_item = total_capital_gp * (max_pct_per_item / 100)
    
    for row in df.iter_rows(named=True):
        if len(allocations) >= max_items:
            break
        
        if remaining_capital <= 0:
            break
        
        # Calculate maximum possible allocation for this item
        buy_price = row["avgLowPrice"]
        
        # Constraint 1: Max % of total capital
        max_by_pct = max_capital_per_item
        
        # Constraint 2: GE buy limit
        ge_limit = row.get("limit", float('inf'))
        max_by_limit = ge_limit * buy_price if ge_limit > 0 else float('inf')
        
        # Constraint 3: Strategy-aware quantity calculation
        # Different strategies have different volume requirements
        hourly_volume = row["hourly_volume"]
        strategy_type = row.get("strategy_type", "UNKNOWN")
        
        # Adjust volume factor based on strategy
        if strategy_type == "INSTANT_FLIP":
            # Need high volume for instant flips
            if hourly_volume >= 50000:
                volume_factor = 0.8  # Very safe
            elif hourly_volume >= 10000:
                volume_factor = 0.5  # Safe
            else:
                volume_factor = 0.15  # Risky for instant flips
        
        elif strategy_type == "SHORT_HOLD":
            # 1-2 day holds - volume less critical
            if hourly_volume >= 20000:
                volume_factor = 0.7  # Good volume
            elif hourly_volume >= 5000:
                volume_factor = 0.5  # Acceptable
            else:
                volume_factor = 0.3  # Still workable for short holds
        
        elif strategy_type == "SWING":
            # Week-long holds - volume even less important
            if hourly_volume >= 10000:
                volume_factor = 0.7  # Plenty of time to sell
            elif hourly_volume >= 2000:
                volume_factor = 0.6  # Will sell over the week
            else:
                volume_factor = 0.4  # Still ok for swing trades
        
        else:
            # Default/BARGAIN/OTHER - conservative but not too strict
            if hourly_volume >= 50000:
                volume_factor = 0.8
            elif hourly_volume >= 10000:
                volume_factor = 0.5
            elif hourly_volume >= 5000:
                volume_factor = 0.3
            else:
                volume_factor = 0.2
        
        max_by_liquidity = (ge_limit * volume_factor * buy_price) if ge_limit > 0 else (hourly_volume * buy_price)
        
        # Constraint 4: Remaining capital
        max_by_capital = remaining_capital
        
        # Take minimum of all constraints
        max_allocation = min(
            max_by_pct,
            max_by_limit,
            max_by_liquidity,
            max_by_capital,
        )
        
        # Calculate quantity and actual allocation
        qty = int(max_allocation / buy_price) if buy_price > 0 else 0
        actual_allocation = qty * buy_price
        
        # Only add if allocation is meaningful
        if qty > 0:
            allocations.append({
                **row,
                "allocation_gp": actual_allocation,
                "allocation_qty": qty,
                "allocation_pct": (actual_allocation / total_capital_gp) * 100,
            })
            remaining_capital -= actual_allocation
    
    # Create new dataframe with allocations
    if allocations:
        allocated_df = pl.DataFrame(allocations)
    else:
        # Return empty dataframe with expected schema
        allocated_df = df.head(0).with_columns([
            pl.lit(0).cast(pl.Int64).alias("allocation_gp"),
            pl.lit(0).cast(pl.Int64).alias("allocation_qty"),
            pl.lit(0.0).alias("allocation_pct"),
        ])
    
    return allocated_df


def calculate_portfolio_stats(allocated_df: pl.DataFrame, total_capital: int) -> Dict[str, float]:
    """
    Calculate portfolio-level statistics
    
    Returns:
        - total_allocated: Total GP allocated
        - total_unallocated: Remaining GP
        - allocation_rate: % of capital deployed
        - num_positions: Number of items in portfolio
        - avg_edge_pct: Weighted average edge %
        - total_potential_profit: Sum of (net_edge * allocation_qty)
    """
    if len(allocated_df) == 0:
        return {
            "total_allocated": 0,
            "total_unallocated": total_capital,
            "allocation_rate": 0.0,
            "num_positions": 0,
            "avg_edge_pct": 0.0,
            "total_potential_profit": 0,
        }
    
    total_allocated = allocated_df["allocation_gp"].sum()
    total_unallocated = total_capital - total_allocated
    allocation_rate = (total_allocated / total_capital) * 100 if total_capital > 0 else 0
    num_positions = len(allocated_df)
    
    # Weighted average edge
    total_edge_weighted = (allocated_df["edge_pct"] * allocated_df["allocation_gp"]).sum()
    avg_edge_pct = total_edge_weighted / total_allocated if total_allocated > 0 else 0
    
    # Potential profit (if all items flip successfully)
    total_potential_profit = (allocated_df["net_edge"] * allocated_df["allocation_qty"]).sum()
    
    return {
        "total_allocated": int(total_allocated),
        "total_unallocated": int(total_unallocated),
        "allocation_rate": allocation_rate,
        "num_positions": num_positions,
        "avg_edge_pct": avg_edge_pct,
        "total_potential_profit": int(total_potential_profit),
    }
