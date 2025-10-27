"""
Day-of-Week Trading Patterns for OSRS
Analyzes when to buy low and sell high based on weekly patterns
"""

import polars as pl
from datetime import datetime, timedelta
from typing import Dict, Tuple


def analyze_day_of_week_patterns(timeseries_df: pl.DataFrame) -> Dict:
    """
    Analyze which day of the week has best buy/sell prices
    
    Returns simple recommendation: "Buy Monday, Sell Saturday" etc.
    """
    
    # Basic validation: need a dataframe with enough history (prefer 14+ points)
    if timeseries_df is None or len(timeseries_df) < 14:
        return {
            'has_data': False,
            'best_buy_day': 'Unknown',
            'best_sell_day': 'Unknown',
            'recommendation': None
        }
    
    # Add day of week (0=Monday, 6=Sunday)
    df = timeseries_df.with_columns([
        pl.col('datetime').dt.weekday().alias('day_of_week')
    ])
    
    # Calculate average buy/sell prices per day
    day_stats = df.group_by('day_of_week').agg([
        pl.col('avgLowPrice').mean().alias('avg_buy_price'),
        pl.col('avgHighPrice').mean().alias('avg_sell_price'),
        pl.col('avgLowPrice').count().alias('sample_size')
    ]).sort('day_of_week')

    # Require at least a few distinct weekday buckets to be meaningful
    if day_stats is None or len(day_stats) < 3:
        return {
            'has_data': False,
            'best_buy_day': 'Unknown',
            'best_sell_day': 'Unknown',
            'recommendation': None
        }

    # Safely extract best/worst days using head/tail patterns to avoid index errors
    try:
        # cheapest day to buy
        cheapest = day_stats.sort('avg_buy_price').head(1).to_dicts()
        if not cheapest:
            raise ValueError("no cheapest day found")
        best_buy_day_num = int(cheapest[0]['day_of_week'])
        best_buy_price = int(cheapest[0]['avg_buy_price'])

        # most expensive day to sell
        priciest = day_stats.sort('avg_sell_price', reverse=True).head(1).to_dicts()
        if not priciest:
            raise ValueError("no priciest day found")
        best_sell_day_num = int(priciest[0]['day_of_week'])
        best_sell_price = int(priciest[0]['avg_sell_price'])
    except Exception:
        # If anything goes wrong, return no-data in a safe way
        return {
            'has_data': False,
            'best_buy_day': 'Unknown',
            'best_sell_day': 'Unknown',
            'recommendation': None
        }
    
    # Convert to day names
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    best_buy_day = days[best_buy_day_num]
    best_sell_day = days[best_sell_day_num]
    
    # Calculate potential profit from timing
    worst_buy_price = int(day_stats['avg_buy_price'].max())
    worst_sell_price = int(day_stats['avg_sell_price'].min())
    
    timing_advantage = best_sell_price - best_buy_price
    bad_timing_loss = worst_buy_price - worst_sell_price
    timing_advantage_pct = ((timing_advantage / best_buy_price) * 100) if best_buy_price > 0 else 0
    
    # Detect weekend patterns
    weekend_group = day_stats.filter(pl.col('day_of_week').is_in([5, 6]))
    weekday_group = day_stats.filter(~pl.col('day_of_week').is_in([5, 6]))
    weekend_avg = weekend_group['avg_sell_price'].mean() if len(weekend_group) > 0 else 0
    weekday_avg = weekday_group['avg_sell_price'].mean() if len(weekday_group) > 0 else 0
    
    weekend_premium = ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0
    
    pattern = None
    if weekend_premium > 5:
        pattern = "🎮 Weekend spike - more players = higher demand"
    elif weekend_premium < -5:
        pattern = "💼 Weekday spike - supply drops midweek"
    
    return {
        'has_data': True,
        'best_buy_day': best_buy_day,
        'best_sell_day': best_sell_day,
        'best_buy_price': best_buy_price,
        'best_sell_price': best_sell_price,
        'timing_advantage_gp': timing_advantage,
        'timing_advantage_pct': timing_advantage_pct,
        'pattern': pattern,
        'weekend_premium_pct': weekend_premium,
    }


def get_current_day_advice(day_of_week_analysis: Dict) -> str:
    """
    Simple advice for today based on day of week
    """
    
    if not day_of_week_analysis.get('has_data', False):
        return ""
    
    today = datetime.now().strftime('%A')
    best_buy = day_of_week_analysis['best_buy_day']
    best_sell = day_of_week_analysis['best_sell_day']
    
    if today == best_buy and today == best_sell:
        return f"📅 **Today ({today})**: Normal day for this item"
    elif today == best_buy:
        return f"📅 **Today ({today})**: ✅ GREAT DAY TO BUY! Historically cheapest day."
    elif today == best_sell:
        return f"📅 **Today ({today})**: 💰 GREAT DAY TO SELL! Historically highest prices."
    else:
        if best_buy == 'Saturday' or best_buy == 'Sunday':
            buy_tip = f"Wait for {best_buy} to buy cheaper"
        else:
            buy_tip = f"Buy on {best_buy} for best price"
        
        if best_sell == 'Saturday' or best_sell == 'Sunday':
            sell_tip = f"Sell on {best_sell} for more GP"
        else:
            sell_tip = f"Sell on {best_sell} when prices peak"
        
        return f"📅 **Today ({today})**: 💡 {buy_tip}, {sell_tip}"


def detect_item_category_pattern(item_name: str) -> str:
    """
    Predict day pattern based on item type
    Combat supplies = weekend spike, skilling supplies = weekday spike
    """
    
    item_lower = item_name.lower()
    
    # Combat supplies (PvP/PvM) - spike on weekends
    combat_keywords = ['potion', 'food', 'shark', 'brew', 'restore', 'prayer', 
                      'super combat', 'ranging', 'dragon', 'barrows', 'armadyl',
                      'bandos', 'bolt', 'arrow', 'rune', 'blood', 'death', 'chaos']
    
    if any(keyword in item_lower for keyword in combat_keywords):
        return "🎮 **Pattern:** Combat item - usually spikes on weekends (more PvP/bossing)"
    
    # Skilling supplies - spike midweek
    skilling_keywords = ['log', 'ore', 'bar', 'herb', 'seed', 'bone', 'hide',
                        'leather', 'raw', 'coal', 'essence']
    
    if any(keyword in item_lower for keyword in skilling_keywords):
        return "📚 **Pattern:** Skilling item - often cheaper on weekends (fewer skillers)"
    
    return None
