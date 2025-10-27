"""
TRADE TIMING INTELLIGENCE
Tells you WHEN to buy and WHEN to sell for maximum profit
"""

from typing import Dict
import polars as pl

class TradeTimingAdvisor:
    """
    SMART TIMING RECOMMENDATIONS
    
    Analyzes market conditions to tell you:
    - Best time to place buy orders
    - How long to wait before selling
    - When to hold vs instant flip
    """
    
    def get_hold_recommendation(self, item: Dict) -> Dict:
        """
        Determines optimal hold strategy for an item
        
        Returns:
        - hold_strategy: INSTANT_FLIP, SHORT_HOLD, OVERNIGHT
        - estimated_time: minutes to hold
        - reasoning: why this strategy
        """
        edge_pct = item.get('edge_pct', 0)
        volume = item.get('hourly_volume', 0)
        risk_score = item.get('risk_score', 50)
        spread_pct = item.get('spread_pct', 0)
        
        # INSTANT FLIP: High volume + moderate edge
        if volume > 100 and edge_pct >= 2 and edge_pct < 6:
            return {
                'hold_strategy': 'INSTANT FLIP ⚡',
                'estimated_time': 30,
                'buy_timing': 'Place buy offer NOW',
                'sell_timing': 'List for sale IMMEDIATELY after buy completes',
                'reasoning': 'High volume means fast trades. Flip quickly before prices change.',
                'confidence': 0.9
            }
        
        # SHORT HOLD: Lower volume or higher edge
        elif edge_pct >= 6 or (volume < 100 and edge_pct >= 3):
            return {
                'hold_strategy': 'SHORT HOLD 🕐',
                'estimated_time': 120,
                'buy_timing': 'Place buy offer and wait patiently',
                'sell_timing': 'Wait 1-2 hours after buy completes, then sell',
                'reasoning': 'Higher edge justifies waiting. Let market absorb your buy before selling.',
                'confidence': 0.8
            }
        
        # OVERNIGHT: Very high edge or low volume
        elif edge_pct >= 10 or volume < 20:
            return {
                'hold_strategy': 'OVERNIGHT 🌙',
                'estimated_time': 480,  # 8 hours
                'buy_timing': 'Place buy before logging off',
                'sell_timing': 'Check next day. Sell when buy completes.',
                'reasoning': 'Low volume or huge edge means this is a slow trade. Be patient!',
                'confidence': 0.7
            }
        
        # RISKY: High risk, be cautious
        elif risk_score > 70:
            return {
                'hold_strategy': 'RISKY - BE CAREFUL ⚠️',
                'estimated_time': 15,
                'buy_timing': 'Only buy if you monitor it closely',
                'sell_timing': 'Sell AS SOON AS buy completes',
                'reasoning': 'High risk! Price could crash. Exit fast.',
                'confidence': 0.6
            }
        
        # DEFAULT
        else:
            return {
                'hold_strategy': 'NORMAL TRADE 📊',
                'estimated_time': 60,
                'buy_timing': 'Place buy offer normally',
                'sell_timing': 'Sell within 1 hour of buy completing',
                'reasoning': 'Standard trade. Normal execution.',
                'confidence': 0.75
            }
    
    def get_market_timing_advice(self, df: pl.DataFrame) -> Dict:
        """
        Overall market timing advice
        
        Returns:
        - best_time_to_trade: NOW, WAIT_15MIN, WAIT_1HR
        - reasoning: why
        """
        if len(df) == 0:
            return {
                'best_time_to_trade': 'WAIT',
                'reasoning': 'No opportunities available'
            }
        
        # Count high quality opportunities
        high_quality = len(df.filter(pl.col('opportunity_score') >= 75)) if 'opportunity_score' in df.columns else 0
        avg_edge = df['edge_pct'].mean() if 'edge_pct' in df.columns else 0
        avg_risk = df['risk_score'].mean() if 'risk_score' in df.columns else 50
        
        # TRADE NOW: Great conditions
        if high_quality >= 10 and avg_edge > 4 and avg_risk < 40:
            return {
                'best_time_to_trade': '🔥 TRADE NOW!',
                'reasoning': f'{high_quality} elite opportunities with {avg_edge:.1f}% average edge. Market is HOT!',
                'confidence': 0.95,
                'urgency': 'HIGH'
            }
        
        # TRADE NORMALLY: Good conditions
        elif high_quality >= 5 and avg_edge > 2.5:
            return {
                'best_time_to_trade': '✅ Good Time to Trade',
                'reasoning': f'{high_quality} quality opportunities available. Normal market conditions.',
                'confidence': 0.80,
                'urgency': 'MEDIUM'
            }
        
        # WAIT: Poor conditions
        elif high_quality < 3 or avg_edge < 2 or avg_risk > 65:
            return {
                'best_time_to_trade': '⏸️ Consider Waiting 15-30 minutes',
                'reasoning': f'Only {high_quality} quality opportunities. Market conditions weak.',
                'confidence': 0.75,
                'urgency': 'LOW'
            }
        
        # DEFAULT
        else:
            return {
                'best_time_to_trade': '📊 Trade if Needed',
                'reasoning': 'Market is neutral. Trade if you find good opportunities.',
                'confidence': 0.70,
                'urgency': 'MEDIUM'
            }
