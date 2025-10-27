"""
AI-Powered Trading Advisor for OSRS
Uses GPT-4 to analyze market opportunities and provide intelligent recommendations
"""

import os
import polars as pl
from typing import Dict, List, Optional
from openai import OpenAI

class TradingAdvisor:
    """AI-powered trading analysis and recommendations"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI advisor with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
    def is_available(self) -> bool:
        """Check if AI advisor is configured and available"""
        return self.client is not None
    
    def analyze_top_opportunities(self, df: pl.DataFrame, strategy: str, top_n: int = 5) -> str:
        """
        Analyze top trading opportunities and provide AI insights
        
        Args:
            df: Trading dataframe with top opportunities
            strategy: Trading strategy ("Fast Flips" or "Overnight Holds")
            top_n: Number of top items to analyze
            
        Returns:
            AI-generated analysis and recommendations
        """
        if not self.is_available():
            return "⚠️ AI Advisor not configured. Set OPENAI_API_KEY environment variable to enable."
        
        # Get top N opportunities
        top_items = df.head(top_n)
        
        # Prepare market data summary
        items_summary = []
        for row in top_items.iter_rows(named=True):
            items_summary.append({
                "name": row["name"],
                "buy_price": int(row["avgLowPrice"]),
                "sell_price": int(row["avgHighPrice"]),
                "profit_per_item": int(row["net_edge"]),
                "roi_percent": round(row["edge_pct"], 2),
                "hourly_volume": int(row["hourly_volume"]),
                "buy_volume": int(row["highPriceVolume"]),
                "sell_volume": int(row["lowPriceVolume"]),
                "confidence_score": int(row.get("confidence_score", 0)),
                "estimated_gp_per_hour": int(row.get("estimated_gp_per_hour", 0)),
            })
        
        # Create analysis prompt
        prompt = f"""You are an expert OSRS (Old School RuneScape) Grand Exchange trading advisor. Analyze these {strategy} opportunities and provide actionable insights.

STRATEGY: {strategy}
{"Focus on high-volume items for rapid turnover" if "Fast" in strategy else "Focus on high-margin items for patient trading"}

TOP {top_n} OPPORTUNITIES:
{self._format_items(items_summary)}

Provide a concise analysis covering:
1. **Best Pick**: Which item is the #1 opportunity and why?
2. **Risk Assessment**: What are the main risks (liquidity, volatility, volume imbalance)?
3. **Execution Tips**: How should the trader approach these opportunities?
4. **Market Insight**: Any patterns or trends you notice?

Keep it practical and actionable. Max 200 words."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional OSRS trading analyst focused on maximizing profits through data-driven insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"⚠️ AI Analysis unavailable: {str(e)}"
    
    def analyze_single_item(self, item_data: Dict, price_history: Optional[pl.DataFrame] = None) -> str:
        """
        Deep dive analysis on a single item
        
        Args:
            item_data: Dictionary with item details
            price_history: Optional price history dataframe
            
        Returns:
            Detailed AI analysis of the item
        """
        if not self.is_available():
            return "AI Advisor not configured"
        
        prompt = f"""Analyze this OSRS trading opportunity:

ITEM: {item_data['name']}
Buy Price: {item_data['buy_price']:,} GP
Sell Price: {item_data['sell_price']:,} GP
Profit/Item: {item_data['profit']:,} GP ({item_data['roi']:.1f}% ROI)
Hourly Volume: {item_data['volume']:,} items
Buy/Sell Volume: {item_data['buy_vol']:,} / {item_data['sell_vol']:,}
Confidence: {item_data['confidence']}/100

Should I trade this? Give me:
1. GO/NO-GO decision
2. Biggest risk
3. One-line strategy

Be direct. Max 80 words."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a decisive OSRS trading expert. Be direct and actionable."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.6,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def compare_strategies(self, fast_flip_stats: Dict, overnight_stats: Dict) -> str:
        """
        Compare two trading strategies and recommend which to use
        
        Args:
            fast_flip_stats: Stats for fast flip strategy
            overnight_stats: Stats for overnight holds strategy
            
        Returns:
            AI recommendation on which strategy to pursue
        """
        if not self.is_available():
            return "AI Advisor not configured"
        
        prompt = f"""Compare these two OSRS trading strategies for today:

FAST FLIPS (High Volume):
- Top profit/hour: {fast_flip_stats.get('top_gp_hour', 0):,} GP/hr
- Avg ROI: {fast_flip_stats.get('avg_roi', 0):.1f}%
- Opportunities: {fast_flip_stats.get('num_items', 0)} items
- Avg confidence: {fast_flip_stats.get('avg_confidence', 0)}/100

OVERNIGHT HOLDS (High Margin):
- Top profit/item: {overnight_stats.get('top_profit', 0):,} GP
- Avg ROI: {overnight_stats.get('avg_roi', 0):.1f}%
- Opportunities: {overnight_stats.get('num_items', 0)} items
- Avg confidence: {overnight_stats.get('avg_confidence', 0)}/100

Which strategy should I focus on TODAY and why? Consider current market conditions. Max 100 words."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an OSRS trading strategist. Recommend the best approach for maximum profit."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Comparison error: {str(e)}"
    
    def chat_with_advisor(self, user_question: str, market_context: Optional[Dict] = None) -> str:
        """
        Interactive chat with AI trading advisor
        
        Args:
            user_question: User's question about trading
            market_context: Optional context about current market state
            
        Returns:
            AI's response to the question
        """
        if not self.is_available():
            return "⚠️ AI Advisor not configured. Set OPENAI_API_KEY environment variable to enable."
        
        # Build context
        context_str = ""
        if market_context:
            context_str = f"\n\nCURRENT MARKET CONTEXT:\n"
            if 'top_items' in market_context:
                context_str += "Top Opportunities:\n"
                for item in market_context['top_items'][:3]:
                    context_str += f"- {item.get('name', 'Unknown')}: {item.get('profit_per_item', 0):,} GP profit, {item.get('roi_percent', 0):.1f}% ROI\n"
            if 'total_capital' in market_context:
                context_str += f"Your Capital: {market_context['total_capital']:,} GP\n"
            if 'market_temp' in market_context:
                context_str += f"Market Temperature: {market_context['market_temp']}\n"
        
        prompt = f"""You are an expert OSRS Grand Exchange trading advisor. Answer the user's question with practical, actionable advice.
{context_str}

USER QUESTION: {user_question}

Provide a clear, concise answer focused on maximizing profit and managing risk. Be specific and actionable."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional OSRS trading expert. Give practical, profit-focused advice."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"⚠️ Chat unavailable: {str(e)}"
    
    def _format_items(self, items: List[Dict]) -> str:
        """Format items list for prompt"""
        formatted = []
        for i, item in enumerate(items, 1):
            formatted.append(
                f"{i}. {item['name']}: "
                f"Buy {item['buy_price']:,} → Sell {item['sell_price']:,} GP | "
                f"Profit: {item['profit_per_item']:,} GP ({item['roi_percent']}% ROI) | "
                f"Vol: {item['hourly_volume']:,}/hr | "
                f"Confidence: {item['confidence_score']}/100"
            )
        return "\n".join(formatted)
