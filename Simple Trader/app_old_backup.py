"""
OSRS Trading Analytics Dashboard
Streamlit app for analyzing Grand Exchange opportunities
"""

import streamlit as st
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import time
import os

from core import (
    fetch_all_bulk_data,
    build_trading_dataframe,
    allocate_capital,
    calculate_portfolio_stats,
    ParquetCache,
)
from core.api import OSRSPricesAPI
from core.ai_advisor import TradingAdvisor
from core.config import (
    DEFAULT_CAPITAL,
    DEFAULT_MAX_PCT_PER_ITEM,
    DEFAULT_MAX_ITEMS,
    DEFAULT_MOMENTUM_WINDOW,
    DEFAULT_MIN_PROFIT_PER_FLIP,
    TABLE_COLUMNS,
)
from core.advanced_analytics import (
    get_top_opportunities_by_strategy,
    calculate_portfolio_metrics,
)
from core.trading_intelligence import TradingIntelligence
from core.quantum_intelligence import QuantumIntelligence
from core.market_dashboard import MarketDashboard
from core.hyper_dimensional import HyperDimensionalIntelligence
from core.adaptive_intelligence import AdaptiveCellularIntelligence
from core.trade_timing import TradeTimingAdvisor
from core.portfolio_tracker import PortfolioTracker


# Initialize session state for portfolio tracker
if 'portfolio_tracker' not in st.session_state:
    st.session_state.portfolio_tracker = PortfolioTracker()

# Initialize trading intelligence engines
intelligence = TradingIntelligence()
quantum = QuantumIntelligence()
market_dashboard = MarketDashboard()
hyper = HyperDimensionalIntelligence()
adaptive = AdaptiveCellularIntelligence()
timing = TradeTimingAdvisor()

# Page config
st.set_page_config(
    page_title="OSRS PRO TRADER - Ultimate Edition",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=3600)
def load_data():
    """
    Load data from API with 1-hour cache
    Returns tuple: (mapping_data, hourly_data, timestamp)
    """
    data = fetch_all_bulk_data()
    timestamp = datetime.now()
    return data["mapping"], data["1h"], timestamp


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_timeseries(item_id: int, timestep: str = "5m"):
    """
    Load timeseries data for a specific item
    Returns DataFrame with timestamp, avgHighPrice, avgLowPrice, volume
    """
    try:
        with OSRSPricesAPI() as api:
            data = api.get_timeseries(item_id, timestep)
        
        if not data or "data" not in data:
            return None
        
        # Convert to polars dataframe
        df = pl.DataFrame(data["data"])
        
        # Convert timestamp to datetime
        if "timestamp" in df.columns:
            df = df.with_columns([
                pl.from_epoch(pl.col("timestamp")).alias("datetime")
            ])
        
        return df
    except Exception as e:
        st.error(f"Failed to load timeseries: {e}")
        return None


def format_gp(value: int) -> str:
    """Format GP values with K/M/B suffixes"""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return str(value)


def show_price_history(item_id: int, item_name: str, current_buy: int, current_sell: int, timestep: str = "5m"):
    """
    Display price history chart and trend analysis for an item
    
    Args:
        item_id: Item ID
        item_name: Item name
        current_buy: Current buy price
        current_sell: Current sell price (after tax)
        timestep: Time interval ('5m', '1h', '6h')
    """
    # Load timeseries data
    df = load_timeseries(item_id, timestep)
    
    timestep_labels = {
        "5m": "5-minute",
        "1h": "1-hour", 
        "6h": "6-hour"
    }
    timestep_label = timestep_labels.get(timestep, timestep)
    
    if df is None or len(df) == 0:
        st.warning(f"No historical data available for {item_name} at {timestep_label} intervals")
        return
    
    # Filter out rows with null prices
    df = df.filter(
        (pl.col("avgHighPrice").is_not_null()) & 
        (pl.col("avgLowPrice").is_not_null())
    )
    
    if len(df) == 0:
        st.warning(f"No valid price data available for {item_name}")
        return
    
    # Calculate price trend
    if len(df) >= 2:
        # Get first and last valid prices
        first_high = df["avgHighPrice"].head(1)[0]
        last_high = df["avgHighPrice"].tail(1)[0]
        first_low = df["avgLowPrice"].head(1)[0]
        last_low = df["avgLowPrice"].tail(1)[0]
        
        # Calculate % change (with null checks)
        high_change = 0
        low_change = 0
        
        if first_high and last_high and first_high > 0:
            high_change = ((last_high - first_high) / first_high * 100)
        
        if first_low and last_low and first_low > 0:
            low_change = ((last_low - first_low) / first_low * 100)
        
        avg_change = (high_change + low_change) / 2
        
        # Determine trend
        if avg_change > 2:
            trend = "📈 Uptrend"
            trend_color = "green"
        elif avg_change < -2:
            trend = "📉 Downtrend"
            trend_color = "red"
        else:
            trend = "➡️ Stable"
            trend_color = "gray"
        
        st.markdown(f"**Trend ({timestep_label}):** :{trend_color}[{trend} ({avg_change:+.1f}%)]")
    
    # Create chart data
    chart_df = df.select([
        "datetime",
        "avgHighPrice",
        "avgLowPrice",
    ]).to_pandas()
    
    # Rename for chart
    chart_df.columns = ["Time", "High Price", "Low Price"]
    
    # Display line chart
    st.line_chart(
        chart_df,
        x="Time",
        y=["High Price", "Low Price"],
        height=250,
    )
    
    # Show current vs historical comparison
    col1, col2 = st.columns(2)
    with col1:
        buy_change = ((current_buy - first_low) / first_low * 100) if (first_low and first_low > 0) else 0
        st.metric(
            "Current Buy Price",
            f"{current_buy:,} GP",
            f"{buy_change:+.1f}% vs start"
        )
    with col2:
        sell_change = ((current_sell - first_high) / first_high * 100) if (first_high and first_high > 0) else 0
        st.metric(
            "Current Sell Price", 
            f"{current_sell:,} GP",
            f"{sell_change:+.1f}% vs start"
        )
    
    # Volume info
    if "highPriceVolume" in df.columns and "lowPriceVolume" in df.columns:
        total_volume = (df["highPriceVolume"].sum() + df["lowPriceVolume"].sum())
        avg_volume = total_volume / len(df) if len(df) > 0 else 0
        st.caption(f"📊 Avg volume per {timestep_label}: {int(avg_volume):,} items")


def main():
    # 🔥 QUANTUM-POWERED HEADER
    st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>💎 OSRS QUANTUM TRADER</h1>
        <p style='color: #f0f0f0; font-size: 1.3em; margin: 10px 0 0 0; font-weight: 500;'>The Most Intelligent Trading System in RuneScape</p>
        <p style='color: #d0d0d0; font-size: 1em; margin: 5px 0 0 0;'>🧠 Quantum Intelligence • 🔮 Price Prediction • 🧬 Genetic Optimization • 🛡️ Manipulation Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Trading Configuration")
    
    # Portfolio value input (any amount)
    total_capital = st.sidebar.number_input(
        "💰 Total Portfolio Value (GP)",
        min_value=100_000,
        max_value=100_000_000_000,  # 100B max
        value=DEFAULT_CAPITAL,
        step=1_000_000,
        format="%d",
        help="Enter your total available GP (can exceed 10M)",
    )
    
    # Minimum profit per flip
    min_profit_per_flip = st.sidebar.number_input(
        "💎 Min Profit per Item Flip (GP)",
        min_value=1_000,
        max_value=10_000_000,
        value=DEFAULT_MIN_PROFIT_PER_FLIP,
        step=5000,
        format="%d",
        help="Only show items with at least this much profit per single flip",
    )
    
    # Trading strategy selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Trading Strategy")
    
    trading_mode = st.sidebar.radio(
        "Select Mode:",
        [
            "🏆 BEST OPPORTUNITIES (All Strategies)",
            "⚡ SCALP (High-Frequency)",
            "📈 SWING (High-Margin Holds)",
            "🐋 WHALE (Big Ticket Items)",
            "🔄 ARBITRAGE (Tight Spreads)",
        ],
        help="Different strategies optimized for different trading styles",
        index=0,
    )
    
    # Parse strategy
    if "BEST" in trading_mode:
        selected_strategy = "ALL"
        strategy_icon = "🏆"
        strategy_label = "Best Opportunities"
    elif "SCALP" in trading_mode:
        selected_strategy = "⚡ SCALP"
        strategy_icon = "⚡"
        strategy_label = "Scalping"
    elif "SWING" in trading_mode:
        selected_strategy = "📈 SWING"
        strategy_icon = "📈"
        strategy_label = "Swing Trading"
    elif "WHALE" in trading_mode:
        selected_strategy = "🐋 WHALE"
        strategy_icon = "🐋"
        strategy_label = "Whale Trades"
    else:  # ARBITRAGE
        selected_strategy = "🔄 ARBITRAGE"
        strategy_icon = "🔄"
        strategy_label = "Arbitrage"
    
    # Minimum confidence score
    min_confidence = st.sidebar.slider(
        "📊 Minimum Confidence Score",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Filter by volume confidence: 🟢70+ = High, 🟡50-69 = Medium, 🔴30-49 = Low, ⚠️<30 = Very Risky",
    )
    
    st.sidebar.markdown("---")
    max_pct_per_item = st.sidebar.slider(
        "Max % per Item",
        min_value=5,
        max_value=50,
        value=DEFAULT_MAX_PCT_PER_ITEM,
        step=5,
        help="Maximum percentage of capital to allocate to a single item",
    )
    
    max_items = st.sidebar.slider(
        "Max Concurrent Items",
        min_value=1,
        max_value=25,
        value=DEFAULT_MAX_ITEMS,
        step=1,
        help="Maximum number of items to trade simultaneously",
    )
    
    # AI Advisor Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Trading Advisor")
    
    # Check for API key
    ai_key = os.getenv("OPENAI_API_KEY")
    if not ai_key:
        ai_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable AI analysis",
        )
        if ai_key:
            os.environ["OPENAI_API_KEY"] = ai_key
    
    enable_ai = st.sidebar.checkbox(
        "Enable AI Analysis",
        value=bool(ai_key),
        disabled=not bool(ai_key),
        help="Get AI-powered insights on trading opportunities" if ai_key else "Enter API key to enable",
    )
    
    if not ai_key:
        st.sidebar.caption("⚠️ Set OPENAI_API_KEY to unlock AI advisor")
    elif enable_ai:
        st.sidebar.success("🤖 AI Advisor Active")
    
    st.sidebar.markdown("---")
    momentum_window = st.sidebar.slider(
        "Momentum Window (hours)",
        min_value=1,
        max_value=24,
        value=DEFAULT_MOMENTUM_WINDOW,
        step=1,
        help="Time window for momentum calculation (not yet implemented)",
    )
    
    # 📦 GE SLOT MANAGER
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📦 Your GE Positions")
    
    tracker = st.session_state.portfolio_tracker
    available_slots = tracker.get_available_slots()
    active_positions = tracker.get_active_positions()
    
    st.sidebar.caption(f"🟢 {len(available_slots)}/8 slots available")
    
    # Show quick summary
    if active_positions:
        st.sidebar.caption(f"💼 {len(active_positions)} active trades")
    
    # Add new position
    with st.sidebar.expander("➕ Add GE Position", expanded=False):
        if available_slots:
            new_slot = st.selectbox("Slot #", available_slots, key="new_slot")
            new_item = st.text_input("Item Name", key="new_item_name")
            
            col1, col2 = st.columns(2)
            with col1:
                new_buy_price = st.number_input("Buy Price", min_value=1, value=1000, key="new_buy_price")
            with col2:
                new_quantity = st.number_input("Quantity", min_value=1, value=1, key="new_qty")
            
            if st.button("📥 Add Position", key="add_position"):
                if new_item:
                    tracker.add_position(new_slot, new_item, new_buy_price, new_quantity)
                    st.success(f"Added {new_item} to Slot #{new_slot}!")
                    st.rerun()
        else:
            st.warning("All 8 GE slots full!")
    
    # 🔔 PRICE ALERTS
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔔 Price Alerts")
    
    active_alerts = tracker.get_active_alerts()
    st.sidebar.caption(f"⏰ {len(active_alerts)} active alerts")
    
    # Add new alert
    with st.sidebar.expander("➕ Add Alert", expanded=False):
        alert_item = st.text_input("Item Name", key="alert_item")
        alert_price = st.number_input("Target Price", min_value=1, value=1000, key="alert_price")
        alert_condition = st.selectbox("Alert When", ["SELL (price goes above)", "BUY (price goes below)"], key="alert_condition")
        
        if st.button("🔔 Create Alert", key="create_alert"):
            if alert_item:
                condition = "SELL" if "SELL" in alert_condition else "BUY"
                tracker.add_alert(alert_item, alert_price, condition)
                st.success(f"Alert created for {alert_item}!")
                st.rerun()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    with st.spinner("Loading data from RuneScape Wiki API..."):
        try:
            mapping_data, hourly_data, data_timestamp = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.info("Check your internet connection and API User-Agent in core/config.py")
            return
    
    # Display last update time
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 Last updated: {data_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption(f"⏱️ Cache expires in: {60 - data_timestamp.minute % 60} min")
    
    # Build trading dataframe
    with st.spinner("Computing trading features..."):
        df = build_trading_dataframe(mapping_data, hourly_data, min_profit_per_flip)
    
    # 🎯🎯🎯 YOUR ACTIVE GE POSITIONS & ALERTS 🎯🎯🎯
    tracker = st.session_state.portfolio_tracker
    active_positions = tracker.get_active_positions()
    
    # Check for triggered alerts
    if len(df) > 0:
        current_prices = {
            row['name']: int(row['avgHighPrice'])
            for row in df.to_dicts()
        }
        triggered_alerts = tracker.check_alerts(current_prices)
        
        # Show triggered alerts at top
        if triggered_alerts:
            st.markdown("---")
            st.markdown("## 🔔 PRICE ALERTS TRIGGERED!")
            for alert in triggered_alerts:
                current_price = current_prices.get(alert.item_name, 0)
                if alert.condition == "SELL":
                    st.success(f"🎉 **{alert.item_name}** hit SELL target! Current: {current_price:,} GP (Target: {alert.target_price:,} GP)")
                else:
                    st.info(f"📉 **{alert.item_name}** hit BUY target! Current: {current_price:,} GP (Target: {alert.target_price:,} GP)")
    
    # Show your active positions
    if active_positions:
        st.markdown("---")
        st.markdown("## 💼 YOUR ACTIVE GE POSITIONS")
        
        if len(df) > 0:
            position_details = tracker.get_position_details(current_prices)
            portfolio_value = tracker.calculate_portfolio_value(current_prices)
            
            # Portfolio summary
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                st.metric("Total Invested", f"{portfolio_value['total_invested']:,} GP")
            with pcol2:
                st.metric("Current Value", f"{portfolio_value['current_value']:,} GP")
            with pcol3:
                profit = portfolio_value['unrealized_profit']
                st.metric("Unrealized Profit", f"{profit:,} GP", 
                         delta=f"{portfolio_value['unrealized_roi']:.1f}% ROI")
            with pcol4:
                st.metric("Slots Used", f"{portfolio_value['slots_used']}/8")
            
            # Position details table
            if position_details:
                st.markdown("### 📊 Position Details")
                
                for pos in position_details:
                    with st.expander(f"Slot #{pos['slot']}: {pos['item']} - {pos['roi']:.1f}% ROI", 
                                   expanded=pos['roi'] >= 2.0):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            **📥 BOUGHT:**
                            - Price: {pos['buy_price']:,} GP each
                            - Quantity: {pos['quantity']:,}
                            - Invested: {pos['invested']:,} GP
                            - Held: {pos['held_minutes']} min
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **📈 CURRENT:**
                            - Price: {pos['current_price']:,} GP each
                            - Value: {pos['current_value']:,} GP
                            - Profit: {pos['profit']:,} GP
                            - ROI: {pos['roi']:.1f}%
                            """)
                        
                        with col3:
                            if pos['roi'] >= 2.0:
                                st.success(f"✅ **SELL NOW!**  \nProfit target hit!")
                                if st.button(f"Mark as Sold", key=f"sell_{pos['slot']}"):
                                    tracker.remove_position(pos['slot'])
                                    st.rerun()
                            else:
                                target_roi = 2.0
                                needed_price = int(pos['buy_price'] * (1 + target_roi/100))
                                st.info(f"⏳ **HOLD**  \nTarget: {needed_price:,} GP  \n({target_roi}% ROI)")
                
                # Quick sell recommendations
                sell_recs = tracker.get_sell_recommendations(current_prices, min_profit_pct=2.0)
                if sell_recs:
                    st.markdown("---")
                    st.success(f"💰 **{len(sell_recs)} position(s) ready to sell!**")
        else:
            st.caption("Loading current prices...")
    
    if len(df) == 0:
        st.warning(f"No items found with minimum {format_gp(min_profit_per_flip)} profit per flip. Try lowering the threshold.")
        return
    
    # Add estimated GP/hour metric for sorting
    df = df.with_columns([
        (pl.col("net_edge") * pl.col("hourly_volume")).alias("estimated_gp_per_hour")
    ])
    
    # Filter by confidence
    df = df.filter(pl.col("confidence_score") >= min_confidence)
    
    if len(df) == 0:
        st.warning(f"""
        No items found meeting your filters.
        
        **Current Filters:**
        - Min profit/flip: **{format_gp(min_profit_per_flip)}**
        - Min confidence: **{min_confidence}/100**
        
        **Try adjusting:**
        - Lower "Min Profit per Item Flip"
        - Lower "Minimum Confidence Score"
        """)
        return
    
    # Filter by selected strategy
    if selected_strategy != "ALL":
        df_strategy = get_top_opportunities_by_strategy(df, selected_strategy, n=50)
    else:
        df_strategy = df.head(50)  # Get top 50 by opportunity score
    
    if len(df_strategy) == 0:
        st.warning(f"""
        No items found for {strategy_label} strategy.
        Try selecting a different strategy or lowering filters.
        """)
        return
    
    df = df_strategy
    strategy_desc = f"Showing top opportunities for {strategy_label} strategy"
    
    # Remove any duplicate items (keep highest ranked)
    df = df.unique(subset=["item_id"], keep="first")
    
    # Allocate capital FIRST for portfolio stats
    with st.spinner("Allocating capital..."):
        allocated_df = allocate_capital(
            df,
            total_capital_gp=total_capital,
            max_pct_per_item=max_pct_per_item,
            max_items=max_items,
        )
        
        portfolio_stats = calculate_portfolio_stats(allocated_df, total_capital)
    
    # �🔥🔥 THE ULTIMATE PICK - BEST TRADE RIGHT NOW 🔥🔥🔥
    st.markdown("---")
    
    # Find THE best trade across ALL strategies
    best_overall = df.sort(['opportunity_score', 'edge_pct'], descending=[True, True]).head(1)
    
    if len(best_overall) > 0:
        best = best_overall[0]
        
        # Create dramatic header
        st.markdown("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; margin-bottom: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.4);'>
            <h1 style='color: white; font-size: 4em; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);'>🏆 THE ULTIMATE PICK 🏆</h1>
            <p style='color: #fff; font-size: 1.5em; margin: 10px 0 0 0; font-weight: 600;'>AI's #1 Recommendation Right Now</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Convert to dict for easier access
        best_row = best.to_dicts()[0]
        best_buy = int(best_row['avgLowPrice'])
        best_sell = int(best_row['avgHighPrice'])
        best_profit_per = int(best_row['net_edge'])
        best_volume = int(best_row['hourly_volume'])
        
        # Calculate best quantity (use GE limit or reasonable amount)
        best_ge_limit = int(best_row['limit']) if best_row['limit'] > 0 else 10000
        best_qty = min(best_ge_limit, max(1, int(total_capital * 0.3 / best_buy)))  # Use 30% of capital or GE limit
        
        best_total_profit = best_profit_per * best_qty
        best_total_cost = best_buy * best_qty
        
        # Get timing recommendation
        best_timing = timing.get_hold_recommendation(best_row)
        
        # Create 3-column layout
        bcol1, bcol2, bcol3 = st.columns([1, 1, 1])
        
        with bcol1:
            st.markdown(f"### 📦 {best_row['name']}")
            st.metric("Opportunity Score", f"{int(best_row.get('opportunity_score', 0))}/100", "🔥 HIGHEST")
            st.metric("Risk Score", f"{int(best_row.get('risk_score', 50))}/100", 
                     "🟢 LOW" if best_row.get('risk_score', 50) < 30 else "🟡 MED" if best_row.get('risk_score', 50) < 60 else "🔴 HIGH")
            st.metric("Strategy Type", best_row.get('strategy_type', 'UNKNOWN'))
        
        with bcol2:
            st.markdown("### 💰 THE TRADE")
            st.markdown(f"""
            **📥 BUY:**
            - Price: **{best_buy:,} GP** each
            - Quantity: **{best_qty:,}** items
            - Total Cost: **{format_gp(best_total_cost)}**
            
            **📤 SELL:**
            - Price: **{best_sell:,} GP** each
            - Total Profit: **{format_gp(best_total_profit)}** 💎
            - ROI: **{best_row['edge_pct']:.1f}%**
            """)
        
        with bcol3:
            st.markdown("### ⏰ TIMING")
            st.info(f"**{best_timing['hold_strategy']}**")
            st.markdown(f"""
            **Buy When:** {best_timing['buy_timing']}
            
            **Sell When:** {best_timing['sell_timing']}
            
            **Hold Time:** ~{best_timing['estimated_time']} minutes
            
            **Volume:** {best_volume:,}/hour
            """)
        
        # Big execution button
        st.markdown("---")
        if st.button("🚀 EXECUTE THE ULTIMATE PICK NOW! 🚀", key="ultimate_pick_execute", width="stretch"):
            st.success(f"""
            # ✅ EXECUTING THE ULTIMATE PICK!
            
            ## Step-by-Step Instructions:
            
            ### 1️⃣ GO TO GRAND EXCHANGE
            
            ### 2️⃣ PLACE BUY ORDER
            - Search for: **{best_row['name']}**
            - Set price: **{best_buy:,} GP** per item
            - Set quantity: **{best_qty:,}** items
            - Total investment: **{format_gp(best_total_cost)}**
            - Click CONFIRM
            
            ### 3️⃣ WAIT FOR BUY TO COMPLETE
            - Estimated time: ~{max(10, int(30 * (best_qty / best_volume))) if best_volume > 0 else 60} minutes
            - **{best_timing['buy_timing']}**
            
            ### 4️⃣ LIST FOR SALE
            - **{best_timing['sell_timing']}**
            - Set price: **{best_sell:,} GP** per item
            - Set quantity: **{best_qty:,}** items
            - Click CONFIRM
            
            ### 5️⃣ COLLECT YOUR PROFIT
            - Expected profit: **{format_gp(best_total_profit)}** 💰
            - ROI: **{best_row['edge_pct']:.1f}%**
            
            ---
            
            **💡 WHY THIS TRADE:**
            - ✅ Highest opportunity score: {int(best_row.get('opportunity_score', 0))}/100
            - ✅ {best_timing['reasoning']}
            - ✅ Risk level: {int(best_row.get('risk_score', 50))}/100
            - ✅ Confidence: {best_timing['confidence']*100:.0f}%
            """)
        
        # Interactive Profit Calculator
        st.markdown("---")
        st.markdown("### 🧮 CUSTOM PROFIT CALCULATOR")
        st.caption("Adjust quantity to see YOUR exact profit based on YOUR capital")
        
        calc_col1, calc_col2 = st.columns([2, 1])
        
        with calc_col1:
            custom_qty = st.slider(
                "How many items will you buy?",
                min_value=1,
                max_value=min(best_ge_limit, 10000),
                value=best_qty,
                step=1 if best_buy > 100000 else 10 if best_buy > 10000 else 100,
                key="ultimate_calc_qty"
            )
            
            custom_cost = custom_qty * best_buy
            custom_profit = custom_qty * best_profit_per
            custom_roi = (custom_profit / custom_cost * 100) if custom_cost > 0 else 0
            
        with calc_col2:
            st.metric("Your Investment", format_gp(custom_cost))
            st.metric("Your Profit", format_gp(custom_profit), f"+{custom_roi:.1f}%")
            
            # Show if it exceeds capital
            if custom_cost > total_capital:
                st.error(f"⚠️ This exceeds your {format_gp(total_capital)} capital!")
            elif custom_cost > total_capital * 0.5:
                st.warning(f"⚠️ This uses {custom_cost/total_capital*100:.0f}% of your capital")
            else:
                st.success(f"✅ Uses {custom_cost/total_capital*100:.0f}% of capital")
        
        # Comparison with other top picks
        if len(df) > 1:
            st.markdown("---")
            st.markdown("### 📊 WHY THIS BEATS OTHER OPTIONS")
            
            # Get top 5 for comparison
            top_5 = df.head(5)
            
            comparison_data = []
            for idx, row in enumerate(top_5.iter_rows(named=True), 1):
                comparison_data.append({
                    "Rank": f"#{idx}" + (" 🏆" if idx == 1 else ""),
                    "Item": row['name'][:30],  # Truncate long names
                    "Opp": f"{int(row.get('opportunity_score', 0))}",
                    "Edge": f"{row['edge_pct']:.1f}%",
                    "Risk": f"{int(row.get('risk_score', 50))}",
                    "Volume": f"{int(row['hourly_volume']):,}",
                })
            
            st.table(comparison_data)
            st.caption("🏆 = THE ULTIMATE PICK | Opp = Opportunity Score | Edge = Profit % | Risk = Risk Score")
        
        # Real-time yield metrics
        st.markdown("---")
        st.markdown("### ⚡ LIVE YIELD METRICS")
        
        # Calculate GP/hour
        estimated_cycle_time = best_timing['estimated_time']
        cycles_per_hour = 60 / estimated_cycle_time if estimated_cycle_time > 0 else 1
        gp_per_hour = best_total_profit * cycles_per_hour
        gp_per_day = gp_per_hour * 8  # Assume 8 hours of trading
        
        yield_col1, yield_col2, yield_col3, yield_col4 = st.columns(4)
        
        with yield_col1:
            st.metric("GP/Hour", format_gp(int(gp_per_hour)), "⚡")
        
        with yield_col2:
            st.metric("GP/Day", format_gp(int(gp_per_day)), "💰")
            st.caption("(8 hours trading)")
        
        with yield_col3:
            st.metric("Cycles/Hour", f"{cycles_per_hour:.1f}", "🔄")
        
        with yield_col4:
            st.metric("Profit/Cycle", format_gp(best_total_profit), "💎")
        
        # Show what this means in real terms
        st.info(f"""
        **🎯 What this means:**
        - If you trade for 1 hour: **{format_gp(int(gp_per_hour))}** profit
        - If you trade for 4 hours: **{format_gp(int(gp_per_hour * 4))}** profit
        - If you trade 8 hours/day for a week: **{format_gp(int(gp_per_day * 7))}** profit
        
        **That's {format_gp(int(gp_per_day * 30))} per MONTH!** 🤑
        """)

        
        st.caption(f"💡 **Why this is THE pick:** Highest opportunity score ({int(best_row.get('opportunity_score', 0))}/100) combined with {best_timing['reasoning']}")
    
    # �💎💎💎 QUICK TRADE IDEAS - READY TO EXECUTE 💎💎💎
    st.markdown("---")
    st.markdown("## 💎 TOP 3 TRADE IDEAS - Copy & Execute!")
    
    # Market timing advice
    market_timing = timing.get_market_timing_advice(df)
    
    timing_colors = {
        'HIGH': 'success',
        'MEDIUM': 'info',
        'LOW': 'warning'
    }
    timing_color = timing_colors.get(market_timing.get('urgency', 'MEDIUM'), 'info')
    
    getattr(st, timing_color)(f"""
    ⏰ **MARKET TIMING**: {market_timing['best_time_to_trade']}
    
    {market_timing['reasoning']} (Confidence: {market_timing.get('confidence', 0)*100:.0f}%)
    """)
    
    st.caption("⚡ **Top opportunities ready to trade immediately - BUY prices, SELL prices, quantities, and hold times!**")
    
    # Get top 3 allocated trades
    top_trades = allocated_df.head(3)
    
    trade_cols = st.columns(3)
    
    for idx, (col, row) in enumerate(zip(trade_cols, top_trades.iter_rows(named=True)), 1):
        with col:
            buy_price = int(row['avgLowPrice'])
            sell_price = int(row['avgHighPrice'])
            qty = int(row['allocation_qty'])
            profit_per = int(row['net_edge'])
            total_profit = profit_per * qty
            total_cost = buy_price * qty
            volume = int(row['hourly_volume'])
            
            # Get timing recommendation
            timing_advice = timing.get_hold_recommendation(row)
            
            # Calculate estimated times
            buy_fill_time = max(10, int(30 * (qty / volume))) if volume > 0 else 60
            sell_fill_time = max(10, int(30 * (qty / volume))) if volume > 0 else 60
            total_time = buy_fill_time + sell_fill_time
            
            # Color code by rank
            rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            
            st.markdown(f"### {rank_emoji} #{idx}: {row['name']}")
            
            # Timing strategy badge
            st.info(f"**Strategy**: {timing_advice['hold_strategy']} | Confidence: {timing_advice['confidence']*100:.0f}%")
            
            # Buy section
            st.markdown("**📥 BUY ORDER:**")
            st.code(f"""
Price:    {buy_price:,} GP each
Quantity: {qty:,} items
Total:    {format_gp(total_cost)}
Est Time: ~{buy_fill_time} min
            """)
            st.caption(f"⏰ {timing_advice['buy_timing']}")
            
            # Sell section
            st.markdown("**📤 SELL ORDER:**")
            st.code(f"""
Price:    {sell_price:,} GP each
Quantity: {qty:,} items  
Est Time: ~{sell_fill_time} min
            """)
            st.caption(f"⏰ {timing_advice['sell_timing']}")
            
            # Profit summary
            st.success(f"**💰 PROFIT: {format_gp(total_profit)}**")
            st.caption(f"⏱️ Total Time: ~{total_time} min | Hold: {timing_advice['estimated_time']} min")
            st.caption(f"📊 ROI: {row['edge_pct']:.1f}% | 🎯 Opp: {int(row.get('opportunity_score', 0))}/100 | ⚠️ Risk: {int(row.get('risk_score', 50))}/100")
            st.caption(f"💡 **Why**: {timing_advice['reasoning']}")
            
            # Action button
            if st.button(f"✅ Copy Trade #{idx}", key=f"copy_trade_{idx}"):
                st.info(f"""
**EXECUTE THIS TRADE ({timing_advice['hold_strategy']}):**

1. Go to GE
2. Buy {qty:,}x {row['name']} at {buy_price:,} GP
3. Wait ~{buy_fill_time} min for buy to complete
4. {timing_advice['sell_timing']}
5. List for sale at {sell_price:,} GP
6. Wait ~{sell_fill_time} min for sell to complete
6. Profit: {format_gp(total_profit)} 🎉
                """)
    
    # 🧬🧬🧬 ADAPTIVE CELLULAR INTELLIGENCE - LEARNS & EVOLVES 🧬🧬🧬
    st.markdown("---")
    st.markdown("## 🧬 ADAPTIVE INTELLIGENCE - Self-Learning System")
    
    with st.expander("ℹ️ WHAT IS ADAPTIVE INTELLIGENCE?", expanded=False):
        st.markdown("""
        **This system is like a LIVING BRAIN that learns from the market:**
        
        - 🧬 **Cellular Learning**: Adapts to market conditions in real-time
        - 🎯 **Context-Aware**: Gives you specific advice for YOUR exact situation
        - ⚠️ **Risk Alerts**: Warns you if your portfolio is too risky or inefficient
        - 🔄 **Strategy Optimization**: Tells you if you should switch strategies
        - 🔬 **Anomaly Detection**: Finds unusual patterns that could be manipulation or errors
        
        **Think of it as your personal trading coach that never sleeps!**
        """)
    
    st.caption("This system LEARNS from market patterns and generates contextual insights specific to YOUR situation")
    
    # Generate adaptive insights
    adaptive_insights = adaptive.generate_adaptive_insights(df, portfolio_stats)
    
    if adaptive_insights:
        # Show critical insights first
        critical_insights = [i for i in adaptive_insights if i.priority == 1]
        important_insights = [i for i in adaptive_insights if i.priority == 2]
        info_insights = [i for i in adaptive_insights if i.priority == 3]
        
        if critical_insights:
            st.markdown("### 🚨 CRITICAL INSIGHTS")
            for insight in critical_insights:
                with st.expander(f"{insight.message} (Confidence: {insight.confidence*100:.0f}%)", expanded=True):
                    st.markdown("**What You Should Do:**")
                    for action in insight.action_items:
                        st.markdown(f"• {action}")
        
        if important_insights:
            st.markdown("### ⚠️ IMPORTANT OBSERVATIONS")
            for insight in important_insights:
                with st.expander(f"{insight.message} (Confidence: {insight.confidence*100:.0f}%)"):
                    st.markdown("**Recommendations:**")
                    for action in insight.action_items:
                        st.markdown(f"• {action}")
        
        if info_insights:
            with st.expander(f"ℹ️ Additional Insights ({len(info_insights)})"):
                for insight in info_insights:
                    st.markdown(f"**{insight.message}** (Confidence: {insight.confidence*100:.0f}%)")
                    for action in insight.action_items:
                        st.caption(f"• {action}")
    
    # Meta-Strategy Advisor
    st.markdown("---")
    st.markdown("### 🎯 Strategy Optimization")
    
    meta_strategy = adaptive.generate_meta_strategy(df, total_capital, selected_strategy)
    
    ms_col1, ms_col2 = st.columns([1, 2])
    
    with ms_col1:
        if meta_strategy['recommendation'] != selected_strategy:
            st.warning(f"**Consider Switching Strategy:**  \n{meta_strategy['recommendation']}")
            st.metric("Potential Gain", meta_strategy.get('opportunity_increase', 'N/A'))
        else:
            st.success(f"**Optimal Strategy:**  \n{meta_strategy['recommendation']}")
        
        st.metric("Confidence", f"{meta_strategy.get('confidence', 0)*100:.0f}%")
    
    with ms_col2:
        st.markdown(f"**Analysis:**")
        st.info(meta_strategy.get('reason', 'Analyzing...'))
        if 'action' in meta_strategy:
            st.markdown(f"**Action:** {meta_strategy['action']}")
    
    # Anomaly Detection
    anomalies = adaptive.detect_market_anomalies(df)
    
    if anomalies:
        st.markdown("---")
        st.markdown("### 🔬 Market Anomaly Detection")
        st.warning(f"⚠️ {len(anomalies)} unusual patterns detected!")
        
        for anomaly in anomalies:
            severity_color = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(anomaly['severity'], '⚪')
            
            with st.expander(f"{severity_color} {anomaly['type']}: {anomaly['item']}", expanded=anomaly['severity'] == 'CRITICAL'):
                st.markdown(f"**Value:** {anomaly['value']}")
                st.markdown(f"**Explanation:** {anomaly['explanation']}")
                st.info(f"**Action:** {anomaly['action']}")
    
    # 🔥🔥🔥 COMPREHENSIVE MARKET INTELLIGENCE DASHBOARD 🔥🔥🔥
    st.markdown("---")
    st.markdown("## 📊 REAL-TIME MARKET INTELLIGENCE")
    
    with st.expander("ℹ️ UNDERSTANDING MARKET TEMPERATURE", expanded=False):
        st.markdown("""
        **Market Temperature tells you HOW GOOD the current opportunities are:**
        
        - 🔥 **HOT** = 10+ elite trades (score 75+) → **GO ALL IN! Deploy maximum capital!**
        - 🌡️ **WARM** = 5-9 elite trades → **Good conditions, trade normally**
        - ❄️ **COOL** = <5 elite trades → **Be selective or wait for better opportunities**
        
        **Think of it like:**
        - HOT = Black Friday sales everywhere!
        - WARM = Regular shopping day with some deals
        - COOL = Meh, maybe come back later
        
        **This is THE MOST IMPORTANT metric - it tells you if NOW is a good time to trade!**
        """)
    
    # Executive Summary - THE MOST IMPORTANT INFO UP TOP
    exec_summary = market_dashboard.generate_executive_summary(df, total_capital)
    st.info(exec_summary)
    
    # Market Metrics Grid
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    
    snapshot = market_dashboard.get_comprehensive_market_state(df)
    
    with mcol1:
        temp_color = "🔥" if "HOT" in snapshot.temperature else "🌡️" if "WARM" in snapshot.temperature else "❄️"
        st.metric("Market Temperature", snapshot.temperature)
        st.caption("**WHAT THIS MEANS:**  \n🔥 HOT = 10+ elite trades (GO ALL IN!)  \n🌡️ WARM = 5-9 elite trades (Trade normally)  \n❄️ COOL = <5 elite trades (Be picky!)")
    
    with mcol2:
        st.metric("Elite Opportunities", f"{snapshot.high_quality_count}", f"of {snapshot.total_opportunities} total")
        st.caption("Score 75+ = Elite quality")
    
    with mcol3:
        sentiment_emoji = "📈" if "BULLISH" in snapshot.market_sentiment else "📉" if "BEARISH" in snapshot.market_sentiment else "➡️"
        st.metric("Sentiment", f"{sentiment_emoji} {snapshot.market_sentiment.split()[0]}")
        st.caption(f"Avg Profit: {snapshot.average_profit_pct:.1f}%  \nAvg Risk: {snapshot.average_risk:.0f}/100")
    
    with mcol4:
        st.metric("Analysis Confidence", f"{snapshot.confidence*100:.0f}%")
        st.caption(f"Volume: {snapshot.total_volume:,}/hr")
    
    # Opportunity & Risk Breakdown
    st.markdown("---")
    breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
    
    with breakdown_col1:
        st.markdown("### 🎯 Opportunity Distribution")
        opp_heatmap = market_dashboard.calculate_opportunity_heatmap(df)
        for category, count in opp_heatmap.items():
            emoji = "🔥" if "ELITE" in category else "⭐" if "EXCELLENT" in category else "✅" if "GOOD" in category else "⚠️" if "MEDIOCRE" in category else "❌"
            
            # Make each category clickable
            if count > 0:
                with st.expander(f"{emoji} **{category}**: {count} items - CLICK FOR TRADES"):
                    # Filter items in this score range
                    if "ELITE" in category:
                        category_items = df.filter(pl.col('opportunity_score') >= 90)
                    elif "EXCELLENT" in category:
                        category_items = df.filter((pl.col('opportunity_score') >= 75) & (pl.col('opportunity_score') < 90))
                    elif "GOOD" in category:
                        category_items = df.filter((pl.col('opportunity_score') >= 60) & (pl.col('opportunity_score') < 75))
                    elif "MEDIOCRE" in category:
                        category_items = df.filter((pl.col('opportunity_score') >= 40) & (pl.col('opportunity_score') < 60))
                    else:  # POOR
                        category_items = df.filter(pl.col('opportunity_score') < 40)
                    
                    # Show top 5 items in this category with trade instructions
                    for idx, item in enumerate(category_items.head(5).iter_rows(named=True), 1):
                        buy_price = int(item['avgLowPrice'])
                        sell_price = int(item['avgHighPrice'])
                        profit_per = int(item['net_edge'])
                        volume = int(item['hourly_volume'])
                        
                        # Calculate hold time based on volume
                        estimated_fill_time = max(15, int(60 * (100 / volume))) if volume > 0 else 120
                        
                        st.markdown(f"**{idx}. {item['name']}**")
                        st.markdown(f"""
                        - 💰 **BUY AT**: {buy_price:,} GP each
                        - 💵 **SELL AT**: {sell_price:,} GP each  
                        - 📊 **PROFIT**: {profit_per:,} GP per item ({item['edge_pct']:.1f}% ROI)
                        - 📦 **QUANTITY**: Buy as many as you can (GE Limit: {int(item['limit']) if item['limit'] > 0 else 'Unlimited'})
                        - ⏱️ **HOLD TIME**: ~{estimated_fill_time} minutes (based on {volume:,}/hr volume)
                        - 🎯 **WHEN TO SELL**: Immediately after buy completes - list at {sell_price:,} GP
                        - ⚠️ **RISK**: {int(item.get('risk_score', 50))}/100 {'🟢' if item.get('risk_score', 50) < 30 else '🟡' if item.get('risk_score', 50) < 60 else '🔴'}
                        """)
                        st.markdown("---")
            else:
                st.markdown(f"{emoji} **{category}**: {count} items")
    
    with breakdown_col2:
        st.markdown("### 🛡️ Risk Distribution")
        risk_dist = market_dashboard.calculate_risk_distribution(df)
        for category, count in risk_dist.items():
            # Make each risk level clickable
            if count > 0:
                with st.expander(f"**{category}**: {count} items - CLICK FOR TRADES"):
                    # Filter items in this risk range
                    if "LOW" in category:
                        risk_items = df.filter(pl.col('risk_score') <= 30)
                    elif "MEDIUM" in category:
                        risk_items = df.filter((pl.col('risk_score') > 30) & (pl.col('risk_score') <= 60))
                    elif "HIGH" in category:
                        risk_items = df.filter((pl.col('risk_score') > 60) & (pl.col('risk_score') <= 80))
                    else:  # EXTREME
                        risk_items = df.filter(pl.col('risk_score') > 80)
                    
                    # Show top 5 items sorted by opportunity score
                    sorted_risk_items = risk_items.sort('opportunity_score', descending=True)
                    
                    for idx, item in enumerate(sorted_risk_items.head(5).iter_rows(named=True), 1):
                        buy_price = int(item['avgLowPrice'])
                        sell_price = int(item['avgHighPrice'])
                        profit_per = int(item['net_edge'])
                        volume = int(item['hourly_volume'])
                        
                        # Calculate hold time
                        estimated_fill_time = max(15, int(60 * (100 / volume))) if volume > 0 else 120
                        
                        # Risk warning
                        risk_warning = ""
                        if "EXTREME" in category:
                            risk_warning = "⚠️ **EXTREME RISK!** Trade with caution or avoid!"
                        elif "HIGH" in category:
                            risk_warning = "⚠️ **HIGH RISK** - Only trade if confident"
                        elif "MEDIUM" in category:
                            risk_warning = "⚡ Moderate risk - normal trading"
                        else:
                            risk_warning = "✅ **LOW RISK** - Safe trade!"
                        
                        st.markdown(f"**{idx}. {item['name']}** - {risk_warning}")
                        st.markdown(f"""
                        - 💰 **BUY AT**: {buy_price:,} GP each
                        - 💵 **SELL AT**: {sell_price:,} GP each
                        - 📊 **PROFIT**: {profit_per:,} GP per item ({item['edge_pct']:.1f}% ROI)
                        - 📦 **QUANTITY**: Suggest buying {min(100, int(item['limit']) if item['limit'] > 0 else 100):,} items
                        - ⏱️ **HOLD TIME**: ~{estimated_fill_time} minutes
                        - 🎯 **OPPORTUNITY**: {int(item.get('opportunity_score', 50))}/100
                        - ⚠️ **RISK**: {int(item.get('risk_score', 50))}/100
                        """)
                        st.markdown("---")
            else:
                st.markdown(f"**{category}**: {count} items")
    
    with breakdown_col3:
        st.markdown("### 💧 Liquidity Analysis")
        vol_analysis = market_dashboard.calculate_volume_analysis(df)
        st.metric("Total Volume", f"{vol_analysis['total_hourly_volume']:,}/hr")
        st.metric("Avg Volume", f"{vol_analysis['average_volume']:,}/hr")
        st.metric("Liquidity Rating", vol_analysis['liquidity_rating'])
        st.caption(f"High-volume items (200+/hr): {vol_analysis['high_volume_items']}")
    
    # Top Movers
    st.markdown("---")
    st.markdown("### 🚀 Top Market Movers")
    
    movers = market_dashboard.get_top_movers(df)
    mover_cols = st.columns(3)
    
    for idx, mover in enumerate(movers[:6]):  # Show top 6 in 3 columns
        with mover_cols[idx % 3]:
            st.markdown(f"**{mover['category']}**")
            st.markdown(f"📦 {mover['item']}")
            st.metric("Value", mover['value'], label_visibility="collapsed")
            st.caption(mover['detail'])
    
    # Display metrics (capital already allocated above)
    st.markdown("---")
    st.header(f"{strategy_icon} {strategy_label}: Portfolio Overview")
    st.caption(f"{strategy_desc} | Showing top {len(allocated_df)} items")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Capital",
            format_gp(total_capital),
        )
    
    with col2:
        st.metric(
            "Allocated",
            format_gp(portfolio_stats["total_allocated"]),
            f"{portfolio_stats['allocation_rate']:.1f}%",
        )
    
    with col3:
        st.metric(
            "Positions",
            portfolio_stats["num_positions"],
            f"/{max_items}",
        )
    
    with col4:
        st.metric(
            "Avg Edge",
            f"{portfolio_stats['avg_edge_pct']:.2f}%",
        )
    
    with col5:
        st.metric(
            "Potential Profit",
            format_gp(portfolio_stats["total_potential_profit"]),
        )
    
    # 🔥 GOD-TIER FEATURES: Profit Forecasting & Market Intelligence
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 💰 Profit Forecast")
        st.caption("⚡ **WHAT THIS IS:** Runs 1,000 simulations to predict your profits with confidence intervals (like a weather forecast but for GP!)")
        with st.spinner("Running Monte Carlo simulation..."):
            forecast = intelligence.forecast_daily_profit(allocated_df, total_capital)
        
        # Create forecast metrics
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.metric(
                "Expected Daily",
                format_gp(forecast["expected_daily_profit"]),
                f"{forecast['probability_of_profit']:.0%} chance"
            )
        with fcol2:
            st.metric(
                "Best Case",
                format_gp(forecast["confidence_interval_high"]),
                "95th %ile"
            )
        with fcol3:
            st.metric(
                "Worst Case",
                format_gp(forecast["confidence_interval_low"]),
                "5th %ile"
            )
        
        # Extended forecasts
        st.caption(f"📅 **Weekly**: ~{format_gp(forecast['expected_weekly_profit'])} | **Monthly**: ~{format_gp(forecast['expected_monthly_profit'])}")
        st.caption(f"⚠️ **Downside Risk**: {format_gp(forecast['downside_risk'])} | **Upside Potential**: {format_gp(forecast['upside_potential'])}")
        
        # Risk-reward bar
        rr_ratio = forecast['risk_reward_ratio']
        if rr_ratio >= 3:
            st.success(f"🔥 Excellent Risk/Reward: {rr_ratio:.1f}:1")
        elif rr_ratio >= 2:
            st.info(f"✅ Good Risk/Reward: {rr_ratio:.1f}:1")
        else:
            st.warning(f"⚠️ Moderate Risk/Reward: {rr_ratio:.1f}:1")
    
    with col_right:
        st.markdown("### 🎯 Market Intelligence")
        st.caption("⚡ **WHAT THIS IS:** Real-time analysis of overall market health and opportunity quality")
        market_intel = intelligence.generate_market_insights(df)
        
        icol1, icol2, icol3 = st.columns(3)
        with icol1:
            temp_emoji = "🔥" if market_intel.get('market_temperature') == "HOT" else "🌤️" if market_intel.get('market_temperature') == "WARM" else "❄️"
            st.metric(
                "Market Temp",
                f"{temp_emoji} {market_intel.get('market_temperature', 'N/A')}",
                f"{market_intel.get('high_quality_opportunities', 0)} premium"
            )
        with icol2:
            st.metric(
                "Avg Opportunity",
                f"{market_intel.get('average_opportunity_score', 0):.0f}/100",
                f"{market_intel.get('opportunity_density', 0):.1f}% density"
            )
        with icol3:
            st.metric(
                "Avg Risk",
                f"{market_intel.get('average_risk_score', 0):.0f}/100",
                "Lower is better"
            )
        
        st.caption(f"📊 **Top Strategy**: {market_intel.get('top_strategy', 'N/A')} | **Avg Volume**: {market_intel.get('average_volume', 0):,}/hr")
        
        # Opportunity alerts
        alerts = intelligence.scan_for_opportunities(df)
        if alerts:
            st.success(f"🚨 {len(alerts)} high-priority alerts detected!")
            for alert in alerts[:3]:  # Show top 3
                st.caption(f"• {alert.item_name}: {alert.message}")
        else:
            st.info("👀 No critical alerts. Market is stable.")
    
    # 🧠 QUANTUM INTELLIGENCE SECTION
    st.markdown("---")
    st.markdown("## 🧠 QUANTUM INTELLIGENCE - Advanced Market Analysis")
    
    with st.expander("ℹ️ WHAT IS QUANTUM INTELLIGENCE?", expanded=False):
        st.markdown("""
        **Advanced pattern recognition and prediction system:**
        
        - 🌐 **Market Regime**: Is the market BULL (aggressive), BEAR (defensive), SIDEWAYS, or VOLATILE?
        - 🔍 **Pattern Recognition**: Detects PUMPS (prices rising), DUMPS (prices crashing), BREAKOUTS, etc.
        - 🎯 **Trade Signals**: STRONG BUY, BUY, HOLD, or AVOID recommendations for each item
        - 🛡️ **Manipulation Detection**: Finds suspicious activity like pump & dumps
        - 🧬 **Genetic Optimization**: Evolves the PERFECT portfolio using AI
        - 💱 **Arbitrage Detection**: Finds correlated items with price divergences
        
        **This is like having a Wall Street quant analyst working for you 24/7!**
        """)
    
    # Row 1: Market Regime + Price Patterns
    qcol1, qcol2 = st.columns([1, 1])
    
    with qcol1:
        st.markdown("### 🌐 Market Regime Detection")
        st.caption("⚡ **WHAT THIS MEANS:** Tells you if the market favors aggressive trading (BULL) or defensive trading (BEAR)")
        regime = quantum.predict_market_regime(df)
        
        # Color-code regime
        regime_colors = {
            "BULL": "🟢",
            "BEAR": "🔴",
            "SIDEWAYS": "🟡",
            "VOLATILE": "🟠"
        }
        regime_emoji = regime_colors.get(regime.regime_type, "⚪")
        
        st.markdown(f"## {regime_emoji} {regime.regime_type} MARKET")
        st.metric("Confidence", f"{regime.confidence*100:.0f}%")
        st.metric("Duration Estimate", f"{regime.duration_estimate} min")
        st.info(f"**Strategy**: {regime.recommendation}")
    
    with qcol2:
        st.markdown("### 🔍 Price Pattern Recognition")
        patterns = quantum.detect_price_patterns(df)
        
        if patterns:
            st.success(f"Detected {len(patterns)} patterns")
            
            # Show top 3 patterns
            for pattern in patterns[:3]:
                pattern_emoji = {
                    "PUMP": "🚀",
                    "DUMP": "📉",
                    "BREAKOUT": "💥",
                    "CONSOLIDATION": "➡️",
                    "REVERSAL": "🔄"
                }.get(pattern.pattern_type, "📊")
                
                st.markdown(f"**{pattern_emoji} {pattern.pattern_type}**: {pattern.item_name}")
                st.caption(f"Strength: {pattern.strength:.0f}/100 | Direction: {pattern.predicted_direction} | Confidence: {pattern.confidence*100:.0f}%")
        else:
            st.info("No significant patterns detected")
    
    # Row 2: Trade Signals + Manipulation Detection
    st.markdown("---")
    scol1, scol2 = st.columns([1, 1])
    
    with scol1:
        st.markdown("### 🎯 Master Trade Signals")
        signals = quantum.generate_trade_signals(df)
        
        if signals:
            # Create signal summary
            strong_buys = [s for s in signals if s['signal'] == 'STRONG_BUY']
            buys = [s for s in signals if s['signal'] == 'BUY']
            
            sscol1, sscol2, sscol3 = st.columns(3)
            with sscol1:
                st.metric("STRONG BUY", len(strong_buys), "🔥")
            with sscol2:
                st.metric("BUY", len(buys), "✅")
            with sscol3:
                st.metric("Total Analyzed", len(signals))
            
            # Show top signals
            st.markdown("**Top Signals:**")
            for signal in signals[:5]:
                signal_emoji = {
                    "STRONG_BUY": "🔥",
                    "BUY": "✅",
                    "HOLD": "⏸️",
                    "AVOID": "⛔"
                }.get(signal['signal'], "📊")
                
                st.markdown(f"{signal_emoji} **{signal['item']}**: {signal['signal']} (Score: {signal['score']:.0f}, Edge: {signal['edge']:.1f}%, Risk: {signal['risk']:.0f})")
                if signal.get('patterns'):
                    st.caption(f"Patterns: {', '.join(signal['patterns'])}")
    
    with scol2:
        st.markdown("### 🛡️ Manipulation Detection")
        suspicious = quantum.detect_manipulation(df)
        
        if suspicious:
            st.warning(f"⚠️ {len(suspicious)} suspicious items detected")
            
            for item in suspicious[:5]:
                risk_color = "🔴" if item['risk_level'] == "HIGH" else "🟡"
                st.markdown(f"{risk_color} **{item['item']}** - {item['recommendation']}")
                st.caption(f"Red flags: {', '.join(item['red_flags'])}")
        else:
            st.success("✅ No manipulation detected. Market appears clean.")
    
    # Row 3: Genetic Portfolio Optimization
    st.markdown("---")
    st.markdown("### 🧬 Genetic Algorithm Portfolio Optimization")
    
    if st.button("🚀 Run Genetic Optimizer (50 generations)", help="Uses evolutionary algorithms to find the mathematically optimal portfolio"):
        with st.spinner("Evolving optimal portfolio... 🧬"):
            genetic_result = quantum.optimize_portfolio_genetic(
                df,
                total_capital,
                generations=50,
                population_size=20
            )
        
        gcol1, gcol2 = st.columns([1, 2])
        
        with gcol1:
            st.metric("Fitness Score", f"{genetic_result['fitness_score']:,.0f}")
            st.metric("Optimal Items", genetic_result['num_items'])
            st.caption(f"Algorithm: {genetic_result['algorithm']}")
        
        with gcol2:
            st.markdown("**Genetically Optimized Portfolio:**")
            for item in genetic_result['optimal_items']:
                st.markdown(f"• {item}")
            
            st.info("💡 This portfolio was evolved through 50 generations of genetic selection to maximize profit/risk ratio")
    
    # Row 4: Arbitrage Opportunities
    st.markdown("---")
    st.markdown("### 💱 Statistical Arbitrage Detector")
    
    arbitrage_opps = quantum.calculate_item_correlation(df)
    
    if arbitrage_opps:
        st.success(f"Found {len(arbitrage_opps)} arbitrage opportunities")
        
        for arb in arbitrage_opps:
            st.markdown(f"**{arb.item_a}** ↔️ **{arb.item_b}**")
            st.caption(f"Spread Z-Score: {arb.spread_zscore:.2f} | Expected: {format_gp(arb.expected_profit)} | Risk: {arb.risk_level}")
    else:
        st.info("No arbitrage opportunities detected")
    
    # 🌌🌌🌌 HYPER-DIMENSIONAL ANALYSIS 🌌🌌🌌
    st.markdown("---")
    st.markdown("## 🌌 HYPER-DIMENSIONAL INTELLIGENCE")
    
    with st.expander("ℹ️ WHAT IS HYPER-DIMENSIONAL INTELLIGENCE?", expanded=False):
        st.markdown("""
        **BEYOND normal trading analysis - uses concepts from physics and biology:**
        
        - ⚛️ **Quantum Analysis**: Calculates theoretical MAXIMUM profit if you could trade everything at once
        - 🌀 **Fractal Patterns**: Finds repeating patterns across different timeframes
        - 🐜 **Swarm Intelligence**: 100 virtual ants find the absolute BEST trade sequence
        - 🧬 **Portfolio Synergy**: Measures how well your trades complement each other
        
        **This is CUTTING-EDGE stuff that doesn't exist in ANY other trading tool!**
        
        **Quantum Efficiency** = How close you are to the theoretical maximum profit
        **Portfolio Synergy** = How well your items work together (diversification + correlation)
        """)
    
    st.caption("*Features beyond what Wall Street uses - Quantum Theory, Swarm Intelligence, Fractal Analysis*")
    
    # Row 1: Quantum Superposition & Opportunity Clusters
    hcol1, hcol2 = st.columns([1, 1])
    
    with hcol1:
        st.markdown("### ⚛️ Quantum Portfolio Analysis")
        st.caption("⚡ **WHAT THIS IS:** Theoretical maximum profit if you could break the laws of physics and trade everything simultaneously")
        quantum_analysis = hyper.quantum_superposition_analysis(allocated_df, total_capital)
        
        qm1, qm2 = st.columns(2)
        with qm1:
            st.metric(
                "Quantum Maximum", 
                format_gp(quantum_analysis.get('quantum_maximum_profit', 0)),
                help="Theoretical max if you could trade everything simultaneously"
            )
        with qm2:
            st.metric(
                "Quantum Efficiency",
                f"{quantum_analysis.get('quantum_efficiency', 0):.1f}%",
                help="How close you are to the theoretical maximum"
            )
        
        # Clamp progress to 0-1 range
        efficiency_progress = min(1.0, max(0.0, quantum_analysis.get('quantum_efficiency', 0) / 100))
        st.progress(efficiency_progress)
        
        if quantum_analysis.get('quantum_efficiency', 0) > 80:
            st.success("🔥 EXCELLENT! You're extracting near-maximum value from the market!")
        elif quantum_analysis.get('quantum_efficiency', 0) > 60:
            st.info("✅ Good efficiency. Room for minor optimization.")
        else:
            st.warning("⚠️ Low efficiency. Market has more potential than you're capturing.")
        
        st.caption(f"**Theoretical Ceiling**: {format_gp(quantum_analysis.get('theoretical_ceiling', 0))}")
        
        with st.expander("🔬 Quantum States Breakdown"):
            for state in quantum_analysis.get('top_quantum_states', [])[:3]:
                st.markdown(f"**{state['item']}**")
                st.caption(f"Max Profit: {format_gp(state['max_theoretical_profit'])} | Probability: {state['probability']*100:.0f}%")
    
    with hcol2:
        st.markdown("### 🎯 Opportunity Clusters")
        clusters = hyper.detect_opportunity_clusters(allocated_df)
        
        if clusters:
            st.success(f"Found {len(clusters)} synergistic clusters")
            
            for cluster in clusters:
                with st.expander(f"{cluster.cluster_name} - {len(cluster.items)} items"):
                    st.metric("Avg Opportunity", f"{cluster.avg_opportunity_score:.0f}/100")
                    st.metric("Cluster Synergy", f"{cluster.cluster_synergy*100:.0f}%")
                    if cluster.total_profit_potential > 0:
                        st.metric("Total Potential", format_gp(cluster.total_profit_potential))
                    
                    st.markdown("**Execution Order:**")
                    for i, item in enumerate(cluster.execution_order[:5], 1):
                        st.caption(f"{i}. {item}")
                    
                    st.info("💡 Trade these items together for maximum GE slot efficiency!")
        else:
            st.info("No strong clusters detected. Items are independent.")
    
    # Row 2: Market Anomalies & Fractal Patterns
    st.markdown("---")
    acol1, acol2 = st.columns([1, 1])
    
    with acol1:
        st.markdown("### 🔥 Market Anomaly Detection")
        anomalies = hyper.detect_market_anomalies(df)
        
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected!")
            
            for anomaly in anomalies[:5]:
                anomaly_emoji = "🚨" if anomaly.severity > 70 else "⚠️" if anomaly.severity > 40 else "ℹ️"
                threat_color = "🟢" if anomaly.opportunity_or_threat == "OPPORTUNITY" else "🔴"
                
                st.markdown(f"{anomaly_emoji} {threat_color} **{anomaly.anomaly_type}**")
                st.caption(f"Severity: {anomaly.severity:.0f}/100 | Items: {', '.join(anomaly.affected_items[:2])}")
                st.info(f"**Action**: {anomaly.action_required}")
        else:
            st.success("✅ Market is normal. No unusual activity detected.")
    
    with acol2:
        st.markdown("### 📐 Fractal Pattern Analysis")
        fractals = hyper.fractal_pattern_detection(allocated_df)
        
        if fractals:
            st.success(f"Detected {len(fractals)} fractal patterns")
            
            for pattern in fractals[:4]:
                pattern_emoji = "🟢" if pattern['pattern_type'] == "FRACTAL_STABILITY" else "🔴"
                
                st.markdown(f"{pattern_emoji} **{pattern['item']}**")
                st.caption(f"{pattern['description']}")
                st.info(f"💡 {pattern['recommendation']}")
        else:
            st.info("No significant fractal patterns detected")
    
    # Row 3: Swarm Intelligence Optimization
    st.markdown("---")
    st.markdown("### 🐜 Swarm Intelligence - Ant Colony Optimization")
    st.caption("⚡ **WHAT THIS IS:** Uses collective ant colony behavior to find the absolute BEST trade sequence - like 100 ants exploring every path simultaneously!")
    
    # Get session hours early for swarm
    swarm_session_hours = st.slider(
        "Swarm Session Duration (hours)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.5,
        key="swarm_session_slider",
        help="How long will swarm optimize for?"
    )
    
    if st.button("🚀 Run Swarm Optimizer (100 ants, 20 generations)", help="Uses collective intelligence to find optimal trade sequence"):
        with st.spinner("Deploying swarm... 🐜🐜🐜"):
            swarm_results = hyper.swarm_optimization_recommendation(allocated_df, swarm_session_hours)
        
        if swarm_results:
            st.success(f"Swarm discovered optimal path with {len(swarm_results):,} trades!")
            
            swarm_col1, swarm_col2 = st.columns([2, 1])
            
            with swarm_col1:
                st.markdown("**Swarm-Optimized Sequence:**")
                total_swarm_profit = 0
                for rec in swarm_results:
                    pheromone_emoji = "🔥" if rec['pheromone_strength'] == "HIGH" else "✨"
                    st.markdown(f"{rec['swarm_rank']}. {pheromone_emoji} **{rec['item']}** - {format_gp(rec['profit'])}")
                    total_swarm_profit += rec['profit']
            
            with swarm_col2:
                st.metric("Total Swarm Profit", format_gp(total_swarm_profit))
                st.caption("This sequence was discovered by 100 virtual ants exploring 20 generations of paths!")
                
                st.info("💡 The swarm found this path has the strongest 'pheromone trail' = highest collective confidence")
    
    # Portfolio Synergy Score
    st.markdown("---")
    st.markdown("### ⚖️ Portfolio Synergy Analysis")
    
    portfolio_items = allocated_df.to_dicts()
    synergy_score = hyper.calculate_portfolio_synergy(portfolio_items[:10])
    
    synergy_col1, synergy_col2 = st.columns([1, 2])
    
    with synergy_col1:
        st.metric("Portfolio Synergy", f"{synergy_score:.0f}/100")
        # Clamp progress to 0-1 range
        synergy_progress = min(1.0, max(0.0, synergy_score / 100))
        st.progress(synergy_progress)
        
        if synergy_score > 75:
            st.success("🔥 EXCELLENT synergy! Items complement each other well")
        elif synergy_score > 50:
            st.info("✅ Good synergy. Reasonably balanced portfolio")
        else:
            st.warning("⚠️ Low synergy. Consider more diversification")
    
    with synergy_col2:
        st.markdown("**Synergy Components:**")
        st.caption("• **Price Diversity**: Different price ranges reduce concentration risk")
        st.caption("• **Volume Balance**: Mix of fast/slow movers optimizes time usage")
        st.caption("• **Risk Distribution**: Spread across risk levels = stable returns")
        st.caption("• **Strategy Diversity**: Multiple strategies = adaptive to market changes")
        
        st.info("💡 High synergy means your portfolio is OPTIMIZED for all market conditions!")
    
    # 🔥 SMART EXECUTION PLANNER
    st.markdown("---")
    st.markdown("### ⚡ Smart Execution Plan")
    
    session_hours = st.slider(
        "Trading session duration (hours)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.5,
        help="How long will you actively trade?"
    )
    
    execution_plan = intelligence.generate_execution_plan(
        allocated_df,
        total_capital,
        session_duration_hours=session_hours
    )
    
    if execution_plan:
        st.success(f"✅ Optimized sequence for {len(execution_plan)} trades in {session_hours}h session")
        
        # Summary metrics
        total_planned_profit = sum(t['expected_profit'] for t in execution_plan)
        total_planned_time = sum(t['expected_time_minutes'] for t in execution_plan)
        
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("Total Expected Profit", format_gp(total_planned_profit))
        with pcol2:
            st.metric("Total Time", f"{int(total_planned_time)} min")
        with pcol3:
            st.metric("GP/Hour Rate", format_gp(int(total_planned_profit / (total_planned_time/60))))
        with pcol4:
            st.metric("Trades Planned", len(execution_plan))
        
        # Execution table
        with st.expander("📋 View Detailed Execution Sequence", expanded=False):
            import pandas as pd
            plan_df = pd.DataFrame(execution_plan)
            plan_df['buy_price'] = plan_df['buy_price'].apply(lambda x: f"{x:,} GP")
            plan_df['sell_price'] = plan_df['sell_price'].apply(lambda x: f"{x:,} GP")
            plan_df['capital_required'] = plan_df['capital_required'].apply(lambda x: format_gp(x))
            plan_df['expected_profit'] = plan_df['expected_profit'].apply(lambda x: format_gp(x))
            plan_df['expected_time_minutes'] = plan_df['expected_time_minutes'].apply(lambda x: f"{x} min")
            
            st.dataframe(
                plan_df[[
                    'sequence', 'item_name', 'priority', 'quantity',
                    'buy_price', 'sell_price', 'expected_profit',
                    'expected_time_minutes', 'opportunity_score', 'risk_score'
                ]],
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("No execution plan could be generated. Try increasing capital or session duration.")
    
    # AI Analysis Section
    if enable_ai and len(allocated_df) > 0:
        st.markdown("---")
        with st.expander("🤖 AI Market Analysis", expanded=False):
            with st.spinner("AI analyzing opportunities..."):
                advisor = TradingAdvisor()
                if advisor.is_available():
                    analysis = advisor.analyze_top_opportunities(
                        allocated_df,
                        strategy_label,
                        top_n=min(5, len(allocated_df))
                    )
                    st.markdown(analysis)
                else:
                    st.warning("AI Advisor not available. Check API key configuration.")
    
    # AI Chat Interface
    if enable_ai:
        st.markdown("---")
        st.markdown("### 💬 Ask the AI Trading Advisor")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Create market context for AI
        market_context = {
            'total_capital': total_capital,
        }
        if len(allocated_df) > 0:
            top_items = allocated_df.head(3).to_dicts()
            market_context['top_items'] = [
                {
                    'name': item.get('name', 'Unknown'),
                    'profit_per_item': int(item.get('net_edge', 0)),
                    'roi_percent': float(item.get('edge_pct', 0))
                }
                for item in top_items
            ]
        if 'market_dashboard' in locals():
            market_context['market_temp'] = market_dashboard.get_comprehensive_market_state(allocated_df)['temperature']
        
        # Chat input
        user_question = st.text_input(
            "Ask anything about trading strategy, items, risks, or market conditions:",
            placeholder="e.g., Should I buy Dragon platelegs now? What's the safest trade? How to use 5M capital?",
            key="ai_chat_input"
        )
        
        if st.button("🤖 Get AI Advice", key="chat_submit"):
            if user_question:
                with st.spinner("AI is thinking..."):
                    advisor = TradingAdvisor()
                    if advisor.is_available():
                        response = advisor.chat_with_advisor(user_question, market_context)
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': response
                        })
        
        # Display chat history (most recent first)
        if st.session_state.chat_history:
            st.markdown("#### 📜 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:80]}...", expanded=(i==0)):
                    st.markdown(f"**You asked:** {chat['question']}")
                    st.markdown(f"**AI Advisor:** {chat['answer']}")
            
            if st.button("🗑️ Clear Chat History", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Allocated items table
    st.header(f"🎯 Action Plan: What to Buy & Sell ({strategy_label})")
    
    if len(allocated_df) > 0:
        # Strategy-specific guidance based on selected_strategy
        strategy_tips = {
            "ALL": "**Best Opportunities:** Top-ranked items across all strategies based on opportunity score.",
            "⚡ SCALP": "**Scalping Strategy:** Ultra-high volume, tight spreads. Aim for quick turnarounds (minutes to hours).",
            "📈 SWING": "**Swing Trading:** High margin items. Hold overnight or longer for maximum profit %.",
            "🐋 WHALE": "**Whale Trades:** Big ticket items with massive profit per flip. Requires large capital.",
            "🔄 ARBITRAGE": "**Arbitrage:** Guaranteed spreads with high confidence. Low risk, steady profits.",
            "📊 STANDARD": "**Standard Trading:** Balanced approach with good risk/reward ratios."
        }
        strategy_tip = strategy_tips.get(selected_strategy, strategy_tips["ALL"])
        
        st.info(f"""
        💡 **How to read this:** Buy at the 'Buy Price', sell at 'GE Sell Price' (GE automatically deducts 2% tax). Your actual profit is shown in 'Profit/Item'.
        
        {strategy_tip}
        
        📊 **Confidence Score:** Measures volume quality and balance:
        - 🟢 **70-100**: High confidence - balanced buy/sell volume, good liquidity
        - 🟡 **50-69**: Medium confidence - decent volume but may be one-sided
        - 🔴 **30-49**: Low confidence - low volume or unbalanced market (risky!)
        """)
        
        # Prepare display dataframe with actionable columns + advanced metrics
        display_df = allocated_df.select([
            "name",
            "strategy_type",
            "opportunity_score",
            "risk_score",
            "buy_price",
            "avgHighPrice",  # GE sell price (before tax)
            "allocation_qty",
            "net_edge",
            "edge_pct",
            "sharpe_ratio",
            "hourly_volume",
            "confidence_score",
            "expected_value_per_hour",
            "allocation_gp",
            "limit",
        ])
        
        # Convert to pandas for display with formatting
        display_pd = display_df.to_pandas()
        
        # Calculate total profit per position
        display_pd["total_profit"] = (display_pd["net_edge"] * display_pd["allocation_qty"]).astype(int)
        
        # Format columns
        display_pd["strategy_type"] = display_pd["strategy_type"]
        display_pd["opportunity_score"] = display_pd["opportunity_score"].apply(lambda x: f"{'🔥' if x >= 80 else '⭐' if x >= 60 else '✓'} {x}/100")
        display_pd["risk_score"] = display_pd["risk_score"].apply(lambda x: f"{'🟢' if x <= 30 else '🟡' if x <= 60 else '🔴'} {x}/100")
        display_pd["buy_price"] = display_pd["buy_price"].apply(lambda x: f"{int(x):,} GP")
        display_pd["avgHighPrice"] = display_pd["avgHighPrice"].apply(lambda x: f"{int(x):,} GP")
        display_pd["net_edge"] = display_pd["net_edge"].apply(lambda x: f"{int(x):,} GP")
        display_pd["expected_value_per_hour"] = display_pd["expected_value_per_hour"].apply(lambda x: f"{format_gp(int(x))}/hr")
        display_pd["total_profit"] = display_pd["total_profit"].apply(lambda x: f"💰 {x:,} GP")
        display_pd["allocation_gp"] = display_pd["allocation_gp"].apply(format_gp)
        display_pd["edge_pct"] = display_pd["edge_pct"].apply(lambda x: f"{x:.1f}%")
        display_pd["sharpe_ratio"] = display_pd["sharpe_ratio"].apply(lambda x: f"{x:.2f}")
        display_pd["allocation_qty"] = display_pd["allocation_qty"].apply(lambda x: f"{x:,}")
        display_pd["limit"] = display_pd["limit"].apply(lambda x: f"{x:,}" if x > 0 else "∞")
        display_pd["confidence_score"] = display_pd["confidence_score"].apply(lambda x: f"{'🟢' if x >= 70 else '🟡' if x >= 50 else '🔴'} {x}/100")
        
        # Rename columns for clarity
        display_pd.columns = [
            "Item Name",
            "Strategy",
            "Opportunity",
            "Risk",
            "🛒 Buy At",
            "💰 Sell At",
            "Qty",
            "Profit/Item",
            "ROI %",
            "Sharpe",
            "Volume/hr",
            "Confidence",
            "💎 EV/Hour",
            "Capital Used",
            "GE Limit",
            "Total Profit",
        ]
        
        st.dataframe(
            display_pd,
            width="stretch",
            hide_index=True,
            height=min(400, len(display_pd) * 35 + 38),
        )
        
        # Add specific trading instructions with price history
        st.markdown("### 📋 Step-by-Step Instructions (Click to Expand)")
        
        # Timeframe selector for charts
        chart_timeframe = st.selectbox(
            "📊 Chart Timeframe",
            options=["5m", "1h", "6h"],
            format_func=lambda x: {"5m": "5 Minutes", "1h": "1 Hour", "6h": "6 Hours"}[x],
            index=1,  # Default to 1h
            help="Select time interval for price history charts"
        )
        
        for idx, row in enumerate(allocated_df.head(10).iter_rows(named=True), 1):
            profit_total = int(row['net_edge'] * row['allocation_qty'])
            with st.expander(f"#{idx} - {row['name']} - 💰 {format_gp(profit_total)} profit"):
                # Show price history chart
                st.markdown("#### 📈 Price History & Trend")
                show_price_history(
                    item_id=int(row['item_id']),
                    item_name=row['name'],
                    current_buy=int(row['buy_price']),
                    current_sell=int(row['sell_price']),
                    timestep=chart_timeframe
                )
                
                st.markdown("---")
                st.markdown("#### � AI Price Prediction")
                
                # Get quantum prediction for this item
                item_data_for_pred = {
                    'highPriceVolume': int(row.get('highPriceVolume', 0)),
                    'lowPriceVolume': int(row.get('lowPriceVolume', 0)),
                    'hourly_volume': int(row['hourly_volume']),
                    'spread_pct': float(row['spread_pct']),
                    'edge_pct': float(row['edge_pct'])
                }
                
                prediction = quantum.predict_price_movement(item_data_for_pred)
                
                pred_emoji = {
                    "RISING": "📈",
                    "FALLING": "📉",
                    "STABLE": "➡️",
                    "UNKNOWN": "❓"
                }.get(prediction['prediction'], "📊")
                
                pcol1, pcol2, pcol3 = st.columns(3)
                with pcol1:
                    st.metric("Prediction", f"{pred_emoji} {prediction['prediction']}")
                with pcol2:
                    st.metric("Confidence", f"{prediction['confidence']*100:.0f}%")
                with pcol3:
                    st.metric("Expected Change", f"{prediction.get('expected_change_pct', 0):.2f}%")
                
                rec_color = {
                    "BUY": "success",
                    "SELL": "error",
                    "HOLD": "info"
                }.get(prediction['recommendation'], "info")
                
                getattr(st, rec_color)(f"🎯 **Recommendation**: {prediction['recommendation']} (next {prediction['time_horizon_minutes']} min)")
                
                st.markdown("---")
                st.markdown("#### �💼 Trading Instructions")
                
                # Calculate actual GE prices
                buy_price = int(row['buy_price'])
                sell_price_before_tax = int(row['avgHighPrice'])  # What you list in GE
                sell_price_after_tax = int(row['sell_price'])  # What you actually receive
                ge_tax = sell_price_before_tax - sell_price_after_tax
                profit_per_item = int(row['net_edge'])
                total_cost = int(row['allocation_gp'])
                qty = int(row['allocation_qty'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **📥 BUY OFFER:**
                    - Place offer at: **{buy_price:,} GP** each
                    - Quantity: **{qty:,}** items
                    - Total cost: **{format_gp(total_cost)}**
                    """)
                with col2:
                    st.markdown(f"""
                    **📤 SELL OFFER (after buying):**
                    - Place offer at: **{sell_price_before_tax:,} GP** each
                    - GE takes 2% tax: **-{ge_tax:,} GP** per item
                    - You receive: **{sell_price_after_tax:,} GP** per item
                    - Total revenue: **{format_gp(sell_price_after_tax * qty)}**
                    """)
                
                # Show profit breakdown
                st.success(f"✅ Net profit: **{format_gp(profit_total)}** = {qty:,} items × {profit_per_item:,} GP profit/item ({row['edge_pct']:.1f}% ROI)")
                st.caption(f"⚠️ GE Limit: {int(row['limit']) if row['limit'] > 0 else 'None'} per 4 hours | Hourly volume: {int(row['hourly_volume'])} items")
                
                # 🔥 ONE-CLICK TRADE SETUP
                st.markdown("---")
                st.markdown("#### 🚀 Quick Actions")
                
                qcol1, qcol2, qcol3 = st.columns(3)
                
                with qcol1:
                    # Generate trade summary for clipboard
                    trade_summary = f"""TRADE: {row['name']}
BUY: {qty:,} @ {buy_price:,} GP = {format_gp(total_cost)}
SELL: {qty:,} @ {sell_price_before_tax:,} GP
PROFIT: {format_gp(profit_total)} ({row['edge_pct']:.1f}% ROI)
                    """
                    if st.button(f"📋 Copy Trade Details", key=f"copy_{idx}", use_container_width=True):
                        st.code(trade_summary, language=None)
                        st.caption("✅ Trade details displayed above. Copy manually.")
                
                with qcol2:
                    # Generate GE search command
                    ge_search = f"{row['name']}"
                    if st.button(f"🔍 GE Search", key=f"search_{idx}", use_container_width=True):
                        st.info(f"Search GE for: **{ge_search}**")
                
                with qcol3:
                    # Quick reference
                    if st.button(f"⚡ Quick Ref", key=f"ref_{idx}", use_container_width=True):
                        st.info(f"**Buy**: {buy_price:,} GP × {qty:,}  \n**Sell**: {sell_price_before_tax:,} GP × {qty:,}")
                
                # Execution Checklist
                with st.expander("✅ Execution Checklist", expanded=False):
                    st.markdown(f"""
                    **Before You Trade:**
                    - [ ] Verify you have **{format_gp(total_cost)}** available GP
                    - [ ] Check current GE prices are still near **{buy_price:,} GP** (buy) and **{sell_price_before_tax:,} GP** (sell)
                    - [ ] Confirm GE limit allows **{qty:,}** items (limit: {int(row['limit']) if row['limit'] > 0 else 'unlimited'})
                    - [ ] Set buy offer at **{buy_price:,} GP** for **{qty:,}** items
                    
                    **After Buy Completes:**
                    - [ ] Immediately list for sale at **{sell_price_before_tax:,} GP**
                    - [ ] GE will deduct **{ge_tax:,} GP** tax per item
                    - [ ] You'll receive **{sell_price_after_tax:,} GP** per item = **{format_gp(sell_price_after_tax * qty)}** total
                    - [ ] Net profit: **{format_gp(profit_total)}**
                    
                    **Risk Management:**
                    - Opportunity Score: **{row.get('opportunity_score', 0)}/100** {'🔥' if row.get('opportunity_score', 0) >= 80 else '⭐' if row.get('opportunity_score', 0) >= 60 else '✓'}
                    - Risk Score: **{row.get('risk_score', 50)}/100** {'🟢' if row.get('risk_score', 50) <= 30 else '🟡' if row.get('risk_score', 50) <= 60 else '🔴'}
                    - Confidence: **{row.get('confidence_score', 0)}/100**
                    - Expected Time: ~**{int(15 + (qty / row['hourly_volume'] * 60))} minutes**
                    """)
                
                # AI Analysis for this specific item
                if enable_ai:
                    st.markdown("---")
                    st.markdown("#### 🤖 AI Item Analysis")
                    with st.spinner("AI analyzing..."):
                        advisor = TradingAdvisor()
                        if advisor.is_available():
                            item_data = {
                                "name": row['name'],
                                "buy_price": buy_price,
                                "sell_price": sell_price_before_tax,
                                "profit": profit_per_item,
                                "roi": row['edge_pct'],
                                "volume": int(row['hourly_volume']),
                                "buy_vol": int(row.get('highPriceVolume', 0)),
                                "sell_vol": int(row.get('lowPriceVolume', 0)),
                                "confidence": int(row.get('confidence_score', 0)),
                            }
                            item_analysis = advisor.analyze_single_item(item_data)
                            st.info(item_analysis)
    else:
        st.info("No items match the current criteria. Try adjusting parameters.")
    
    # Top opportunities (all items, not just allocated)
    st.header("� All Trading Opportunities (Top 50)")
    
    st.caption(f"Showing items with minimum {format_gp(min_profit_per_flip)} profit per flip, sorted by rank score")
    
    # Show top 50 by rank score
    top_opportunities = df.head(50).select([
        "name",
        "buy_price",
        "sell_price",
        "net_edge",
        "edge_pct",
        "hourly_volume",
        "limit",
        "rank_score",
    ])
    
    top_pd = top_opportunities.to_pandas()
    top_pd["buy_price"] = top_pd["buy_price"].apply(lambda x: f"{int(x):,}")
    top_pd["sell_price"] = top_pd["sell_price"].apply(lambda x: f"{int(x):,}")
    top_pd["net_edge"] = top_pd["net_edge"].apply(lambda x: f"{int(x):,} GP")
    top_pd["edge_pct"] = top_pd["edge_pct"].apply(lambda x: f"{x:.1f}%")
    top_pd["limit"] = top_pd["limit"].apply(lambda x: f"{int(x):,}" if x > 0 else "∞")
    top_pd["rank_score"] = top_pd["rank_score"].apply(lambda x: f"{x:.1f}")
    
    top_pd.columns = [
        "Item",
        "Buy Price",
        "Sell Price",
        "Profit/Flip",
        "ROI %",
        "Hourly Vol",
        "4h Limit",
        "Score",
    ]
    
    st.dataframe(
        top_pd,
        width="stretch",
        hide_index=True,
        height=600,
    )
    
    # Footer
    st.markdown("---")
    st.caption("Data from [RuneScape Wiki Real-Time Prices API](https://prices.runescape.wiki/)")
    st.caption("⚠️ This is for educational purposes. Real trading involves risk and GE limits.")


if __name__ == "__main__":
    main()
