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
import plotly.graph_objects as go

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
    TABLE_COLUMNS,
)
from core.advanced_analytics import (
    get_top_opportunities_by_strategy,
    calculate_portfolio_metrics,
)
from core.trading_intelligence import TradingIntelligence
from core.quantum_intelligence import QuantumIntelligence
from core.market_dashboard import MarketDashboard
from core.adaptive_intelligence import AdaptiveCellularIntelligence
from core.trade_timing import TradeTimingAdvisor
from core.portfolio_tracker import PortfolioTracker
from core.falling_knife_detector import detect_falling_knife
from core.day_of_week_analyzer import analyze_day_of_week_patterns, get_current_day_advice, detect_item_category_pattern
from core.performance_tracker import PerformanceTracker


# Initialize session state for portfolio tracker
if 'portfolio_tracker' not in st.session_state:
    st.session_state.portfolio_tracker = PortfolioTracker()

# Initialize performance tracker
if 'performance_tracker' not in st.session_state:
    st.session_state.performance_tracker = PerformanceTracker()

# Initialize trading intelligence engines
intelligence = TradingIntelligence()
quantum = QuantumIntelligence()
market_dashboard = MarketDashboard()
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


def create_price_chart(df, title="Price History"):
    """
    Create a Plotly chart with y-axis on the right side for better readability
    
    Args:
        df: Polars dataframe with datetime, avgHighPrice, avgLowPrice columns
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    if df is None or len(df) == 0:
        return None
    
    # Convert to pandas for Plotly
    chart_df = df.select(["datetime", "avgHighPrice", "avgLowPrice"]).to_pandas()
    
    # Create figure
    fig = go.Figure()
    
    # Add high price line
    fig.add_trace(go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["avgHighPrice"],
        name="High Price",
        line=dict(color='#00ff00', width=2),
        mode='lines'
    ))
    
    # Add low price line
    fig.add_trace(go.Scatter(
        x=chart_df["datetime"],
        y=chart_df["avgLowPrice"],
        name="Low Price",
        line=dict(color='#00aaff', width=2),
        mode='lines'
    ))
    
    # Update layout with y-axis on the right
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (GP)",
        yaxis=dict(
            side='right',  # Move y-axis to right side
            tickformat=',d',  # Format numbers with commas
        ),
        hovermode='x unified',
        template='plotly_dark',
        height=350,
        margin=dict(l=20, r=80, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


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


def display_performance_tracker():
    """Display the Performance Tracker dashboard"""
    
    tracker = st.session_state.performance_tracker
    
    st.markdown("## 📊 Historical Performance Tracker")
    st.markdown("Track your trading performance over time and analyze what's working")
    
    # Get performance stats
    stats = tracker.get_performance_stats()
    
    # Overall performance metrics
    st.markdown("### 💰 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total P&L",
            format_gp(stats['total_pnl']),
            f"{stats['total_roi']:.1f}% ROI"
        )
    
    with col2:
        st.metric(
            "Win Rate",
            f"{stats['win_rate']*100:.1f}%",
            f"{stats['total_wins']}W - {stats['total_losses']}L"
        )
    
    with col3:
        st.metric(
            "Total Trades",
            f"{stats['total_trades']}",
            f"{stats['closed_trades']} closed"
        )
    
    with col4:
        st.metric(
            "Avg Profit/Trade",
            format_gp(stats['avg_profit_per_trade']),
            f"Best: {format_gp(stats['best_trade'])}"
        )
    
    # Strategy performance breakdown
    if stats['total_trades'] > 0:
        st.markdown("---")
        st.markdown("### 🎯 Strategy Performance")
        
        strategy_stats = tracker.get_best_strategies()
        
        if strategy_stats:
            # Create dataframe for display
            strategy_rows = []
            for strategy, data in strategy_stats.items():
                strategy_rows.append({
                    "Strategy": strategy,
                    "Trades": data['trade_count'],
                    "Total Profit": format_gp(data['total_profit']),
                    "Avg Profit": format_gp(data['avg_profit']),
                    "Win Rate": f"{data['win_rate']*100:.1f}%",
                })
            
            # Sort by total profit
            strategy_rows.sort(key=lambda x: strategy_stats[x['Strategy']]['total_profit'], reverse=True)
            
            # Display as table
            st.dataframe(
                strategy_rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No strategy data yet. Close some trades to see performance by strategy.")
    
    # Recent trades
    st.markdown("---")
    st.markdown("### 📝 Recent Trades")
    
    recent_trades = tracker.get_recent_trades(limit=10)
    
    if recent_trades:
        # Display recent trades
        for trade in recent_trades:
            status_icon = "✅" if trade.get('status') == 'CLOSED' else "⏳"
            profit = trade.get('total_profit', 0)
            profit_color = "green" if profit > 0 else "red" if profit < 0 else "gray"
            
            with st.expander(f"{status_icon} {trade.get('item_name', 'Unknown')} - {format_gp(profit)} ({trade.get('strategy', 'N/A')})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Buy Price:** {trade.get('buy_price', 0):,} GP")
                    st.write(f"**Sell Price:** {trade.get('sell_price', 0):,} GP")
                    st.write(f"**Quantity:** {trade.get('quantity', 0):,}")
                
                with col2:
                    st.write(f"**Profit/Item:** {trade.get('profit_per_item', 0):,} GP")
                    st.write(f"**Total Profit:** {profit:,} GP")
                    st.write(f"**Status:** {trade.get('status', 'UNKNOWN')}")
                
                if 'entry_time' in trade:
                    st.caption(f"Entered: {trade['entry_time']}")
                if trade.get('status') == 'CLOSED' and 'exit_time' in trade:
                    st.caption(f"Closed: {trade['exit_time']}")
    else:
        st.info("📭 No trades logged yet")
        st.markdown("""
        **How to use the Performance Tracker:**
        1. Go to the Trading Dashboard
        2. Find a trade you executed
        3. Use the "Log Trade" button to record it
        4. Come back here to see your performance stats!
        
        💡 The more trades you log, the better insights you'll get about what strategies work best for you.
        """)
    
    # Manual trade logging form
    st.markdown("---")
    st.markdown("### ➕ Log a New Trade")
    
    with st.expander("📝 Click to log a trade manually"):
        with st.form("log_trade_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                item_name = st.text_input("Item Name", placeholder="e.g., Dragon bones")
                item_id = st.number_input("Item ID", min_value=1, value=1, step=1)
                buy_price = st.number_input("Buy Price (GP)", min_value=1, value=1000, step=1)
                quantity = st.number_input("Quantity", min_value=1, value=100, step=1)
            
            with col2:
                sell_price = st.number_input("Sell Price (GP)", min_value=1, value=1100, step=1)
                strategy = st.selectbox("Strategy", ["INSTANT_FLIP", "SHORT_HOLD", "SWING", "WHALE", "STANDARD"])
                status = st.selectbox("Status", ["OPEN", "CLOSED"])
                entry_time = st.date_input("Entry Date")
            
            submit = st.form_submit_button("💾 Log Trade")
            
            if submit:
                # Calculate profit
                profit_per_item = sell_price - buy_price
                total_profit = profit_per_item * quantity
                
                # Create trade data
                trade_data = {
                    "item_name": item_name,
                    "item_id": item_id,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "quantity": quantity,
                    "profit_per_item": profit_per_item,
                    "total_profit": total_profit,
                    "strategy": strategy,
                    "status": status,
                    "entry_time": entry_time.isoformat(),
                    "exit_time": datetime.now().isoformat() if status == "CLOSED" else None,
                    "predicted_opportunity_score": 0,  # Not available for manual entries
                    "predicted_risk_score": 0,
                }
                
                # Log trade
                if tracker.log_trade(trade_data):
                    st.success(f"✅ Trade logged: {item_name} - {format_gp(total_profit)} profit")
                    st.rerun()
                else:
                    st.error("Failed to log trade")


def main():
    # 🔐 PASSWORD PROTECTION
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
            <h1 style='color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>💎 OSRS QUANTUM TRADER</h1>
            <p style='color: #f0f0f0; font-size: 1.3em; margin: 10px 0 0 0; font-weight: 500;'>Private Access Only</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔐 Enter Password to Continue")
        
        password_input = st.text_input("Password:", type="password", key="password_input")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔓 Login", use_container_width=True):
                # Password stored in .streamlit/secrets.toml (not in code!)
                correct_password = st.secrets.get("passwords", {}).get("app_password", "mv29")
                if password_input == correct_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect password. Try again.")
        
        st.info("💡 This is a private trading dashboard for authorized users only.")
        return
    
    # Initialize auto-refresh timer
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Auto-refresh every 5 minutes
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh >= 300:  # 300 seconds = 5 minutes
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()
    
    # 🔥 MV29 HEADER
    st.markdown("""
    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='color: white; font-size: 3.5em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>💎 MV29 CAPITAL</h1>
        <p style='color: #f0f0f0; font-size: 1.3em; margin: 10px 0 0 0; font-weight: 500;'>Grand Exchange Algorithmic Trading Desk</p>
        <p style='color: #d0d0d0; font-size: 1em; margin: 5px 0 0 0;'>📊 Quantitative Analysis • 🎯 Alpha Generation • ⚡ High-Frequency Execution • 🛡️ Risk Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Trading Configuration")
    
    # Logout button at top of sidebar
    if st.sidebar.button("🔓 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Investible GP slider (500K to 10B in 5M increments)
    total_capital = st.sidebar.slider(
        "💰 Investible GP",
        min_value=500_000,      # 500K minimum
        max_value=10_000_000_000,  # 10B maximum
        value=10_000_000_000,  # Default to MAX (10B)
        step=5_000_000,         # 5M increments
        format="%d",
        help="How much GP you want to invest. Slide to quickly test different amounts (500K to 10B in 5M steps)",
    )
    
    # Show formatted value
    st.sidebar.caption(f"Selected: **{format_gp(total_capital)}**")
    
    # Minimum ROI percentage (better than fixed GP for mass quantity trades)
    min_roi_pct = st.sidebar.slider(
        "📊 Min ROI % per Flip",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Minimum return on investment %. 1% = great for mass quantity, 5%+ = high margin trades",
    )
    
    # Minimum total profit filter
    min_total_profit = st.sidebar.slider(
        "💰 Min Total Profit per Trade",
        min_value=0,
        max_value=5_000_000,  # 5M max
        value=0,  # Default: no filter
        step=50_000,  # 50K increments
        format="%d",
        help="Only show trades where total profit (profit per item × quantity) is at least this amount",
    )
    
    # Show formatted value
    if min_total_profit > 0:
        st.sidebar.caption(f"Min profit: **{format_gp(min_total_profit)}**")
    
    # Trading strategy selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Trading Strategy")
    
    trading_mode = st.sidebar.radio(
        "Select Mode:",
        [
            "🏆 BEST OPPORTUNITIES (All Strategies)",
            "⚡ INSTANT FLIP (Same-Day Flips)",
            "� SHORT HOLD (1-2 Days)",
            "� SWING (3-7 Days)",
            "� WHALE (Big Ticket Items)",
        ],
        help="Different strategies optimized for different trading styles and timeframes",
        index=0,
    )
    
    # Parse strategy
    if "BEST" in trading_mode:
        selected_strategy = "ALL"
        strategy_icon = "🏆"
        strategy_label = "Best Opportunities"
    elif "INSTANT" in trading_mode:
        selected_strategy = "INSTANT_FLIP"
        strategy_icon = "⚡"
        strategy_label = "Instant Flip"
    elif "SHORT" in trading_mode:
        selected_strategy = "SHORT_HOLD"
        strategy_icon = "�"
        strategy_label = "Short Hold (1-2 days)"
    elif "SWING" in trading_mode:
        selected_strategy = "SWING"
        strategy_icon = "�"
        strategy_label = "Swing Trading (3-7 days)"
    else:  # WHALE
        selected_strategy = "WHALE"
        strategy_icon = "�"
        strategy_label = "Whale Trades"
    
    # Minimum confidence score
    
    st.sidebar.markdown("---")
    # Set to show ALL items - no filtering
    max_pct_per_item = 50  # Max allocation per item
    max_items = 25  # Show up to 25 items
    min_confidence = 0  # Show all confidence levels
    
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
        value=2,  # Default to 2 hours
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
    
    # Item Price History Lookup
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Item Price Lookup")
    
    with st.sidebar.expander("📊 Search Any Item", expanded=False):
        search_item = st.text_input("Item Name", placeholder="e.g., Dragon bones", key="item_search")
        
        if search_item and st.button("🔎 Search Price History", key="search_btn"):
            st.session_state.searched_item = search_item
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()
        st.rerun()
    
    # Section selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dashboard Section")
    view_mode = st.sidebar.radio(
        "Select View:",
        ["📈 Trading Dashboard", "📊 Performance Tracker"],
        help="Switch between live trading opportunities and historical performance",
        index=0,
    )
    
    # Load data
    with st.spinner("Loading data from RuneScape Wiki API..."):
        try:
            mapping_data, hourly_data, data_timestamp = load_data()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.info("Check your internet connection and API User-Agent in core/config.py")
            return
    
    # Display last update time and auto-refresh info
    st.sidebar.markdown("---")
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    minutes_until_refresh = max(0, int((300 - time_since_refresh) / 60))
    seconds_until_refresh = max(0, int((300 - time_since_refresh) % 60))
    
    st.sidebar.caption(f"📅 Last updated: {data_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption(f"⏱️ Cache expires in: {60 - data_timestamp.minute % 60} min")
    
    if minutes_until_refresh > 0 or seconds_until_refresh > 0:
        st.sidebar.caption(f"🔄 Auto-refresh in: {minutes_until_refresh}m {seconds_until_refresh}s")
    else:
        st.sidebar.caption(f"🔄 Refreshing now...")
    
    # Build trading dataframe
    with st.spinner("Computing trading features..."):
        # Pass 0 for min_profit since we'll filter by ROI % instead
        df = build_trading_dataframe(mapping_data, hourly_data, min_profit_per_flip=0)
    
    # Filter by minimum ROI percentage
    if len(df) > 0:
        df = df.filter(pl.col("edge_pct") >= min_roi_pct)
    
    # ========================================
    # PERFORMANCE TRACKER VIEW
    # ========================================
    if view_mode == "📊 Performance Tracker":
        display_performance_tracker()
        return  # Don't show trading dashboard
    
    # ========================================
    # TRADING DASHBOARD VIEW (DEFAULT)
    # ========================================
    
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
        st.warning(f"No items found with minimum {min_roi_pct}% ROI. Try lowering the threshold.")
        return
    
    # Add estimated GP/hour metric for sorting
    df = df.with_columns([
        (pl.col("net_edge") * pl.col("hourly_volume")).alias("estimated_gp_per_hour")
    ])
    
    # No confidence filtering - show ALL opportunities!
    
    if len(df) == 0:
        st.warning(f"""
        No items found meeting your filters.
        
        **Current Filter:**
        - Min ROI: **{min_roi_pct}%**
        
        **Try adjusting:**
        - Lower "Min ROI % per Flip"
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
    
    # 🔥 MARKET MICROSTRUCTURE ANALYSIS - The Missing Piece! 🔥
    with st.spinner("Analyzing market microstructure (time-to-fill, pressure, competition)..."):
        from core.market_microstructure import analyze_market_microstructure
        from core.competitive_analysis import analyze_competition
        allocated_df = analyze_market_microstructure(allocated_df)
        allocated_df = analyze_competition(allocated_df)
    
    # Apply minimum total profit filter
    if min_total_profit > 0 and len(allocated_df) > 0:
        # Filter by total profit (quantity × profit per item)
        allocated_df = allocated_df.filter(
            (pl.col("allocation_qty") * pl.col("net_edge")) >= min_total_profit
        )
        # Also update df to match filtered allocated_df items
        if len(allocated_df) > 0:  # Only filter if we still have items
            filtered_item_ids = allocated_df.select("item_id")
            df = df.join(filtered_item_ids, on="item_id", how="inner")
        else:
            # No items meet the profit threshold - show warning
            st.warning(f"⚠️ No trades found with total profit ≥ {format_gp(min_total_profit)}. Try lowering the Min Total Profit slider.")
            return  # Stop here, no data to show
    
    # Check if we have any data left
    if len(df) == 0 or len(allocated_df) == 0:
        st.info("📭 No opportunities match your current filters. Try adjusting the sliders.")
        return
    
    # 🔥🔥🔥 THE ULTIMATE PICK - BEST TRADE RIGHT NOW 🔥🔥🔥
    st.markdown("---")
    
    # Find THE best trade - filter out suspicious items and falling knives FIRST
    clean_df = df.filter(~pl.col('suspicious'))  # Remove manipulation
    
    if len(clean_df) > 0:
        # Check top candidates for falling knives
        top_candidates = clean_df.sort(['opportunity_score', 'edge_pct'], descending=[True, True]).head(10)
        
        best_overall = None
        falling_knife_analysis = None
        
        # Find first item that's NOT a falling knife
        for candidate_row in top_candidates.iter_rows(named=True):
            fk_check = detect_falling_knife(int(candidate_row['item_id']), candidate_row['name'])
            
            # Skip strong falling knives for THE ULTIMATE PICK
            if fk_check['trend'] not in ["STRONG_DOWN", "DOWN"]:
                best_overall = pl.DataFrame([candidate_row])
                falling_knife_analysis = fk_check
                break
        
        # If all top 10 are falling knives, take the best one but warn heavily
        if best_overall is None:
            best_overall = top_candidates.head(1)
            falling_knife_analysis = detect_falling_knife(
                int(top_candidates[0, 'item_id']),
                top_candidates[0, 'name']
            )
    else:
        best_overall = None
        falling_knife_analysis = None
    
    if best_overall is not None and len(best_overall) > 0:
        # Safely extract the top row as a dict
        try:
            best_list = best_overall.head(1).to_dicts()
            if not best_list:
                raise ValueError("no best row available")
            best_row = best_list[0]
        except Exception:
            st.warning("⚠️ Unable to determine THE ULTIMATE PICK due to insufficient data.")
            best_row = None
        
        # Create dramatic header
        st.markdown("""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; margin-bottom: 20px; box-shadow: 0 15px 40px rgba(0,0,0,0.4);'>
            <h1 style='color: white; font-size: 4em; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);'>🏆 THE ULTIMATE PICK 🏆</h1>
            <p style='color: #fff; font-size: 1.5em; margin: 10px 0 0 0; font-weight: 600;'>AI's #1 Recommendation Right Now</p>
        </div>
        """, unsafe_allow_html=True)
        
        if best_row is None:
            st.info("No suitable 'Ultimate Pick' available")
        else:
            best_buy = int(best_row.get('avgLowPrice', 0))
            best_sell = int(best_row.get('avgHighPrice', 0))
            best_profit_per = int(best_row.get('net_edge', 0))
            best_volume = int(best_row.get('hourly_volume', 0))
            best_strategy = best_row.get('strategy_type', 'UNKNOWN')
        
        # Show momentum status (already checked above)
        if falling_knife_analysis['is_falling_knife']:
            if falling_knife_analysis['trend'] == "STRONG_DOWN":
                st.error(f"🚨 **WARNING: FALLING KNIFE!** {falling_knife_analysis['warning']}")
                st.error("⚠️ **All top items are declining!** This is the best available but RISKY. Consider waiting for market to stabilize.")
            else:
                st.warning(f"⚠️ **Price Declining:** {falling_knife_analysis['warning']}")
                st.info("**Recommendation:** Exercise caution. Consider waiting for price to stabilize.")
        elif falling_knife_analysis['trend'] in ["UP", "STRONG_UP"]:
            st.success(f"✅ **Momentum Check:** Price trending {falling_knife_analysis['trend'].replace('_', ' ').title()} ({falling_knife_analysis['momentum_24h']:+.1f}% in 24h)")
        
        # Execution warning for low-volume items
        exec_confidence = best_row.get('execution_confidence', 100)
        if exec_confidence < 60:
            st.warning(f"⚠️ **Low Volume Warning:** Only {exec_confidence}% confident you'll fill at these prices. May take longer or fill at worse prices.")
        
        # Calculate suggested quantity based on strategy + volume
        best_ge_limit = int(best_row['limit']) if best_row['limit'] > 0 else 10000
        
        # Strategy-aware quantity suggestion
        if best_strategy == "INSTANT_FLIP":
            # Need high volume for instant flips
            if best_volume >= 50000:
                volume_factor = 0.8
                flip_confidence = "VERY HIGH"
            elif best_volume >= 10000:
                volume_factor = 0.5
                flip_confidence = "HIGH"
            else:
                volume_factor = 0.15
                flip_confidence = "LOW (risky for instant flips)"
        
        elif best_strategy == "SHORT_HOLD":
            # 1-2 day holds - more flexible
            if best_volume >= 20000:
                volume_factor = 0.7
                flip_confidence = "VERY HIGH"
            elif best_volume >= 5000:
                volume_factor = 0.5
                flip_confidence = "HIGH"
            else:
                volume_factor = 0.3
                flip_confidence = "MEDIUM (will flip in 1-2 days)"
        
        elif best_strategy == "SWING":
            # Week-long holds - volume less critical
            if best_volume >= 10000:
                volume_factor = 0.7
                flip_confidence = "HIGH (plenty of time)"
            elif best_volume >= 2000:
                volume_factor = 0.6
                flip_confidence = "MEDIUM-HIGH (week to sell)"
            else:
                volume_factor = 0.4
                flip_confidence = "MEDIUM (will sell over week)"
        
        else:
            # Default for other strategies
            if best_volume >= 50000:
                volume_factor = 0.8
                flip_confidence = "VERY HIGH"
            elif best_volume >= 10000:
                volume_factor = 0.5
                flip_confidence = "HIGH"
            elif best_volume >= 5000:
                volume_factor = 0.3
                flip_confidence = "MEDIUM"
            else:
                volume_factor = 0.2
                flip_confidence = "LOW-MEDIUM"
        
        best_qty = max(1, int(best_ge_limit * volume_factor))
        
        # Use realistic profit (slippage-adjusted) if available, otherwise net_edge
        realistic_profit_per = int(best_row.get('realistic_profit', best_profit_per))
        best_total_profit = realistic_profit_per * best_qty
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
            
            # Show realistic vs theoretical if there's slippage
            slippage_pct = best_row.get('slippage_pct', 0)
            if slippage_pct > 2.0:
                st.markdown(f"""
                **📥 BUY:**
                - Price: **{best_buy:,} GP** each
                - Quantity: **{best_qty:,}** items ({int(volume_factor*100)}% of GE limit)
                - Total Cost: **{format_gp(best_total_cost)}**
                
                **📤 SELL:**
                - Price: **{best_sell:,} GP** each
                - Expected Profit: **{format_gp(best_total_profit)}** 💎
                - ROI: **{(realistic_profit_per / best_buy * 100):.1f}%**
                
                ⚠️ *Profit adjusted for realistic fill prices (low volume)*
                **🎯 Flip Confidence:** {flip_confidence}
                """)
            else:
                st.markdown(f"""
                **📥 BUY:**
                - Price: **{best_buy:,} GP** each
                - Quantity: **{best_qty:,}** items ({int(volume_factor*100)}% of GE limit)
                - Total Cost: **{format_gp(best_total_cost)}**
                
                **📤 SELL:**
                - Price: **{best_sell:,} GP** each
                - Total Profit: **{format_gp(best_total_profit)}** 💎
                - ROI: **{best_row['edge_pct']:.1f}%**
                
                **🎯 Flip Confidence:** {flip_confidence}
                """)
        
        with bcol3:
            st.markdown("### ⏰ TIMING")
            
            # Simple, clear hold times based on strategy
            if best_strategy == "INSTANT_FLIP":
                hold_explanation = "**INSTANT FLIP**: List for sale immediately after buying"
                expected_time = "Same day (few hours)"
            elif best_strategy == "SHORT_HOLD":
                hold_explanation = "**SHORT HOLD**: Wait 1-2 days for price to rise"
                expected_time = "1-2 days"
            elif best_strategy == "SWING":
                hold_explanation = "**SWING TRADE**: Hold for a week, sell when profitable"
                expected_time = "3-7 days"
            else:
                hold_explanation = f"**{best_strategy}**"
                expected_time = "Varies"
            
            st.info(hold_explanation)
            
            # Simple max loss warning
            max_downside = best_row.get('max_downside_pct', 10)
            max_loss_gp = int(best_buy * max_downside / 100)
            
            if max_downside >= 15:
                st.warning(f"⚠️ **Risk:** If trade fails, could lose up to {max_downside}% (~{format_gp(max_loss_gp)} per item)")
            
            st.markdown(f"""
            **⏰ Expected Hold Time:** {expected_time}
            
            **📊 Trading Volume:** {best_volume:,}/hour
            
            **💡 Simple Steps:**
            1. Buy at {best_buy:,} GP
            2. {"List immediately" if best_strategy == "INSTANT_FLIP" else f"Wait {expected_time}"}
            3. Sell at {best_sell:,} GP
            """)
        
        # Market Microstructure Insights
        st.markdown("---")
        st.markdown("### ⚡ Market Microstructure Analysis")
        st.caption("💡 **Understanding execution conditions** - How fast your orders will fill and market pressure")
        
        micro_col1, micro_col2, micro_col3 = st.columns(3)
        
        with micro_col1:
            # Time to fill
            fill_hours = best_row.get('time_to_fill_hours', 0)
            fill_difficulty = best_row.get('fill_difficulty', 'UNKNOWN')
            fill_minutes = int(fill_hours * 60)
            
            if fill_hours < 1:
                time_display = f"{fill_minutes} min"
            else:
                time_display = f"{fill_hours:.1f} hrs"
            
            st.metric("⏱️ Time to Fill", time_display, fill_difficulty, help="How long it will take to buy/sell your full quantity based on market volume")
            if fill_hours > 4:
                st.caption("⚠️ Slow fills - may take multiple hours to accumulate position")
            elif fill_hours < 0.5:
                st.caption("✅ Very fast - fills within 30 minutes")
            else:
                st.caption("📊 Assumes you capture 30% of hourly market volume")
        
        with micro_col2:
            # Buy/Sell pressure
            pressure_imbalance = best_row.get('pressure_ratio', 1.0)
            market_pressure = best_row.get('pressure_direction', '➡️ BALANCED')
            pressure_forecast = best_row.get('price_forecast', 'FLAT')
            
            st.metric("📊 Market Pressure", market_pressure, pressure_forecast, help="Are more people buying or selling? Helps predict if price will rise or fall")
            
            if "SELL PRESSURE" in market_pressure:
                st.caption("🔴 More sellers than buyers = price likely dropping")
            elif "BUY PRESSURE" in market_pressure:
                st.caption("🟢 More buyers than sellers = price likely rising")
            else:
                st.caption("⚪ Equal buyers and sellers = stable price")
        
        with micro_col3:
            # Market depth
            depth_score = best_row.get('market_depth_score', 50)
            volume_to_qty_ratio = best_volume / max(best_qty, 1)
            
            if volume_to_qty_ratio >= 10:
                depth_status = "🌊 DEEP"
                depth_color = "🟢"
                depth_explanation = "Market has 10x+ your order size - plenty of liquidity"
            elif volume_to_qty_ratio >= 3:
                depth_status = "💧 MODERATE"
                depth_color = "🟡"
                depth_explanation = "Market has 3-10x your order size - decent liquidity"
            else:
                depth_status = "🏜️ SHALLOW"
                depth_color = "🔴"
                depth_explanation = "Market has less than 3x your order size - low liquidity"
            
            st.metric("💧 Market Depth", depth_status, f"{volume_to_qty_ratio:.1f}x your order", help="Can the market absorb your order without moving the price? Higher = better")
            st.caption(f"{depth_color} {depth_explanation}")
        
        # Combined execution warning
        execution_warnings = []
        if fill_hours > 4:
            execution_warnings.append("⏰ SLOW FILLS expected - position may take hours to build")
        if "STRONG SELL PRESSURE" in market_pressure or "SELL PRESSURE" in market_pressure:
            execution_warnings.append("🔴 SELL PRESSURE - price likely declining, avoid buying now")
        if volume_to_qty_ratio < 2:
            execution_warnings.append("🏜️ LOW LIQUIDITY - large orders will move market")
        
        if execution_warnings:
            st.warning("**⚠️ EXECUTION RISKS:**\n- " + "\n- ".join(execution_warnings))
        else:
            st.success("✅ **EXCELLENT EXECUTION CONDITIONS:** Fast fills, balanced pressure, good liquidity!")
        
        # Competition Analysis
        st.markdown("---")
        st.markdown("### 🏆 Competition Analysis")
        st.caption("💡 **How crowded is this trade?** - Less competition = easier profits and faster fills")
        
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            competition_score = best_row.get('competition_score', 50)
            competition_level = best_row.get('competition_level', '🟡 MODERATE')
            
            st.metric("🎯 Competition Level", competition_level, f"{competition_score}/100", help="Based on volume, spread tightness, and trader activity. Lower = easier to profit")
            
            if competition_score <= 20:
                st.success("🏝️ **BLUE OCEAN!** Very few traders competing. Easy profits!")
            elif competition_score <= 40:
                st.info("🟢 **LOW COMPETITION**: Good opportunity, market not crowded")
            elif competition_score <= 60:
                st.warning("🟡 **MODERATE COMPETITION**: Some traders active, still workable")
            else:
                st.error("🔴 **HIGH COMPETITION**: Very crowded, harder to get optimal fills")
        
        with comp_col2:
            ease_of_profit = best_row.get('ease_of_profit', '⭐ MODERATE')
            bot_risk = best_row.get('bot_risk_score', 0)
            bot_warning = best_row.get('bot_warning', None)
            
            st.metric("⭐ Ease of Profit", ease_of_profit, help="How easy is it to make profit on this trade? Based on competition level")
            
            if bot_warning:
                st.warning(f"🤖 {bot_warning}")
                st.caption("⚠️ Bots compete instantly - your orders may not fill at target price")
            else:
                st.success("✅ No bot activity detected")
                st.caption("👤 Mostly human traders - more predictable market")
        
        with comp_col3:
            # Show what this means
            st.markdown("**Why This Matters:**")
            if competition_score <= 40:
                st.success("""
                - ✅ Less competition for optimal prices  
                - ✅ Your orders fill faster  
                - ✅ Less bot interference  
                - ✅ More predictable profits
                """)
            elif competition_score <= 60:
                st.info("""
                - 🟡 Some traders competing  
                - 🟡 May need patience for fills  
                - 🟡 Watch for price changes
                """)
            else:
                st.error("""
                - 🔴 Many traders competing  
                - 🔴 Harder to get optimal fills  
                - 🔴 Possible bot activity  
                - 🔴 May need patience
                """)
        
        # Historical Price Analysis
        st.markdown("---")
        st.markdown("### 📊 Historical Price Analysis")
        
        with st.spinner("Fetching 30-day price history..."):
            try:
                # Fetch timeseries data for this item
                best_item_id = int(best_row['item_id'])
                timeseries_df = load_timeseries(best_item_id, timestep="24h")
                
                if timeseries_df is not None and len(timeseries_df) > 0:
                    # Calculate historical averages
                    import numpy as np
                    
                    # Get average prices (use avgHighPrice for selling price)
                    prices = timeseries_df.select('avgHighPrice').to_series().to_list()
                    prices = [p for p in prices if p is not None]
                    
                    if len(prices) >= 7:
                        current_price = best_sell
                        last_7d_avg = int(np.mean(prices[-7:]))
                        last_30d_avg = int(np.mean(prices))
                        last_30d_low = int(np.min(prices))
                        last_30d_high = int(np.max(prices))
                        
                        # Calculate trend
                        recent_trend = "RISING 📈" if prices[-1] > np.mean(prices[-7:]) else "FALLING 📉" if prices[-1] < np.mean(prices[-7:]) else "STABLE ➡️"
                        
                        # Show in 4 columns
                        hcol1, hcol2, hcol3, hcol4 = st.columns(4)
                        
                        with hcol1:
                            st.metric("Last Week Avg", f"{last_7d_avg:,} GP",
                                     delta=f"{((current_price - last_7d_avg) / last_7d_avg * 100):+.1f}%")
                        
                        with hcol2:
                            st.metric("Last Month Avg", f"{last_30d_avg:,} GP",
                                     delta=f"{((current_price - last_30d_avg) / last_30d_avg * 100):+.1f}%")
                        
                        with hcol3:
                            st.metric("30-Day Range", f"{last_30d_low:,} - {last_30d_high:,} GP")
                        
                        with hcol4:
                            st.metric("Current Trend", recent_trend)
                        
                        # Value assessment
                        vs_avg_pct = ((current_price - last_30d_avg) / last_30d_avg) * 100
                        
                        if vs_avg_pct < -10:
                            st.success(f"💎 **BARGAIN DETECTED!** Current price is {abs(vs_avg_pct):.1f}% BELOW 30-day average!")
                        elif vs_avg_pct > 10:
                            st.warning(f"⚠️ **PREMIUM PRICING:** Current price is {vs_avg_pct:.1f}% ABOVE 30-day average. Consider waiting.")
                        else:
                            st.info(f"📊 **FAIR VALUE:** Price is within {abs(vs_avg_pct):.1f}% of 30-day average.")
                        
                        # Day-of-Week Analysis
                        st.markdown("---")
                        st.markdown("### 📅 Historical Day-of-Week Patterns")
                        
                        day_analysis = analyze_day_of_week_patterns(timeseries_df)
                        
                        if day_analysis['has_data']:
                            # Show historical pattern data (informational only)
                            dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
                            
                            with dcol1:
                                st.metric("💰 Historically Cheapest", day_analysis['best_buy_day'], 
                                         f"Avg: {day_analysis['best_buy_price']:,} GP")
                            
                            with dcol2:
                                st.metric("💎 Historically Highest", day_analysis['best_sell_day'],
                                         f"Avg: {day_analysis['best_sell_price']:,} GP")
                            
                            with dcol3:
                                timing_boost = day_analysis['timing_advantage_gp']
                                timing_pct = day_analysis.get('timing_advantage_pct', 0)
                                st.metric("📊 Avg Price Swing", f"{timing_boost:,} GP",
                                         f"{timing_pct:.1f}%" if timing_pct > 0 else "None")
                            
                            # Show pattern explanation (informational)
                            st.caption("📊 **Historical Pattern** (based on 30-day average prices per day of week)")
                            
                            # Show pattern if detected
                            if day_analysis.get('pattern'):
                                st.caption(day_analysis['pattern'])
                        
                        # Item category prediction
                        category_pattern = detect_item_category_pattern(best_row['name'])
                        if category_pattern:
                            st.caption(category_pattern)
                    
                    else:
                        st.info("📊 Limited historical data available (less than 7 days)")
                else:
                    st.info("📊 No historical price data available for this item")
            except Exception as e:
                st.error(f"❌ Error fetching historical data: {str(e)}")
        
        # Price History Charts (Collapsible)
        with st.expander("📊 View Price History Charts", expanded=False):
            st.markdown("**Compare price trends across different timeframes**")
            
            best_item_id = int(best_row['item_id'])
            
            # Create tabs for different timeframes
            tab1d, tab1w, tab1m, tab6m = st.tabs(["📅 1 Day", "📅 1 Week", "📅 1 Month", "📅 6 Months"])
            
            with tab1d:
                with st.spinner("Loading 1-day chart..."):
                    df_5m = load_timeseries(best_item_id, timestep="5m")
                    if df_5m is not None and len(df_5m) > 0:
                        # Filter last 24 hours
                        cutoff = datetime.now() - timedelta(hours=24)
                        df_5m = df_5m.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_5m) > 0:
                            fig = create_price_chart(df_5m, title="24-Hour Price Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_5m["avgHighPrice"].mean())
                            avg_low = int(df_5m["avgLowPrice"].mean())
                            st.caption(f"24h avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 24 hours")
                    else:
                        st.caption("No 5-minute data available")
            
            with tab1w:
                with st.spinner("Loading 1-week chart..."):
                    df_1h = load_timeseries(best_item_id, timestep="1h")
                    if df_1h is not None and len(df_1h) > 0:
                        # Filter last 7 days
                        cutoff = datetime.now() - timedelta(days=7)
                        df_1h = df_1h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_1h) > 0:
                            fig = create_price_chart(df_1h, title="7-Day Price Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_1h["avgHighPrice"].mean())
                            avg_low = int(df_1h["avgLowPrice"].mean())
                            st.caption(f"7-day avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 7 days")
                    else:
                        st.caption("No 1-hour data available")
            
            with tab1m:
                with st.spinner("Loading 1-month chart..."):
                    df_6h = load_timeseries(best_item_id, timestep="6h")
                    if df_6h is not None and len(df_6h) > 0:
                        # Filter last 30 days
                        cutoff = datetime.now() - timedelta(days=30)
                        df_6h = df_6h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_6h) > 0:
                            fig = create_price_chart(df_6h, title="30-Day Price Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_6h["avgHighPrice"].mean())
                            avg_low = int(df_6h["avgLowPrice"].mean())
                            st.caption(f"30-day avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 30 days")
                    else:
                        st.caption("No 6-hour data available")
            
            with tab6m:
                with st.spinner("Loading 6-month chart..."):
                    df_24h = load_timeseries(best_item_id, timestep="24h")
                    if df_24h is not None and len(df_24h) > 0:
                        # Filter last 180 days
                        cutoff = datetime.now() - timedelta(days=180)
                        df_24h = df_24h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_24h) > 0:
                            fig = create_price_chart(df_24h, title="6-Month Price Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_24h["avgHighPrice"].mean())
                            avg_low = int(df_24h["avgLowPrice"].mean())
                            st.caption(f"6-month avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 6 months")
                    else:
                        st.caption("No 24-hour data available")
        
        # Recent Trades - 5 minute data
        st.markdown("---")
        st.markdown("### 📈 Recent Trading Activity")
        
        with st.spinner("Loading recent trades..."):
            try:
                # Fetch 5-minute data for detailed recent trades
                best_item_id = int(best_row['item_id'])
                trades_df = load_timeseries(best_item_id, timestep="5m")
                
                if trades_df is not None and len(trades_df) > 0:
                    # Get last 10 trades (50 minutes of data) and sort newest first
                    recent_trades = trades_df.tail(10).sort('datetime', descending=True)
                    
                    # Create display dataframe with CST timezone
                    if 'datetime' in recent_trades.columns:
                        display_df = recent_trades.select([
                            # Convert UTC to CST (UTC-6) and format with date
                            (pl.col('datetime').dt.offset_by('-6h')).dt.strftime('%m/%d %I:%M %p').alias('Time (CST)'),
                            pl.col('avgLowPrice').fill_null(0).cast(pl.Int64).alias('Buy Price'),
                            pl.col('lowPriceVolume').fill_null(0).cast(pl.Int64).alias('Buy Volume'),
                            pl.col('avgHighPrice').fill_null(0).cast(pl.Int64).alias('Sell Price'),
                            pl.col('highPriceVolume').fill_null(0).cast(pl.Int64).alias('Sell Volume')
                        ])
                        
                        # Convert to pandas for display
                        display_pandas = display_df.to_pandas()
                        
                        st.dataframe(
                            display_pandas,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.caption("📊 Last 10 trades (5-minute intervals). '0' means no trades in that period.")
                    else:
                        st.info("📊 Recent trade data not available")
                else:
                    st.info("📊 No recent trading activity available")
            except Exception as e:
                st.error(f"❌ Error fetching recent trades: {str(e)}")
        
        # Big execution button
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🚀 EXECUTE THE ULTIMATE PICK NOW! 🚀", key="ultimate_pick_execute", use_container_width=True):
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
        
        with col2:
            if st.button("📝 Log Trade", key="log_ultimate_pick", use_container_width=True, help="Record this trade in Performance Tracker"):
                # Log as OPEN trade
                tracker = st.session_state.performance_tracker
                trade_data = {
                    "item_name": best_row['name'],
                    "item_id": int(best_row['item_id']),
                    "buy_price": best_buy,
                    "sell_price": best_sell,
                    "quantity": best_qty,
                    "profit_per_item": int(best_row['net_edge']),
                    "total_profit": best_total_profit,
                    "strategy": best_row.get('strategy_type', 'STANDARD'),
                    "status": "OPEN",
                    "entry_time": datetime.now().isoformat(),
                    "predicted_opportunity_score": int(best_row.get('opportunity_score', 0)),
                    "predicted_risk_score": int(best_row.get('risk_score', 50)),
                }
                
                if tracker.log_trade(trade_data):
                    st.success(f"✅ Trade logged! Check Performance Tracker to see stats.")
                else:
                    st.error("Failed to log trade")
        
        # Comparison with other top picks
        if len(df) > 1:
            st.markdown("---")
            st.markdown("### 📊 TOP 5 BEST VALUE TRADES")
            
            # Calculate realistic value score for all items (profit × tradability)
            comparison_list = []
            for row in df.head(20).iter_rows(named=True):  # Check top 20 by opp score
                buy_price = int(row['avgLowPrice'])
                sell_price = int(row['avgHighPrice'])
                profit_per = int(row['net_edge'])
                volume = int(row['hourly_volume'])
                item_id = int(row['item_id'])
                item_name = row['name']
                
                # Calculate quantity (same logic as TOP 3)
                strategy_type = row.get('strategy_type', 'UNKNOWN')
                ge_limit = int(row['limit']) if row['limit'] > 0 else 10000
                
                if strategy_type == "INSTANT_FLIP":
                    volume_factor = 0.5 if volume >= 10000 else 0.15
                    speed_label = "⚡ Instant"
                elif strategy_type == "SHORT_HOLD":
                    volume_factor = 0.5 if volume >= 5000 else 0.3
                    speed_label = "📅 1-2 days"
                elif strategy_type == "SWING":
                    volume_factor = 0.6 if volume >= 2000 else 0.4
                    speed_label = "📈 3-7 days"
                else:
                    volume_factor = 0.5 if volume >= 10000 else 0.2
                    speed_label = "🎯 Varies"
                
                max_affordable = int(total_capital * 0.4 / buy_price) if buy_price > 0 else 0
                volume_based_qty = int(volume * volume_factor * 24)
                qty = min(ge_limit, max_affordable, volume_based_qty, 10000)
                qty = max(qty, 1)
                
                total_profit = profit_per * qty
                
                # Calculate tradability score (how easy to buy/sell)
                # High volume = easy to trade, low volume = hard to trade
                if volume >= 50000:
                    tradability = 1.0  # Very easy
                    trade_label = "🟢 Very Easy"
                elif volume >= 10000:
                    tradability = 0.8  # Easy
                    trade_label = "🟢 Easy"
                elif volume >= 5000:
                    tradability = 0.6  # Moderate
                    trade_label = "🟡 Moderate"
                elif volume >= 1000:
                    tradability = 0.4  # Slow
                    trade_label = "🟠 Slow"
                else:
                    tradability = 0.2  # Very slow/risky
                    trade_label = "🔴 Very Slow"
                
                # Realistic value = total profit × tradability
                # This balances high profit with tradability
                realistic_value = total_profit * tradability
                
                comparison_list.append({
                    "name": item_name,
                    "item_id": item_id,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "qty": qty,
                    "total_profit": total_profit,
                    "realistic_value": realistic_value,
                    "tradability": tradability,
                    "trade_label": trade_label,
                    "speed_label": speed_label,
                    "volume": volume,
                    "opp": int(row.get('opportunity_score', 0)),
                    "risk": int(row.get('risk_score', 50)),
                    "suspicious": row.get('suspicious', False),
                })
            
            # Sort by realistic value (profit × tradability) descending
            comparison_list.sort(key=lambda x: x['realistic_value'], reverse=True)
            
            # Filter out suspicious items but count them
            clean_items = [item for item in comparison_list if not item.get('suspicious', False)]
            suspicious_count = len([item for item in comparison_list[:10] if item.get('suspicious', False)])
            
            # Store top 5 in session state to prevent reordering on dropdown change
            if 'top_5_items' not in st.session_state or st.session_state.get('force_refresh_top5', False):
                st.session_state.top_5_items = clean_items[:5]
                st.session_state.force_refresh_top5 = False
            
            top_5 = st.session_state.top_5_items
            
            if suspicious_count > 0:
                st.warning(f"⚠️ Filtered out {suspicious_count} suspicious item(s) with manipulation patterns.")
            
            # Check top 5 items for falling knives (use cached session state)
            st.info("🔍 Checking price momentum for top 5 items...")
            
            falling_knife_checks = {}
            for item in top_5:
                fk_analysis = detect_falling_knife(item['item_id'], item['name'])
                falling_knife_checks[item['item_id']] = fk_analysis
            
            # Build table data
            comparison_data = []
            for idx, item in enumerate(top_5, 1):
                fk = falling_knife_checks[item['item_id']]
                
                # Determine trend emoji
                if fk['trend'] == "STRONG_DOWN":
                    trend_display = "🚨 FALLING"
                elif fk['trend'] == "DOWN":
                    trend_display = "📉 Down"
                elif fk['trend'] == "UP":
                    trend_display = "📈 Up"
                elif fk['trend'] == "STRONG_UP":
                    trend_display = "🚀 Rising"
                elif fk['trend'] == "NEUTRAL":
                    trend_display = "➡️ Stable"
                else:
                    trend_display = "❓ Unknown"
                
                comparison_data.append({
                    "Rank": f"#{idx}" + (" 🏆" if idx == 1 else ""),
                    "Item": item['name'][:20],
                    "Buy Price": f"{item['buy_price']:,} GP",
                    "Sell Price": f"{item['sell_price']:,} GP",
                    "Qty to Buy": f"{item['qty']:,}",
                    "Total Profit": format_gp(item['total_profit']),
                    "Speed": item['speed_label'],
                    "24h Trend": trend_display,
                    "Tradability": item['trade_label'],
                })
            
            st.table(comparison_data)
            st.caption("🏆 = BEST VALUE | 24h Trend: 🚨 FALLING = Avoid, 📉 = Caution, ➡️ = Stable, 📈 = Good, 🚀 = Great | Tradability = How easy to buy/sell")
            
            # Interactive: Click to view charts
            st.markdown("---")
            st.markdown("#### 📊 Click an item to view price charts")
            
            selected_item = st.selectbox(
                "Select an item to analyze:",
                options=[f"#{idx+1} - {item['name']}" for idx, item in enumerate(top_5)],
                key="top5_selector"
            )
            
            if selected_item:
                # Extract item index from selection
                selected_idx = int(selected_item.split(" - ")[0].replace("#", "")) - 1
                selected_data = top_5[selected_idx]
                selected_item_id = selected_data['item_id']
                selected_name = selected_data['name']
                
                st.markdown(f"### 📈 Price History: {selected_name}")
                
                # Create tabs for different timeframes
                tab1d, tab1w, tab1m, tab6m = st.tabs(["📅 1 Day", "📅 1 Week", "📅 1 Month", "📅 6 Months"])
                
                with tab1d:
                    with st.spinner("Loading 1-day chart..."):
                        df_5m = load_timeseries(selected_item_id, timestep="5m")
                        if df_5m is not None and len(df_5m) > 0:
                            cutoff = datetime.now() - timedelta(hours=24)
                            df_5m = df_5m.filter(pl.col("datetime") >= cutoff)
                            
                            if len(df_5m) > 0:
                                fig = create_price_chart(df_5m, title=f"{selected_name} - 24 Hour Trend")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No data available for last 24 hours")
                        else:
                            st.info("No 5-minute data available")
                
                with tab1w:
                    with st.spinner("Loading 1-week chart..."):
                        df_1h = load_timeseries(selected_item_id, timestep="1h")
                        if df_1h is not None and len(df_1h) > 0:
                            cutoff = datetime.now() - timedelta(days=7)
                            df_1h = df_1h.filter(pl.col("datetime") >= cutoff)
                            
                            if len(df_1h) > 0:
                                fig = create_price_chart(df_1h, title=f"{selected_name} - 7 Day Trend")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No data available for last 7 days")
                        else:
                            st.info("No hourly data available")
                
                with tab1m:
                    with st.spinner("Loading 1-month chart..."):
                        df_6h = load_timeseries(selected_item_id, timestep="6h")
                        if df_6h is not None and len(df_6h) > 0:
                            cutoff = datetime.now() - timedelta(days=30)
                            df_6h = df_6h.filter(pl.col("datetime") >= cutoff)
                            
                            if len(df_6h) > 0:
                                fig = create_price_chart(df_6h, title=f"{selected_name} - 30 Day Trend")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No data available for last 30 days")
                        else:
                            st.info("No 6-hour data available")
                
                with tab6m:
                    with st.spinner("Loading 6-month chart..."):
                        df_24h = load_timeseries(selected_item_id, timestep="24h")
                        if df_24h is not None and len(df_24h) > 0:
                            cutoff = datetime.now() - timedelta(days=180)
                            df_24h = df_24h.filter(pl.col("datetime") >= cutoff)
                            
                            if len(df_24h) > 0:
                                fig = create_price_chart(df_24h, title=f"{selected_name} - 6 Month Trend")
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No data available for last 6 months")
                        else:
                            st.info("No daily data available")
        
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

        
        st.caption(f"💡 **Why this is THE pick:** Highest-scoring item with {falling_knife_analysis['trend'].replace('_', ' ').lower()} price momentum and no manipulation flags.")
    else:
        st.warning("⚠️ **No clean trades available right now.** All top items show manipulation or strong declining prices. Check back later or adjust filters.")
    
    # 💎💎💎 QUICK TRADE IDEAS - READY TO EXECUTE 💎💎💎
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
    
    # Get top 3 from FULL dataframe (not filtered by selected strategy)
    top_trades = df.sort(['opportunity_score', 'edge_pct'], descending=[True, True]).head(3)
    
    trade_cols = st.columns(3)
    
    for idx, (col, row) in enumerate(zip(trade_cols, top_trades.iter_rows(named=True)), 1):
        with col:
            buy_price = int(row['avgLowPrice'])
            sell_price = int(row['avgHighPrice'])
            profit_per = int(row['net_edge'])
            volume = int(row['hourly_volume'])
            strategy_type = row.get('strategy_type', 'UNKNOWN')
            ge_limit = int(row['limit']) if row['limit'] > 0 else 10000
            
            # Calculate quantity based on strategy and volume (same logic as Ultimate Pick)
            if strategy_type == "INSTANT_FLIP":
                volume_factor = 0.5 if volume >= 10000 else 0.15
            elif strategy_type == "SHORT_HOLD":
                volume_factor = 0.5 if volume >= 5000 else 0.3
            elif strategy_type == "SWING":
                volume_factor = 0.6 if volume >= 2000 else 0.4
            else:
                volume_factor = 0.5 if volume >= 10000 else 0.2
            
            # Calculate safe quantity
            max_affordable = int(total_capital * 0.4 / buy_price) if buy_price > 0 else 0
            volume_based_qty = int(volume * volume_factor * 24)
            qty = min(ge_limit, max_affordable, volume_based_qty, 10000)
            qty = max(qty, 1)
            
            total_profit = profit_per * qty
            total_cost = buy_price * qty
            
            # Simple hold time based on strategy
            if strategy_type == "INSTANT_FLIP":
                hold_time = "Same day"
                strategy_emoji = "⚡"
            elif strategy_type == "SHORT_HOLD":
                hold_time = "1-2 days"
                strategy_emoji = "📅"
            elif strategy_type == "SWING":
                hold_time = "3-7 days"
                strategy_emoji = "📈"
            else:
                hold_time = "Varies"
                strategy_emoji = "🎯"
            
            # Color code by rank
            rank_emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            
            st.markdown(f"### {rank_emoji} #{idx}: {row['name']}")
            
            # Simple strategy display
            st.info(f"{strategy_emoji} **{strategy_type}** | ⏰ Hold: {hold_time}")
            
            # Buy section
            st.markdown("**📥 BUY ORDER:**")
            st.code(f"""
Price:    {buy_price:,} GP each
Quantity: {qty:,} items
Total:    {format_gp(total_cost)}
            """)
            
            # Sell section
            st.markdown("**📤 SELL ORDER:**")
            st.code(f"""
Price:    {sell_price:,} GP each
Quantity: {qty:,} items  
Profit:   {format_gp(total_profit)}
            """)
            
            # Profit summary
            st.success(f"**💰 PROFIT: {format_gp(total_profit)} ({row['edge_pct']:.1f}% ROI)**")
            st.caption(f"🎯 Opportunity: {int(row.get('opportunity_score', 0))}/100 | ⚠️ Risk: {int(row.get('risk_score', 50))}/100")
            
            # Show simple trade execution steps
            if strategy_type == "INSTANT_FLIP":
                st.markdown(f"""
**📋 SIMPLE STEPS:**

1. Go to GE
2. Buy {qty:,}x {row['name']} at {buy_price:,} GP
3. **Immediately** list for sale at {sell_price:,} GP
4. Wait for orders to fill (same day)
5. Profit: {format_gp(total_profit)} 🎉
                """)
            elif strategy_type == "SHORT_HOLD":
                st.markdown(f"""
**📋 SIMPLE STEPS:**

1. Go to GE
2. Buy {qty:,}x {row['name']} at {buy_price:,} GP
3. Wait 1-2 days for price to rise
4. List for sale at {sell_price:,} GP (or higher)
5. Profit: {format_gp(total_profit)}+ 🎉
                """)
            elif strategy_type == "SWING":
                st.markdown(f"""
**📋 SIMPLE STEPS:**

1. Go to GE
2. Buy {qty:,}x {row['name']} at {buy_price:,} GP
3. Hold for 3-7 days
4. Watch price, sell when profitable
5. Target profit: {format_gp(total_profit)}+ 🎉
                """)
            else:
                st.markdown(f"""
**📋 SIMPLE STEPS:**

1. Go to GE
2. Buy {qty:,}x {row['name']} at {buy_price:,} GP
3. List for sale at {sell_price:,} GP
4. Profit: {format_gp(total_profit)} 🎉
                """)
            
            # Price history graphs
            st.markdown("---")
            st.markdown("**📊 Price History**")
            
            item_id = int(row['item_id'])
            
            # Create tabs for different timeframes
            tab1d, tab1w, tab1m, tab6m = st.tabs(["📅 1 Day", "📅 1 Week", "📅 1 Month", "📅 6 Months"])
            
            with tab1d:
                with st.spinner("Loading 1-day chart..."):
                    df_5m = load_timeseries(item_id, timestep="5m")
                    if df_5m is not None and len(df_5m) > 0:
                        # Filter last 24 hours
                        cutoff = datetime.now() - timedelta(hours=24)
                        df_5m = df_5m.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_5m) > 0:
                            fig = create_price_chart(df_5m, title="24-Hour Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_5m["avgHighPrice"].mean())
                            avg_low = int(df_5m["avgLowPrice"].mean())
                            st.caption(f"24h avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 24 hours")
                    else:
                        st.caption("No 5-minute data available")
            
            with tab1w:
                with st.spinner("Loading 1-week chart..."):
                    df_1h = load_timeseries(item_id, timestep="1h")
                    if df_1h is not None and len(df_1h) > 0:
                        # Filter last 7 days
                        cutoff = datetime.now() - timedelta(days=7)
                        df_1h = df_1h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_1h) > 0:
                            fig = create_price_chart(df_1h, title="7-Day Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_1h["avgHighPrice"].mean())
                            avg_low = int(df_1h["avgLowPrice"].mean())
                            st.caption(f"7-day avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 7 days")
                    else:
                        st.caption("No 1-hour data available")
            
            with tab1m:
                with st.spinner("Loading 1-month chart..."):
                    df_6h = load_timeseries(item_id, timestep="6h")
                    if df_6h is not None and len(df_6h) > 0:
                        # Filter last 30 days
                        cutoff = datetime.now() - timedelta(days=30)
                        df_6h = df_6h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_6h) > 0:
                            fig = create_price_chart(df_6h, title="30-Day Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_6h["avgHighPrice"].mean())
                            avg_low = int(df_6h["avgLowPrice"].mean())
                            st.caption(f"30-day avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 30 days")
                    else:
                        st.caption("No 6-hour data available")
            
            with tab6m:
                with st.spinner("Loading 6-month chart..."):
                    df_24h = load_timeseries(item_id, timestep="24h")
                    if df_24h is not None and len(df_24h) > 0:
                        # Filter last 180 days
                        cutoff = datetime.now() - timedelta(days=180)
                        df_24h = df_24h.filter(pl.col("datetime") >= cutoff)
                        
                        if len(df_24h) > 0:
                            fig = create_price_chart(df_24h, title="6-Month Trend")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show price stats
                            avg_high = int(df_24h["avgHighPrice"].mean())
                            avg_low = int(df_24h["avgLowPrice"].mean())
                            st.caption(f"6-month avg: High {avg_high:,} GP | Low {avg_low:,} GP")
                        else:
                            st.caption("No data for last 6 months")
                    else:
                        st.caption("No 24-hour data available")
    
    # 📦📦📦 MASS QUANTITY STRATEGY 📦📦📦
    st.markdown("---")
    st.markdown("## 📦 MASS QUANTITY - High Volume, No Tax")
    
    with st.expander("ℹ️ WHAT IS MASS QUANTITY TRADING?", expanded=False):
        st.markdown("""
        **Trade THOUSANDS of cheap items with NO GE TAX:**
        
        - 💸 **No Tax**: Items under 100 GP = **0% GE tax** (huge advantage!)
        - 📦 **High Volume**: Need daily volume to support mass trading
        - 🔄 **Fast Flips**: Quick turnover on thousands of units
        - 💰 **Small margins, BIG totals**: 5 GP profit × 10,000 units = 50K profit!
        
        **Perfect items:**
        - Potions (4-dose, 3-dose, 2-dose)
        - Darts / Dart tips
        - Runes (elemental, catalytic)
        - Arrows / Bolts
        - Herbs
        
        **Why it works:**
        - No 2% tax = higher margins
        - High liquidity = fast trades
        - Low capital per unit = can trade HUGE quantities
        
        **Example:**
        - Adamant dart tips: 50 GP buy, 55 GP sell
        - Profit: 5 GP × 10,000 = 50,000 GP
        - No tax taken!
        """)
    
    st.caption("Perfect for: Low capital, fast flips, trading thousands of units")
    
    # Filter for mass quantity opportunities
    if len(df) > 0:
        mass_opps = df.filter(
            (pl.col('avgHighPrice') < 100) &  # Under 100 GP = no tax
            (pl.col('hourly_volume') > 100) &  # Good volume
            (pl.col('net_edge') > 0)  # Profitable
        )
        
        if len(mass_opps) > 0:
            # Calculate daily volume (more important for mass trading)
            mass_opps = mass_opps.with_columns([
                (pl.col('hourly_volume') * 24).alias('daily_volume')
            ])
            
            # Sort by daily profit potential
            mass_opps = mass_opps.with_columns([
                (pl.col('net_edge') * pl.col('daily_volume')).alias('daily_profit_potential')
            ]).sort('daily_profit_potential', descending=True)
            
            st.success(f"📦 Found {len(mass_opps)} mass quantity opportunities!")
            
            for row in mass_opps.head(10).iter_rows(named=True):
                item_name = row['name']
                buy_price = int(row['avgLowPrice'])
                sell_price = int(row['avgHighPrice'])  # No tax under 100 GP!
                profit_per = int(row['net_edge'])
                hourly_vol = int(row['hourly_volume'])
                daily_vol = int(row['daily_volume'])
                ge_limit = int(row.get('limit', 10000))
                
                # Calculate realistic quantity based on daily volume
                # Can safely trade up to 10% of daily volume
                safe_quantity = min(ge_limit, int(daily_vol * 0.1))
                total_profit = profit_per * safe_quantity
                
                with st.expander(f"📦 {item_name} - {format_gp(total_profit)} daily potential", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        **💰 THE TRADE:**
                        - Buy: **{buy_price} GP** each
                        - Sell: **{sell_price} GP** each
                        - Profit: **{profit_per} GP** per unit
                        - **NO GE TAX!** ✅
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **📊 VOLUME:**
                        - Hourly: {hourly_vol:,}
                        - Daily: **{daily_vol:,}**
                        - GE Limit: {ge_limit:,}
                        - Safe Qty: **{safe_quantity:,}**
                        """)
                    
                    with col3:
                        st.markdown(f"""
                        **💎 POTENTIAL:**
                        - Per Unit: {profit_per} GP
                        - {safe_quantity:,} units
                        - **Total: {format_gp(total_profit)}**
                        - ROI: {row.get('edge_pct', 0):.1f}%
                        """)
                    
                    if daily_vol > 1000:
                        st.success(f"🔥 HIGH LIQUIDITY! {daily_vol:,} daily volume supports mass trading!")
                    elif daily_vol < 500:
                        st.warning(f"⚠️ Low liquidity. May take time to move {safe_quantity:,} units.")
                    else:
                        st.info(f"✅ Moderate liquidity. Should move {safe_quantity:,} units in a day.")
        else:
            st.info("No mass quantity opportunities at current prices.")
    
    # 🔍🔍🔍 ITEM PRICE HISTORY SEARCH 🔍🔍🔍
    if 'searched_item' in st.session_state and st.session_state.searched_item:
        st.markdown("---")
        st.markdown(f"## 🔍 Price History: {st.session_state.searched_item}")
        
        # Find the item in the dataframe
        search_term = st.session_state.searched_item.lower()
        matching_items = df.filter(pl.col('name').str.to_lowercase().str.contains(search_term))
        
        if len(matching_items) > 0:
            # Show all matching items
            if len(matching_items) > 1:
                st.info(f"Found {len(matching_items)} items matching '{st.session_state.searched_item}':")
                
                # Let user select which one
                item_names = matching_items.select('name').to_series().to_list()
                selected_item_name = st.selectbox("Select the item:", item_names, key="item_selector")
                
                selected_item = matching_items.filter(pl.col('name') == selected_item_name)
            else:
                selected_item = matching_items
                # Safely extract the single match
                try:
                    row_list = selected_item.head(1).to_dicts()
                    selected_item_name = row_list[0].get('name') if row_list else selected_item.select('name').to_series().to_list()[0]
                except Exception:
                    selected_item_name = selected_item.select('name').to_series().to_list()[0]

            if len(selected_item) > 0:
                try:
                    item_row = selected_item.head(1).to_dicts()[0]
                except Exception:
                    st.error("❌ Unable to read selected item data")
                    item_row = None

                if item_row:
                    item_id = int(item_row.get('item_id', 0))
                
                # Display item info
                icol1, icol2, icol3 = st.columns([1, 1, 1])
                
                with icol1:
                    st.metric("Current Buy Price", f"{int(item_row['avgLowPrice']):,} GP")
                    st.metric("Current Sell Price", f"{int(item_row['avgHighPrice']):,} GP")
                
                with icol2:
                    st.metric("Profit per Flip", f"{int(item_row['net_edge']):,} GP")
                    st.metric("ROI", f"{item_row['edge_pct']:.1f}%")
                
                with icol3:
                    st.metric("Hourly Volume", f"{int(item_row['hourly_volume']):,}")
                    st.metric("GE Limit", f"{int(item_row['limit']):,}" if item_row['limit'] > 0 else "Unknown")
                
                # Fetch 30-day history
                st.markdown("### 📊 30-Day Price History")
                
                with st.spinner("Loading price history..."):
                    try:
                        timeseries_df = load_timeseries(item_id, timestep="24h")
                        
                        if timeseries_df is not None and len(timeseries_df) > 0:
                            import numpy as np
                            
                            prices = timeseries_df.select('avgHighPrice').to_series().to_list()
                            prices = [p for p in prices if p is not None]
                            
                            if len(prices) >= 7:
                                current_price = int(item_row['avgHighPrice'])
                                last_7d_avg = int(np.mean(prices[-7:]))
                                last_30d_avg = int(np.mean(prices))
                                last_30d_low = int(np.min(prices))
                                last_30d_high = int(np.max(prices))
                                
                                # Show metrics
                                hcol1, hcol2, hcol3, hcol4 = st.columns(4)
                                
                                with hcol1:
                                    st.metric("Last Week Avg", f"{last_7d_avg:,} GP",
                                             delta=f"{((current_price - last_7d_avg) / last_7d_avg * 100):+.1f}%")
                                
                                with hcol2:
                                    st.metric("Last Month Avg", f"{last_30d_avg:,} GP",
                                             delta=f"{((current_price - last_30d_avg) / last_30d_avg * 100):+.1f}%")
                                
                                with hcol3:
                                    st.metric("30-Day Low", f"{last_30d_low:,} GP")
                                
                                with hcol4:
                                    st.metric("30-Day High", f"{last_30d_high:,} GP")
                                
                                # Value assessment
                                vs_avg_pct = ((current_price - last_30d_avg) / last_30d_avg) * 100
                                
                                if vs_avg_pct < -10:
                                    st.success(f"💎 **BARGAIN!** Current price is {abs(vs_avg_pct):.1f}% BELOW 30-day average!")
                                elif vs_avg_pct > 10:
                                    st.warning(f"⚠️ **EXPENSIVE:** Current price is {vs_avg_pct:.1f}% ABOVE 30-day average.")
                                else:
                                    st.info(f"📊 **FAIR VALUE:** Price is within {abs(vs_avg_pct):.1f}% of 30-day average.")
                                
                                # Recent trades (5-minute)
                                st.markdown("### 📈 Recent Trading Activity (Last Hour)")
                                
                                trades_df = load_timeseries(item_id, timestep="5m")
                                
                                if trades_df is not None and len(trades_df) > 0:
                                    recent_trades = trades_df.tail(12).sort('datetime', descending=True)
                                    
                                    if 'datetime' in recent_trades.columns:
                                        display_df = recent_trades.select([
                                            (pl.col('datetime').dt.offset_by('-6h')).dt.strftime('%m/%d %I:%M %p').alias('Time (CST)'),
                                            pl.col('avgLowPrice').fill_null(0).cast(pl.Int64).alias('Buy Price'),
                                            pl.col('lowPriceVolume').fill_null(0).cast(pl.Int64).alias('Buy Volume'),
                                            pl.col('avgHighPrice').fill_null(0).cast(pl.Int64).alias('Sell Price'),
                                            pl.col('highPriceVolume').fill_null(0).cast(pl.Int64).alias('Sell Volume')
                                        ])
                                        
                                        display_pandas = display_df.to_pandas()
                                        
                                        st.dataframe(
                                            display_pandas,
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                        
                                        st.caption("📊 Last 12 trades (5-minute intervals). '0' means no trades in that period.")
                            else:
                                st.info("📊 Limited historical data available (less than 7 days)")
                        else:
                            st.warning("⚠️ No historical price data available for this item")
                    except Exception as e:
                        st.error(f"❌ Error loading price history: {str(e)}")
                
                # Clear search button
                if st.button("🗑️ Clear Search"):
                    del st.session_state.searched_item
                    st.rerun()
        else:
            st.warning(f"❌ No items found matching '{st.session_state.searched_item}'. Try a different search term.")
            
            if st.button("🗑️ Clear Search"):
                del st.session_state.searched_item
                st.rerun()
    
    # 🧬🧬🧬 ADAPTIVE INTELLIGENCE - TOP INSIGHTS 🧬🧬🧬
    st.markdown("---")
    st.markdown("## 🧬 ADAPTIVE INTELLIGENCE - Top Market Insights")
    
    st.caption("AI-powered insights that learn from current market conditions")
    
    # Generate adaptive insights
    adaptive_insights = adaptive.generate_adaptive_insights(df, portfolio_stats)
    
    if adaptive_insights:
        # Get top 3 most important insights (priority 1 and 2)
        critical_insights = [i for i in adaptive_insights if i.priority == 1]
        important_insights = [i for i in adaptive_insights if i.priority == 2]
        
        # Combine and take top 3
        top_insights = (critical_insights + important_insights)[:3]
        
        if top_insights:
            for insight in top_insights:
                priority_emoji = "🚨" if insight.priority == 1 else "⚠️"
                with st.expander(f"{priority_emoji} {insight.message} (Confidence: {insight.confidence*100:.0f}%)", expanded=insight.priority == 1):
                    st.markdown("**Recommended Actions:**")
                    for action in insight.action_items:
                        st.markdown(f"• {action}")
        else:
            st.info("✅ No critical insights - market looks normal!")
    
    # 🔥🔥🔥 MARKET INTELLIGENCE - KEY METRICS 🔥🔥🔥
    st.markdown("---")
    st.markdown("## 📊 Market Intelligence")
    
    st.caption("Real-time market overview - current opportunities and activity")
    
    # Executive Summary
    exec_summary = market_dashboard.generate_executive_summary(df, total_capital)
    st.info(exec_summary)
    
    # Key Metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    
    snapshot = market_dashboard.get_comprehensive_market_state(df)
    
    with mcol1:
        temp_color = "🔥" if "HOT" in snapshot.temperature else "🌡️" if "WARM" in snapshot.temperature else "❄️"
        st.metric("Market Temperature", snapshot.temperature)
        st.caption("🔥 HOT = 10+ elite | 🌡️ WARM = 5-9 | ❄️ COOL = <5")
    
    with mcol2:
        st.metric("Elite Opportunities", f"{snapshot.high_quality_count}", f"of {snapshot.total_opportunities} total")
        st.caption("Score 75+ = Elite")
    
    with mcol3:
        sentiment_emoji = "📈" if "BULLISH" in snapshot.market_sentiment else "📉" if "BEARISH" in snapshot.market_sentiment else "➡️"
        st.metric("Sentiment", f"{sentiment_emoji} {snapshot.market_sentiment.split()[0]}")
        st.caption(f"Avg: {snapshot.average_profit_pct:.1f}% profit")
    
    # Liquidity + Top Movers
    liq_col, mover_col = st.columns([1, 2])
    
    with liq_col:
        st.markdown("### 💧 Liquidity")
        vol_analysis = market_dashboard.calculate_volume_analysis(df)
        st.metric("Total Volume", f"{vol_analysis['total_hourly_volume']:,}/hr")
        st.metric("Rating", vol_analysis['liquidity_rating'])
        st.caption(f"High-volume: {vol_analysis['high_volume_items']} items")
    
    with mover_col:
        st.markdown("### 🚀 Top Movers")
        movers = market_dashboard.get_top_movers(df)
        for mover in movers[:3]:
            st.markdown(f"**{mover['category']}**: {mover['item']}")
            st.caption(f"{mover['value']} - {mover['detail']}")
    
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
    
    # 💎💎💎 BARGAIN HUNTER - LONG-TERM HOLDS 💎💎💎
    st.markdown("---")
    st.markdown("## 💎 BARGAIN HUNTER - Long-Term Value Plays")
    
    with st.expander("ℹ️ WHAT IS BARGAIN HUNTING?", expanded=False):
        st.markdown("""
        **Find items at TRULY CHEAP prices for week-long holds:**
        
        - 📉 **Historical Low**: Items at/near their 30-day lowest price
        - 📈 **Stable/Uptrending**: Not falling knives - price is stabilizing or rising
        - 💰 **Value Rating**: Compares current price vs 30-day average
        - ⏰ **Perfect for**: Buying today, selling next week when price recovers
        
        **How it works:**
        1. Analyzes 30 days of price history
        2. Finds items 10%+ below average
        3. Filters OUT items still dropping (falling knives)
        4. Shows ONLY items that are cheap AND stable/rising
        
        **These are TRUE bargains - not just cheap trash going to zero!**
        """)
    
    st.caption("Perfect for: Week-long holds, patient traders, buying the dip")
    
    # Analyze bargains from current data
    with st.spinner("Analyzing bargain opportunities..."):
        if len(df) > 0:
            # Find items with good fundamentals but lower than usual prices
            bargain_candidates = df.filter(
                (pl.col('edge_pct') >= 5.0) &  # At least 5% profit margin
                (pl.col('hourly_volume') >= 100) &  # Reasonable volume
                (pl.col('opportunity_score') >= 50) &  # Decent opportunity
                (pl.col('risk_score') <= 60)  # Not too risky
            )
            
            if len(bargain_candidates) > 0:
                # Sort by best value (high profit %, low risk)
                bargain_candidates = bargain_candidates.sort(
                    ['edge_pct', 'opportunity_score'], 
                    descending=[True, True]
                ).head(10)
                
                st.success(f"💎 Found {len(bargain_candidates)} potential bargain plays!")
                
                # Display bargains
                for idx, row in enumerate(bargain_candidates.iter_rows(named=True), 1):
                    with st.expander(f"#{idx} - {row['name']} - {row['edge_pct']:.1f}% ROI", expanded=(idx==1)):
                        bcol1, bcol2, bcol3 = st.columns(3)
                        
                        with bcol1:
                            st.metric("Buy Price", f"{int(row['avgLowPrice']):,} GP")
                            st.metric("Sell Price", f"{int(row['avgHighPrice']):,} GP")
                        
                        with bcol2:
                            st.metric("Profit Margin", f"{row['edge_pct']:.1f}%")
                            st.metric("Hourly Volume", f"{int(row['hourly_volume']):,}")
                        
                        with bcol3:
                            st.metric("Opportunity", f"{int(row.get('opportunity_score', 0))}/100")
                            st.metric("Risk Level", f"{int(row.get('risk_score', 50))}/100")
                        
                        # Trading recommendation
                        strategy = row.get('strategy_type', 'UNKNOWN')
                        if strategy == "SWING":
                            st.info("� **SWING TRADE**: Best for week-long holds. Buy now, sell in 3-7 days when profitable.")
                        elif strategy == "SHORT_HOLD":
                            st.info("📅 **SHORT HOLD**: Hold 1-2 days for price recovery.")
                        else:
                            st.info(f"🎯 **{strategy}**: Good value play with solid fundamentals.")
                        
                        # Calculate suggested quantity
                        buy_price = int(row['avgLowPrice'])
                        ge_limit = int(row['limit']) if row['limit'] > 0 else 10000
                        max_affordable = int(total_capital * 0.3 / buy_price) if buy_price > 0 else 0
                        suggested_qty = min(ge_limit, max_affordable, 5000)
                        total_investment = buy_price * suggested_qty
                        potential_profit = int(row['net_edge']) * suggested_qty
                        
                        st.markdown(f"""
                        **💰 Suggested Trade:**
                        - Buy {suggested_qty:,} units at {buy_price:,} GP each
                        - Total investment: {format_gp(total_investment)}
                        - Potential profit: {format_gp(potential_profit)}
                        - Hold time: 3-7 days for best results
                        """)
            else:
                st.warning("⚠️ No clear bargains detected at current market prices. Try adjusting your filters or check back later.")
                
            # Interactive: Click to view charts for bargain items
            if len(bargain_candidates) > 0:
                st.markdown("---")
                st.markdown("#### 📊 Click an item to view price charts")
                
                bargain_items = []
                for idx, row in enumerate(bargain_candidates.iter_rows(named=True), 1):
                    bargain_items.append({
                        'name': row['name'],
                        'item_id': int(row['item_id']),
                        'roi': row['edge_pct']
                    })
                
                selected_bargain = st.selectbox(
                    "Select a bargain item to analyze:",
                    options=[f"#{idx+1} - {item['name']} - {item['roi']:.1f}% ROI" for idx, item in enumerate(bargain_items)],
                    key="bargain_selector"
                )
                
                if selected_bargain:
                    # Extract item index from selection
                    selected_idx = int(selected_bargain.split(" - ")[0].replace("#", "")) - 1
                    selected_data = bargain_items[selected_idx]
                    selected_item_id = selected_data['item_id']
                    selected_name = selected_data['name']
                    
                    st.markdown(f"### 📈 Price History: {selected_name}")
                    
                    # Create tabs for different timeframes
                    tab1d, tab1w, tab1m, tab6m = st.tabs(["📅 1 Day", "📅 1 Week", "📅 1 Month", "📅 6 Months"])
                    
                    with tab1d:
                        with st.spinner("Loading 1-day chart..."):
                            df_5m = load_timeseries(selected_item_id, timestep="5m")
                            if df_5m is not None and len(df_5m) > 0:
                                cutoff = datetime.now() - timedelta(hours=24)
                                df_5m = df_5m.filter(pl.col("datetime") >= cutoff)
                                
                                if len(df_5m) > 0:
                                    fig = create_price_chart(df_5m, title=f"{selected_name} - 24 Hour Trend")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No data available for last 24 hours")
                            else:
                                st.info("No 5-minute data available")
                    
                    with tab1w:
                        with st.spinner("Loading 1-week chart..."):
                            df_1h = load_timeseries(selected_item_id, timestep="1h")
                            if df_1h is not None and len(df_1h) > 0:
                                cutoff = datetime.now() - timedelta(days=7)
                                df_1h = df_1h.filter(pl.col("datetime") >= cutoff)
                                
                                if len(df_1h) > 0:
                                    fig = create_price_chart(df_1h, title=f"{selected_name} - 7 Day Trend")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No data available for last 7 days")
                            else:
                                st.info("No hourly data available")
                    
                    with tab1m:
                        with st.spinner("Loading 1-month chart..."):
                            df_6h = load_timeseries(selected_item_id, timestep="6h")
                            if df_6h is not None and len(df_6h) > 0:
                                cutoff = datetime.now() - timedelta(days=30)
                                df_6h = df_6h.filter(pl.col("datetime") >= cutoff)
                                
                                if len(df_6h) > 0:
                                    fig = create_price_chart(df_6h, title=f"{selected_name} - 30 Day Trend")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No data available for last 30 days")
                            else:
                                st.info("No 6-hour data available")
                    
                    with tab6m:
                        with st.spinner("Loading 6-month chart..."):
                            df_24h = load_timeseries(selected_item_id, timestep="24h")
                            if df_24h is not None and len(df_24h) > 0:
                                cutoff = datetime.now() - timedelta(days=180)
                                df_24h = df_24h.filter(pl.col("datetime") >= cutoff)
                                
                                if len(df_24h) > 0:
                                    fig = create_price_chart(df_24h, title=f"{selected_name} - 6 Month Trend")
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No data available for last 6 months")
                            else:
                                st.info("No daily data available")
        else:
            st.info("No data available for bargain analysis.")
    
    # 🏝️💎 BLUE OCEAN OPPORTUNITIES - Low Competition Trades 💎🏝️
    st.markdown("---")
    st.markdown("## 🏝️ BLUE OCEAN OPPORTUNITIES - Undertraded Gems")
    
    with st.expander("ℹ️ WHAT ARE BLUE OCEAN TRADES?", expanded=False):
        st.markdown("""
        **Find profitable trades with LESS COMPETITION:**
        
        - 🏝️ **Blue Ocean**: Markets with fewer traders competing
        - 💰 **Easier Profits**: Less competition = easier to get good fills
        - 🤖 **Less Bot Activity**: Avoid heavily bot-traded items
        - ⚡ **Better Execution**: Your orders fill faster with less slippage
        
        **Why this matters:**
        - Popular items (Dragon bones, Rune bars) = 100+ merchers fighting for fills
        - Niche items = Maybe 5-10 traders = YOU get better prices!
        
        **Blue Ocean Strategy:**
        1. Find items with good profit (60+ opportunity score)
        2. BUT low competition (not many traders active)
        3. Profit easier with less fighting for fills!
        """)
    
    st.caption("Perfect for: Traders who want easier fills, less stress, avoid bot-heavy items")
    
    # Find blue ocean opportunities
    with st.spinner("Searching for undertraded opportunities..."):
        from core.competitive_analysis import find_undertraded_opportunities
        
        if len(allocated_df) > 0:
            blue_ocean_trades = find_undertraded_opportunities(allocated_df, min_opportunity=60)
            
            if len(blue_ocean_trades) > 0:
                st.success(f"🏝️ Found {len(blue_ocean_trades)} blue ocean opportunities!")
                
                # Display as cards
                for idx, trade in enumerate(blue_ocean_trades, 1):
                    with st.expander(f"#{idx} - {trade['name']} - {trade['ease_of_profit']}", expanded=(idx==1)):
                        bcol1, bcol2, bcol3 = st.columns(3)
                        
                        with bcol1:
                            st.metric("Opportunity Score", f"{trade['opportunity_score']}/100")
                            st.metric("Competition", trade['competition_level'])
                        
                        with bcol2:
                            st.metric("Profit/Item", f"{trade['profit_per_item']:,} GP")
                            st.metric("ROI", f"{trade['roi_pct']:.1f}%")
                        
                        with bcol3:
                            st.metric("Ease of Profit", trade['ease_of_profit'])
                            st.metric("Hourly Volume", f"{trade['hourly_volume']:,}")
                        
                        # Why it's a blue ocean
                        if trade['competition_score'] <= 20:
                            st.success("🏝️ **BLUE OCEAN!** Very few traders active on this item. You'll get excellent fills!")
                        elif trade['competition_score'] <= 40:
                            st.info("🟢 **LOW COMPETITION**: Good opportunity with reasonable competition. Solid pick!")
                        
                        st.caption(f"💡 **Strategy**: With low competition, you can be more aggressive with offers and still get good fills. Less stress!")
            else:
                st.warning("⚠️ No blue ocean opportunities found. Current market is competitive. Try lowering minimum opportunity score filter.")
        else:
            st.info("No data available for blue ocean analysis.")
    
    #  SMART EXECUTION PLANNER
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
        
        # Collect day-of-week data for items with meaningful patterns
        show_day_columns = False  # Initialize flag
        day_data = {}
        
        for row in allocated_df.iter_rows(named=True):
            try:
                item_id = int(row['item_id'])
                ts_df = load_timeseries(item_id, timestep="1h")
                if ts_df is not None and len(ts_df) >= 14:
                    analysis = analyze_day_of_week_patterns(ts_df)
                    if analysis['has_data']:
                        timing_pct = (analysis['timing_advantage_gp'] / analysis['best_buy_price'] * 100) if analysis['best_buy_price'] > 0 else 0
                        if timing_pct >= 5 or analysis['timing_advantage_gp'] >= 10000:
                            day_data[row['name']] = {
                                'buy_day': analysis['best_buy_day'][:3],
                                'sell_day': analysis['best_sell_day'][:3],
                                'advantage_gp': analysis['timing_advantage_gp'],
                                'advantage_pct': timing_pct
                            }
                            show_day_columns = True  # Found at least one meaningful pattern
            except:
                pass
        
        # Build conditional day column explanation
        day_info = ""
        if show_day_columns:
            day_info = """
        📅 **Day Patterns:** Shows historical price patterns (30-day avg) when meaningful (5%+ swing or 10K+ GP).
        - Example: "Mon" under Cheapest Day = Mondays historically have lowest avg prices
        - **Note:** Current week prices may differ - use as reference only, not a buy/sell signal
        
        """
        
        st.info(f"""
        💡 **How to read this:** Buy at the 'Buy Price', sell at 'GE Sell Price' (GE automatically deducts 2% tax). Your actual profit is shown in 'Profit/Item'.
        
        {strategy_tip}
        {day_info}📊 **Confidence Score:** Measures volume quality and balance:
        - 🟢 **70-100**: High confidence - balanced buy/sell volume, good liquidity
        - 🟡 **50-69**: Medium confidence - decent volume but may be one-sided
        - 🔴 **30-49**: Low confidence - low volume or unbalanced market (risky!)
        
        ⏱️ **Fill Speed:** Time to accumulate your position (⚡ INSTANT = <30min, 🟢 QUICK = <2hrs, 🟡 MODERATE = 2-8hrs, 🟠 SLOW = 8-24hrs)
        
        📊 **Pressure:** Market buy/sell dynamics (🟢 = BUY PRESSURE = price rising, 🔴 = SELL PRESSURE = price falling, ⚪ = balanced)
        """)
        
        # Prepare display dataframe with actionable columns + advanced metrics
        display_df = allocated_df.select([
            "name",
            "item_id",
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
            "fill_difficulty",  # Microstructure: how fast it fills
            "pressure_direction",  # Microstructure: buy/sell pressure
        ])
        
        # Convert to pandas for display with formatting
        display_pd = display_df.to_pandas()
        
        # Add day columns if we found meaningful patterns
        if show_day_columns:
            display_pd['best_buy_day'] = display_pd['name'].map(lambda x: day_data.get(x, {}).get('buy_day', '-'))
            display_pd['best_sell_day'] = display_pd['name'].map(lambda x: day_data.get(x, {}).get('sell_day', '-'))
        
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
        if show_day_columns:
            display_pd.columns = [
                "Item Name",
                "item_id_hidden",
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
                "⏱️ Fill Speed",
                "� Pressure",
                "�📅 Cheapest Day",
                "📅 Peak Day",
                "Total Profit",
            ]
        else:
            display_pd.columns = [
                "Item Name",
                "item_id_hidden",
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
                "⏱️ Fill Speed",
                "📊 Pressure",
                "Total Profit",
            ]
        
        # Drop the hidden ID column
        display_pd = display_pd.drop(columns=['item_id_hidden'])
        
        st.dataframe(
            display_pd,
            use_container_width=True,
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
                
                # Show trade summary directly (no buttons to prevent refresh)
                st.markdown(f"""
**📋 TRADE DETAILS:**
```
TRADE: {row['name']}
BUY: {qty:,} @ {buy_price:,} GP = {format_gp(total_cost)}
SELL: {qty:,} @ {sell_price_before_tax:,} GP
PROFIT: {format_gp(profit_total)} ({row['edge_pct']:.1f}% ROI)
```
**� GE Search:** {row['name']}  
**⚡ Quick Ref:** Buy {buy_price:,} GP × {qty:,} | Sell {sell_price_before_tax:,} GP × {qty:,}
                """)
                
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
    
    st.caption(f"Showing items with minimum {min_roi_pct}% ROI per flip, sorted by rank score")
    
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
        use_container_width=True,
        hide_index=True,
        height=600,
    )
    
    # Footer
    st.markdown("---")
    st.caption("Data from [RuneScape Wiki Real-Time Prices API](https://prices.runescape.wiki/)")
    st.caption("⚠️ This is for educational purposes. Real trading involves risk and GE limits.")


if __name__ == "__main__":
    main()
