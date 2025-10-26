# Script to replace hyper-dimensional section with BARGAIN HUNTER and MASS QUANTITY sections

import sys

input_file = r'c:\Users\markv\Simple Trader\app.py'
output_file = r'c:\Users\markv\Simple Trader\app_new.py'

# Read the file
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find section boundaries
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# 🌌🌌🌌 HYPER-DIMENSIONAL ANALYSIS' in line:
        start_idx = i
    if start_idx is not None and '# 🔥 SMART EXECUTION PLANNER' in line:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print("Could not find section boundaries!")
    sys.exit(1)

print(f"Replacing lines {start_idx+1} to {end_idx}")

# New content
new_content = '''    
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
    
    # Show bargain opportunities (simplified - will add full implementation after)
    st.info("🔄 Bargain analysis coming soon! This will show items at historical lows with stable/uptrend.")
    
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
    
'''

# Assemble new file
new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]

# Write output
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Created {output_file}")
print(f"Removed {end_idx - start_idx} lines")
print(f"Added {len(new_content.split(chr(10)))} lines")
