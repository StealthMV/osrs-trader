# OSRS Trader - Improvements Summary

## ✅ CHANGES MADE:

### 1. **FIXED GE TAX FOR CHEAP ITEMS** ⭐⭐⭐
**Location:** `core/features.py`

**What Changed:**
- Items UNDER 100 GP now have **NO GE TAX**
- Items 100 GP+ have 2% tax as before

**Impact:**
- Potions, darts, runes, arrows now show correct profits!
- Example: Adamant dart tips at 50 GP each
  - Before: 2% tax applied = wrong profit
  - After: 0% tax = correct profit

**Code:**
```python
# GE tax only applies to items >= 100 GP
pl.when(pl.col("avgHighPrice") >= 100)
  .then(pl.col("avgHighPrice") * GE_TAX_RATE)
  .otherwise(0)
  .alias("ge_tax")
```

---

### 2. **PRICE HISTORY ANALYZER** ⭐⭐⭐
**Location:** `core/price_history_analyzer.py` (NEW FILE)

**What It Does:**
Analyzes 30-day price history to detect:

#### **BARGAIN 🔥**
- Item is 10%+ below 30-day average
- At or near historical low (within 5%)
- Trend is UPTREND or STABLE
- **Recommendation:** BUY MASS QUANTITY for long-term holds

#### **FALLING KNIFE ⚠️**
- Item is below average BUT still trending DOWN
- Price keeps dropping (recent 7d < previous 7d by 5%+)
- **Recommendation:** AVOID! Wait for trend to stabilize

#### **OVERPRICED ❌**
- Item is 10%+ above 30-day average
- **Recommendation:** Wait for better entry

#### **FAIR ➡️**
- Near 30-day average, normal opportunity

**How It Works:**
1. Fetches 30 days of 5-minute price data
2. Calculates avg, high, low
3. Detects trend (comparing recent vs older prices)
4. Rates value: BARGAIN, FAIR, OVERPRICED, FALLING_KNIFE

**Perfect for your use case:**
- Long holds (weeks): Shows if truly cheap
- Mass quantity: Identifies items at historical lows
- Avoids false positives: Filters out items that keep dropping

---

### 3. **GE SLOT MANAGER & PRICE ALERTS** ⭐⭐
**Location:** `core/portfolio_tracker.py` (NEW FILE), sidebar in `app.py`

**Features:**
- Track all 8 GE slots
- See real-time profit/loss
- Get alerts when sell price hits
- Shows which positions to sell

**Already implemented and working!**

---

## 🗑️ SECTIONS TO REMOVE/HIDE:

### **Option A: Comment Out (Keep code, hide from UI)**
Best if you want to keep the code but not show it to users.

### **Option B: Delete Entirely**
If you're sure you don't need it.

### **Sections to Consider:**

#### 1. **Fractal Pattern Detection**
**Location:** Lines ~1530-1545 in app.py
**What it does:** Looks for repeating patterns across timeframes
**Your feedback:** "I don't know how much the fractal patterns are doing"
**Recommendation:** **HIDE** - too theoretical

#### 2. **Swarm Intelligence / Ant Colony**
**Location:** Lines ~1545-1580 in app.py
**What it does:** 100 virtual ants find optimal trade sequence
**Your feedback:** "I don't know how much the ant theory is doing"
**Recommendation:** **HIDE** - not actionable, takes time to compute

#### 3. **Quantum Superposition**
**Location:** Lines ~1447-1475 in app.py
**What it does:** Theoretical maximum profit
**Usefulness:** Shows if you're capturing market potential
**Recommendation:** **KEEP BUT SIMPLIFY** - can be useful but maybe too complex

#### 4. **Portfolio Synergy**
**Location:** Lines ~1575-1590 in app.py
**What it does:** How well items work together
**Your feedback:** Implied not useful (diversification concerns)
**Recommendation:** **HIDE** - you don't care about "synergy"

---

## 🎯 WHAT TO DO NEXT:

### **IMMEDIATE ACTION:**
I can add a simple toggle in the sidebar:
```python
show_advanced_features = st.sidebar.checkbox("Show Advanced Analytics", value=False)
```

Then wrap all the fractal/swarm/quantum stuff in:
```python
if show_advanced_features:
    # ... complex stuff ...
```

This way:
- **Default:** Clean, simple, actionable interface
- **Optional:** Turn on if you want to see the nerdy stuff

### **PRIORITY ADDITIONS:**

1. **✅ DONE:** Fixed GE tax for cheap items
2. **✅ DONE:** Price history analyzer (bargains vs falling knives)
3. **✅ DONE:** GE slot manager
4. **TODO:** Add "Price Value Analysis" section to main page showing:
   - Items at historical lows (BARGAINS)
   - Items with stable uptrends (SAFE LONG HOLDS)
   - Items to avoid (FALLING KNIVES)
5. **TODO:** Filter for mass-quantity opportunities (cheap items, no GE tax, high volume)

---

## 💡 YOUR FEEDBACK NEEDED:

1. **Do you want me to:**
   - [ ] Hide fractal/swarm/quantum behind a toggle?
   - [ ] Delete them entirely?
   - [ ] Keep as-is?

2. **Should I add a dedicated "BARGAIN HUNTER" section** showing:
   - Items at 30-day lows
   - Stable/uptrending only
   - Perfect for week-long holds?

3. **Should I add a "MASS QUANTITY" strategy** for:
   - Items <100 GP (no GE tax)
   - High volume
   - Good for flipping thousands of units?

Let me know and I'll implement!
