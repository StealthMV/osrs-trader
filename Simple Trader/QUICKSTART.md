# 🚀 Quick Start Guide

## Step 1: Update User-Agent (REQUIRED)

Edit `core/config.py` and replace the placeholder:

```python
USER_AGENT = "osrs-trader-bot - @YourDiscordHandle - github.com/yourname/osrs-trader"
```

**Example:**
```python
USER_AGENT = "osrs-trader-bot - @markv#1234 - personal-project"
```

This is required by the RuneScape Wiki API to track usage responsibly.

## Step 2: Install Dependencies

Open PowerShell in this directory and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

## Step 3: Test API Connection

Run the examples to verify everything works:

```powershell
python examples.py
```

You should see:
- ✅ Item mapping loaded
- ✅ Rate limiting working
- ✅ Top 10 trading opportunities

## Step 4: Launch Dashboard

```powershell
streamlit run app.py
```

Your browser will open to `http://localhost:8501`

## Step 5: Configure Your Strategy

In the sidebar, adjust:
- **Total Capital**: How much GP you want to allocate
- **Max % per Item**: Risk limit per position (default 20%)
- **Max Concurrent Items**: Portfolio size (default 10)

## 🎯 Understanding the Results

### Portfolio Overview
- **Allocated**: How much GP will be used
- **Positions**: Number of items selected
- **Avg Edge**: Expected profit margin after GE tax
- **Potential Profit**: Total profit if all items flip

### Recommended Trades Table
- **Edge %**: Profit margin after 1% GE tax
- **Spread %**: Price gap between buy and sell
- **Hourly Vol**: Trading volume (liquidity indicator)
- **Rank Score**: Composite score (higher = better)

## ⚡ Tips

1. **Start small**: Test with 1-5M GP first
2. **Check GE limits**: Don't try to buy more than your 4-hour limit
3. **Monitor volume**: Low volume = harder to buy/sell
4. **Refresh regularly**: Click the refresh button every hour
5. **Verify prices in-game**: Always double-check before trading

## 🔧 Troubleshooting

### "Import could not be resolved"
- Your IDE is warning about missing packages
- Run `pip install -r requirements.txt` in your activated venv
- Restart VS Code to pick up the virtual environment

### "Failed to load data"
- Check internet connection
- Verify User-Agent is set in `core/config.py`
- Check API status at https://prices.runescape.wiki/

### "No tradeable items found"
- Filters in `core/config.py` might be too strict
- Try lowering `MIN_PRICE` or `MIN_HOURLY_VOLUME`

## 📚 Next Steps

- Implement `/timeseries` for real momentum calculation
- Add item icons from the API
- Build a backtesting module
- Add Discord webhook notifications
- Create a CLI version for automated monitoring

Happy trading! 💰
