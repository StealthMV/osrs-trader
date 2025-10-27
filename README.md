# 💰 OSRS Trading Analytics

A lightweight **Old School RuneScape** trading analytics app that uses the official [RuneScape Wiki Real-Time Prices API](https://prices.runescape.wiki/api/v1/osrs) to identify high-value Grand Exchange arbitrage opportunities.

## 🚀 Features

- **Real-time price data** from official Wiki API endpoints
- **Smart capital allocation** with GE limits and liquidity constraints
- **Composite ranking** based on edge, liquidity, momentum, and volatility
- **GE tax-adjusted profit** calculations (accounts for 1% tax)
- **Streamlit dashboard** with interactive sliders
- **Parquet caching** for historical analysis (1-hour cache)

## 📊 How It Works

1. Fetches `/mapping` (item metadata) and `/1h` (hourly prices) in bulk
2. Computes features:
   - `mid_price` = average of avgHigh and avgLow
   - `spread_pct` = % difference between high and low
   - `net_edge` = (avgHigh - GE_tax) - avgLow
   - `edge_pct` = net_edge / avgLow × 100
3. Ranks items by weighted score:
   - 45% edge percentage
   - 25% log(liquidity)
   - 20% momentum (placeholder)
   - -10% volatility
4. Allocates capital greedily respecting:
   - Max % per item
   - GE buy limits
   - Hourly liquidity
   - Remaining capital

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Setup

```powershell
# Clone or navigate to project directory
cd "C:\Users\markv\Simple Trader"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

**IMPORTANT**: Before running, update your User-Agent in `core/config.py`:

```python
USER_AGENT = "osrs-trader-bot - @YourDiscordHandle - github.com/yourname/osrs-trader"
```

This is **required** by the RuneScape Wiki API. See their [usage guidelines](https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices).

### Other Settings (in `core/config.py`)

- `GE_TAX_RATE`: Grand Exchange tax (default 1%)
- `MIN_PRICE`: Minimum item price filter (default 1000 GP)
- `MIN_HOURLY_VOLUME`: Minimum traded volume (default 10)
- `RANK_WEIGHTS`: Adjust scoring weights
- `CACHE_TTL`: API cache duration (default 3600s = 1 hour)

## 🎮 Usage

### Run the Dashboard

```powershell
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Dashboard Controls

- **Total Capital**: Your available GP (default 10M)
- **Max % per Item**: Maximum allocation per item (default 20%)
- **Max Concurrent Items**: Portfolio size limit (default 10)
- **Momentum Window**: Historical lookback (not yet implemented)
- **Refresh Data**: Force reload from API

## 📁 Project Structure

```
Simple Trader/
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Python dependencies
├── core/
│   ├── __init__.py       # Package exports
│   ├── api.py            # API client with retry logic
│   ├── features.py       # Feature engineering
│   ├── allocate.py       # Capital allocation engine
│   ├── config.py         # Configuration constants
│   └── storage.py        # Parquet caching layer
├── data/                 # Cached Parquet files (auto-created)
│   ├── mapping.parquet
│   ├── 1h.parquet
│   └── metadata.json
└── README.md
```

## 🔌 API Endpoints Used

| Endpoint | Purpose | Refresh Rate |
|----------|---------|--------------|
| `/mapping` | Item names, limits, alch values, icons | Daily |
| `/latest` | Latest high/low snapshot | Real-time |
| `/5m` | 5-minute averages + volumes | 5 minutes |
| `/1h` | 1-hour averages + volumes | 1 hour |
| `/timeseries` | Historical data (max 365 points) | On-demand |

## ⚠️ Disclaimer

This tool is for **educational and analytical purposes only**.

- **No guarantees**: Market conditions change rapidly
- **GE limits apply**: You can't instantly buy unlimited quantities
- **Risk exists**: Prices can move against you
- **Use responsibly**: Follow RuneScape's rules and ToS

## 🤝 Contributing

This is a community tool. Improvements welcome:

- Add `/timeseries` integration for real momentum
- Implement volatility using historical std dev
- Add item icons and Wiki links
- Build backtesting framework
- Add Discord/Telegram notifications

## 📚 Resources

- [RuneScape Wiki API Docs](https://prices.runescape.wiki/)
- [RuneLite Discord #api-discussion](https://discord.gg/runelite)
- [OSRS Grand Exchange](https://oldschool.runescape.wiki/w/Grand_Exchange)

## 📄 License

MIT License - Use freely, no warranty provided.

---

**Happy flipping! 💰**
