"""
Configuration constants for OSRS Trading Analytics
Follows official RuneScape Wiki API specifications
"""

# API Base URL
BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"

# User-Agent Header (REQUIRED by API)
# Replace with your Discord handle: @yourDiscord
USER_AGENT = "osrs-trader-bot - @markv - simple-trader-analytics"

# API Endpoints
ENDPOINTS = {
    "mapping": f"{BASE_URL}/mapping",
    "latest": f"{BASE_URL}/latest",
    "5m": f"{BASE_URL}/5m",
    "1h": f"{BASE_URL}/1h",
    "timeseries": f"{BASE_URL}/timeseries",  # ?timestep=5m&id=<item_id>
}

# Grand Exchange Tax Rate (2% on sale price)
GE_TAX_RATE = 0.02

# Request Settings
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # exponential backoff multiplier
CACHE_TTL = 3600  # 1 hour cache for mapping/hourly data

# Analysis Defaults
DEFAULT_MOMENTUM_WINDOW = 6  # hours
DEFAULT_VOLATILITY_WINDOW = 24  # hours
DEFAULT_CAPITAL = 50_000_000  # 50M GP (configurable in UI)
DEFAULT_MAX_PCT_PER_ITEM = 20  # max 20% of capital per item
DEFAULT_MAX_ITEMS = 10  # max concurrent positions
DEFAULT_MIN_PROFIT_PER_FLIP = 10_000  # minimum GP profit per item flip (10K default)
DEFAULT_MIN_HOURLY_PROFIT = 4_000_000  # 4M GP/hour target

# Ranking Weights
RANK_WEIGHTS = {
    "edge_pct": 0.45,
    "log_liquidity": 0.25,
    "momentum": 0.20,
    "volatility": -0.10,  # negative because lower is better
}

# Minimum Filters (avoid illiquid/low-value items)
MIN_PRICE = 1_000  # minimum item price in GP (1K+ to catch all opportunities)
MIN_HOURLY_VOLUME = 20  # minimum traded volume per hour (more inclusive)
MIN_SPREAD_PCT = 0.2  # minimum spread % to consider
MIN_PROFIT_PER_FLIP = 1_000  # minimum GP profit per single item (1K minimum)
MAX_SPREAD_PCT = 50  # maximum spread % (wider net for opportunities)
MIN_EDGE_PCT = 0.3  # minimum edge percentage (profit margin after tax)

# Advanced Filters
MIN_MARKET_CAP = 50_000  # minimum item market cap (price × volume)
MIN_VELOCITY_SCORE = 10  # minimum turnover speed score
MAX_RISK_SCORE = 80  # maximum risk tolerance (0-100)

# Strategy Thresholds
SCALP_MIN_VOLUME = 200  # High-frequency scalping threshold
SWING_MIN_EDGE = 3.0  # Swing trading minimum edge %
WHALE_MIN_PROFIT = 100_000  # Big ticket item minimum profit
ARBITRAGE_MAX_SPREAD = 5.0  # Arbitrage spread ceiling

# Profitability Targets
MIN_PROFIT_PER_HOUR = 4_000_000  # 4M GP/hour minimum (better than farming)
ESTIMATED_FLIPS_PER_HOUR = 1  # Conservative: assume 1 full flip per hour

# Display Settings
TABLE_COLUMNS = [
    "name",
    "mid_price",
    "spread_pct",
    "edge_pct",
    "momentum_pct",
    "hourly_volume",
    "rank_score",
    "allocation_gp",
    "allocation_qty",
]

# Data Storage
DATA_DIR = "data"
MAPPING_FILE = f"{DATA_DIR}/mapping.parquet"
HOURLY_FILE = f"{DATA_DIR}/1h.parquet"
