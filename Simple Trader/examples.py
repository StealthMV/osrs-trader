"""
Example: API-safe data fetching with proper User-Agent and rate limiting
Demonstrates best practices for RuneScape Wiki API usage
"""

from core.api import OSRSPricesAPI
from core.features import build_trading_dataframe
from core.storage import ParquetCache
import time


def example_basic_fetch():
    """Basic example: Fetch and display mapping data"""
    print("=" * 60)
    print("Example 1: Basic Fetch with Proper User-Agent")
    print("=" * 60)
    
    with OSRSPricesAPI() as api:
        # Fetch mapping data
        print("\n📦 Fetching item mapping...")
        mapping = api.get_mapping()
        
        # Display sample items
        print(f"✅ Loaded {len(mapping)} items")
        print("\nSample items:")
        for item in mapping[:5]:
            print(f"  - {item['name']} (ID: {item['id']}, Limit: {item.get('limit', 'N/A')})")


def example_bulk_fetch_with_rate_limiting():
    """Example: Fetch multiple endpoints with automatic rate limiting"""
    print("\n" + "=" * 60)
    print("Example 2: Bulk Fetch with Rate Limiting")
    print("=" * 60)
    
    with OSRSPricesAPI() as api:
        start_time = time.time()
        
        # The API client automatically enforces 1-second minimum between requests
        print("\n⏱️  Fetching mapping (request 1)...")
        mapping = api.get_mapping()
        print(f"✅ Mapping: {len(mapping)} items")
        
        print("⏱️  Fetching latest prices (request 2, auto rate-limited)...")
        latest = api.get_latest()
        print(f"✅ Latest: {len(latest.get('data', {}))} items with prices")
        
        print("⏱️  Fetching 1h averages (request 3, auto rate-limited)...")
        hourly = api.get_1h()
        print(f"✅ Hourly: {len(hourly.get('data', {}))} items with averages")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed:.2f}s (includes automatic rate limiting)")


def example_with_caching():
    """Example: Use caching to avoid redundant API calls"""
    print("\n" + "=" * 60)
    print("Example 3: Parquet Caching")
    print("=" * 60)
    
    cache = ParquetCache()
    
    # Check if we have fresh cached data
    if cache.is_cached("mapping", max_age_seconds=3600):
        print("\n💾 Found fresh cached mapping data (< 1 hour old)")
        mapping_df = cache.load_mapping()
        print(f"✅ Loaded {len(mapping_df)} items from cache")
    else:
        print("\n🌐 No cache or stale data, fetching from API...")
        with OSRSPricesAPI() as api:
            mapping = api.get_mapping()
        
        from core.features import parse_mapping_data
        mapping_df = parse_mapping_data(mapping)
        cache.save_mapping(mapping_df)
        print(f"✅ Fetched and cached {len(mapping_df)} items")
    
    # Show cache info
    print("\n📊 Cache status:")
    info = cache.get_cache_info()
    for key, value in info["cached_items"].items():
        print(f"  - {key}: {value['rows']} rows, updated {value['timestamp']}")


def example_timeseries_fetch():
    """Example: Fetch historical timeseries for a specific item"""
    print("\n" + "=" * 60)
    print("Example 4: Timeseries for Specific Item")
    print("=" * 60)
    
    # Example: Fetch timeseries for item 2 (Cannonball)
    item_id = 2
    item_name = "Cannonball"
    
    with OSRSPricesAPI() as api:
        print(f"\n📈 Fetching 5m timeseries for {item_name} (ID: {item_id})...")
        timeseries = api.get_timeseries(item_id, timestep="5m")
        
        data_points = timeseries.get("data", [])
        print(f"✅ Received {len(data_points)} data points")
        
        if data_points:
            latest = data_points[-1]
            print(f"\nLatest 5m data:")
            print(f"  - Avg High: {latest.get('avgHighPrice', 'N/A')} GP")
            print(f"  - Avg Low: {latest.get('avgLowPrice', 'N/A')} GP")
            print(f"  - High Volume: {latest.get('highPriceVolume', 'N/A')}")
            print(f"  - Low Volume: {latest.get('lowPriceVolume', 'N/A')}")


def example_full_analysis():
    """Example: Complete analysis pipeline"""
    print("\n" + "=" * 60)
    print("Example 5: Full Trading Analysis Pipeline")
    print("=" * 60)
    
    print("\n🔄 Running full analysis...")
    
    with OSRSPricesAPI() as api:
        mapping = api.get_mapping()
        hourly = api.get_1h()
    
    print("📊 Building trading dataframe...")
    df = build_trading_dataframe(mapping, hourly)
    
    print(f"✅ Found {len(df)} tradeable items after filtering")
    
    # Show top 10 by rank score
    print("\n🏆 Top 10 Trading Opportunities:")
    top_10 = df.head(10)
    
    for idx, row in enumerate(top_10.iter_rows(named=True), 1):
        print(f"\n{idx}. {row['name']}")
        print(f"   Mid Price: {row['mid_price']:,} GP")
        print(f"   Edge: {row['edge_pct']:.2f}%")
        print(f"   Spread: {row['spread_pct']:.2f}%")
        print(f"   Hourly Volume: {row['hourly_volume']:,}")
        print(f"   Rank Score: {row['rank_score']:.2f}")


if __name__ == "__main__":
    print("\n🚀 OSRS Trading API Examples")
    print("Demonstrates API-safe usage with proper headers and rate limiting\n")
    
    try:
        # Run all examples
        example_basic_fetch()
        example_bulk_fetch_with_rate_limiting()
        example_with_caching()
        # example_timeseries_fetch()  # Uncomment to test specific item
        example_full_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tips:")
        print("  - Check your User-Agent in core/config.py")
        print("  - Ensure you have internet connection")
        print("  - Visit https://prices.runescape.wiki/ to verify API status")
