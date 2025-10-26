"""
Parquet-based caching layer for historical data
Reduces API calls and enables offline analysis
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

from .config import DATA_DIR, MAPPING_FILE, HOURLY_FILE


class ParquetCache:
    """Simple cache manager for OSRS price data"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_file = self.data_dir / "metadata.json"
    
    def _read_metadata(self) -> Dict[str, Any]:
        """Read cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _write_metadata(self, metadata: Dict[str, Any]):
        """Write cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_cached(self, cache_key: str, max_age_seconds: int = 3600) -> bool:
        """
        Check if cached data exists and is fresh
        
        Args:
            cache_key: Key identifying the cached data (e.g., 'mapping', '1h')
            max_age_seconds: Maximum age in seconds (default 1 hour)
        """
        metadata = self._read_metadata()
        
        if cache_key not in metadata:
            return False
        
        cached_time = datetime.fromisoformat(metadata[cache_key]["timestamp"])
        age = datetime.now() - cached_time
        
        return age.total_seconds() < max_age_seconds
    
    def save_mapping(self, df: pl.DataFrame):
        """Save mapping data to Parquet"""
        df.write_parquet(MAPPING_FILE)
        
        metadata = self._read_metadata()
        metadata["mapping"] = {
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
        }
        self._write_metadata(metadata)
    
    def load_mapping(self) -> Optional[pl.DataFrame]:
        """Load mapping data from Parquet"""
        path = Path(MAPPING_FILE)
        if path.exists():
            return pl.read_parquet(path)
        return None
    
    def save_hourly(self, df: pl.DataFrame):
        """Save 1h data to Parquet"""
        df.write_parquet(HOURLY_FILE)
        
        metadata = self._read_metadata()
        metadata["1h"] = {
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
        }
        self._write_metadata(metadata)
    
    def load_hourly(self) -> Optional[pl.DataFrame]:
        """Load 1h data from Parquet"""
        path = Path(HOURLY_FILE)
        if path.exists():
            return pl.read_parquet(path)
        return None
    
    def save_timeseries(self, item_id: int, df: pl.DataFrame, timestep: str = "5m"):
        """
        Save timeseries data for a specific item
        
        Args:
            item_id: Item ID
            df: Timeseries DataFrame
            timestep: Timestep ('5m', '1h', '6h')
        """
        filename = self.data_dir / f"timeseries_{item_id}_{timestep}.parquet"
        df.write_parquet(filename)
        
        metadata = self._read_metadata()
        key = f"timeseries_{item_id}_{timestep}"
        metadata[key] = {
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
        }
        self._write_metadata(metadata)
    
    def load_timeseries(self, item_id: int, timestep: str = "5m") -> Optional[pl.DataFrame]:
        """Load timeseries data for a specific item"""
        filename = self.data_dir / f"timeseries_{item_id}_{timestep}.parquet"
        if filename.exists():
            return pl.read_parquet(filename)
        return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        metadata = self._read_metadata()
        return {
            "cache_dir": str(self.data_dir),
            "cached_items": metadata,
        }
    
    def clear_cache(self):
        """Delete all cached Parquet files"""
        for file in self.data_dir.glob("*.parquet"):
            file.unlink()
        
        if self.metadata_file.exists():
            self.metadata_file.unlink()
