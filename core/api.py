"""
API client for RuneScape Wiki Real-Time Prices API
Implements proper rate limiting, retries, and User-Agent headers
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional
import time

from .config import (
    USER_AGENT,
    ENDPOINTS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_BACKOFF,
)


class OSRSPricesAPI:
    """Client for fetching OSRS price data from official Wiki API"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
        self.client = httpx.Client(
            headers=self.headers,
            timeout=REQUEST_TIMEOUT,
        )
        self.last_request_time = 0
        self.min_request_interval = 1.0  # minimum 1 second between requests
    
    def _rate_limit(self):
        """Ensure we don't spam the API"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_BACKOFF, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def _fetch(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fetch data with retry logic"""
        self._rate_limit()
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_mapping(self) -> Dict[str, Any]:
        """
        Fetch item mapping data
        Returns: {
            "id": {
                "examine": str,
                "id": int,
                "members": bool,
                "lowalch": int,
                "limit": int,
                "value": int,
                "highalch": int,
                "icon": str,
                "name": str
            }
        }
        """
        return self._fetch(ENDPOINTS["mapping"])
    
    def get_latest(self) -> Dict[str, Any]:
        """
        Fetch latest price snapshot
        Returns: {
            "data": {
                "item_id": {
                    "high": int,
                    "highTime": int,
                    "low": int,
                    "lowTime": int
                }
            }
        }
        """
        return self._fetch(ENDPOINTS["latest"])
    
    def get_5m(self) -> Dict[str, Any]:
        """
        Fetch 5-minute average prices and volumes
        Returns: {
            "data": {
                "item_id": {
                    "avgHighPrice": int,
                    "highPriceVolume": int,
                    "avgLowPrice": int,
                    "lowPriceVolume": int
                }
            }
        }
        """
        return self._fetch(ENDPOINTS["5m"])
    
    def get_1h(self) -> Dict[str, Any]:
        """
        Fetch 1-hour average prices and volumes
        Returns: {
            "data": {
                "item_id": {
                    "avgHighPrice": int,
                    "highPriceVolume": int,
                    "avgLowPrice": int,
                    "lowPriceVolume": int
                }
            }
        }
        """
        return self._fetch(ENDPOINTS["1h"])
    
    def get_timeseries(self, item_id: int, timestep: str = "5m") -> Dict[str, Any]:
        """
        Fetch historical timeseries for a specific item
        
        Args:
            item_id: The item ID to fetch
            timestep: '5m', '1h', or '6h' (default: '5m')
        
        Returns: {
            "data": [
                {
                    "timestamp": int,
                    "avgHighPrice": int,
                    "highPriceVolume": int,
                    "avgLowPrice": int,
                    "lowPriceVolume": int
                }
            ]
        }
        
        Note: Max 365 data points returned
        """
        params = {
            "timestep": timestep,
            "id": item_id,
        }
        return self._fetch(ENDPOINTS["timeseries"], params=params)
    
    def close(self):
        """Close the HTTP client"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for one-off requests
def fetch_all_bulk_data() -> Dict[str, Any]:
    """
    Fetch all bulk data in one call session
    Returns mapping, latest, and 1h data
    """
    with OSRSPricesAPI() as api:
        return {
            "mapping": api.get_mapping(),
            "latest": api.get_latest(),
            "1h": api.get_1h(),
        }
