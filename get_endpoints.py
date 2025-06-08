import os
import time
import requests
from dotenv import load_dotenv
import pprint
import socket
from urllib.parse import urlparse

load_dotenv()

# Cache storage for multiple endpoints
cached_endpoints = []
cache_timestamp = None
CACHE_DURATION = 60 * 60  # 1 hour
# FALLBACK_API_URLS = os.getenv("FALLBACK_API_URLS", "").split(",")  # Comma-separated list
NGROK_API_KEY = os.getenv("NGROK_API_KEY")

# Add this function to test TCP connectivity

def test_tcp_connection(tcp_url, timeout=5):
    """
    Try to connect to the TCP endpoint (e.g., tcp://host:port).
    Returns True if connection succeeds, False otherwise.
    """
    if not tcp_url.startswith("tcp://"):
        return False
    parsed = urlparse(tcp_url)
    host = parsed.hostname
    port = parsed.port
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception as e:
        print(f"TCP connection failed for {tcp_url}: {e}")
        return False
    
def validate_endpoint(url: str) -> bool:
    if url.startswith("tcp://"):
        # Test TCP connection instead of /health
        return test_tcp_connection(url)
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        return health.get("status") == "healthy"
    except Exception as e:
        print(f"⚠️ Health check failed for {url}: {e}")
        return False

def fetch_with_retry(retries=3, backoff=1):
    for attempt in range(retries):
        try:
            headers = {
                "Authorization": f"Bearer {NGROK_API_KEY}",
                "Ngrok-Version": "2"
            }
            response = requests.get("https://api.ngrok.com/tunnels", headers=headers)
            response.raise_for_status()
            pprint.pprint(response.json())
            return response.json()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

def get_endpoints():
    global cached_endpoints, cache_timestamp

    # Use cache if still fresh
    if cached_endpoints and cache_timestamp and (time.time() - cache_timestamp) < CACHE_DURATION:
        print("🔄 Validating cached endpoints...")
        valid_cached = [url for url in cached_endpoints if validate_endpoint(url)]
        if valid_cached:
            print(f"✅ {len(valid_cached)} cached endpoint(s) are valid.")
            return valid_cached
        print("❌ Cached endpoints invalid or expired.")

    # Otherwise, fetch new tunnels
    try:
        data = fetch_with_retry()
        tunnels = data.get("tunnels", [])
        urls = [t["public_url"] for t in tunnels if "public_url" in t]

        print(f"🌐 Found {len(urls)} ngrok tunnel(s)")

        valid_urls = []
        for url in urls:
            if validate_endpoint(url):
                print(f"✅ Valid: {url}")
                valid_urls.append(url)
            else:
                print(f"❌ Invalid: {url}")

        if valid_urls:
            cached_endpoints = valid_urls
            cache_timestamp = time.time()
            return valid_urls

    except Exception as e:
        print(f"❌ Error fetching tunnels: {e}")

    return valid_urls

def get_tcp_endpoint(port="3128"):
    endpoints = get_endpoints()
    # Only return TCP tunnels forwarding to the SOCKS5 port
    for url in endpoints:
        if url.startswith("tcp://"):
            # Find the tunnel in the ngrok API response that matches this url
            data = fetch_with_retry()
            for tunnel in data.get("tunnels", []):
                if tunnel.get("public_url") == url and tunnel.get("proto") == "tcp":
                    if tunnel.get("forwards_to", "").endswith(f":{port}"):
                        return url
    return None

# Run script
if __name__ == "__main__":
    endpoints = get_endpoints()
    if endpoints:
        print("🎯 Valid Ngrok Endpoints:")
        for ep in endpoints:
            print(f"  - {ep}")
    else:
        print("🚫 No valid ngrok endpoints found.")
