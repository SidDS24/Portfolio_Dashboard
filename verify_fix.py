import requests
import json

def test_api():
    url = "http://127.0.0.1:8000/correlation/batch"
    payload = {
        "tickers": ["HYUNDAI.NS", "MARUTI.NS", "TATAMOTORS.NS", "TMCV.NS"],
        "index_tickers": ["^NSEI"],
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        results = response.json()
        
        print("\nResults Summary:")
        for res in results:
            ticker = res['ticker']
            benchmarks = res['benchmarks']
            # Get stats from the first benchmark
            first_bench = list(benchmarks.values())[0] if benchmarks else None
            stats = first_bench['stats'] if first_bench else {}
            
            print(f"Ticker: {ticker}")
            print(f"  Count: {stats.get('count')}")
            print(f"  Mean: {stats.get('mean'):.6f}")
            print(f"  Sum: {stats.get('sum'):.6f}")
            
            # Check correlation for Nifty 50
            if '^NSEI' in benchmarks:
                print(f"  Nifty 50 Correlation: {benchmarks['^NSEI']['correlation']:.4f}")
            print("-" * 20)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
