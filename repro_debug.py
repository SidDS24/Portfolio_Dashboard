import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import datetime

def test_batch():
    # Adding a dead ticker and a larger batch
    tickers = [
        "HYUNDAI.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "TMCV.NS", "MARUTI.NS",
        "NONEXISTENT_TICKER.NS", "ATHENERG.NS", "TATAMOTORS.NS",
        "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS"
    ]
    index_tickers = ["^NSEI", "^CRSLDX", "^NSEBANK"]
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    
    all_tickers = list(set(tickers + index_tickers))
    print(f"Downloading for: {all_tickers}")
    
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns[:10]}...")
    
    if data.empty:
        print("DATA IS EMPTY!")
        return

    # Price extraction check
    prices = pd.DataFrame(index=data.index)
    for t in all_tickers:
        if ('Close', t) in data.columns:
            prices[t] = data[('Close', t)]
        elif t in data.columns:
            prices[t] = data[t]
        elif len(all_tickers) == 1 and 'Close' in data.columns:
            prices[t] = data['Close']
        else:
            print(f"FAILED TO EXTRACT: {t}")

    print(f"Extracted prices shape: {prices.shape}")
    print(f"First few prices for {tickers[0]}:\n{prices[tickers[0]].head()}")
    
    returns = prices.pct_change()
    print(f"Returns for {tickers[0]} (head):\n{returns[tickers[0]].head()}")
    
    ticker_returns = returns[tickers[0]].dropna()
    print(f"Cleaned returns count for {tickers[0]}: {len(ticker_returns)}")

if __name__ == "__main__":
    test_batch()
