import pandas as pd
import numpy as np
from collections import defaultdict

def calculate_company_rankings(df):
    """
    Analyzes and ranks companies based on fundamental financial metrics.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.title()
    
    # Data Cleaning
    cols_to_numeric = [
        'Pledged Percentage', 'Debtor Days', 'Public Holding', 'Promoter Holding',
        'Qoq Sales', 'Qoq Opm', 'Qoq Net Profit', 'Qoq Pbt',
        'Yoy Sales', 'Yoy Opm', 'Yoy Pbt', 'Yoy Net Profit',
        'Return On Capital Employed', 'Return On Equity', 'Market Capitalization'
    ]
    for col in cols_to_numeric:
        if col in df.columns:
            if col == 'Pledged Percentage':
                df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Scoring Logic
    score_cols = []
    
    if 'Pledged Percentage' in df.columns:
        df['Pledge Score'] = df['Pledged Percentage'].apply(lambda x: 1 if x == 0 else 0)
        score_cols.append('Pledge Score')
        
    if 'Debtor Days' in df.columns:
        def debtor_score(v):
            if 0 <= v <= 30: return 5
            elif 30 < v < 60: return 4
            elif 60 <= v < 90: return 3
            elif 90 <= v < 120: return 2
            elif 120 <= v < 180: return 1
            return 0
        df['Debtor Days Score'] = df['Debtor Days'].apply(debtor_score)
        score_cols.append('Debtor Days Score')

    # Fundamental Growth Scores
    def growth_score(v):
        if pd.isna(v): return 0
        return 1 if v >= 20 else 0.5 if v >= 1 else 0

    growth_metrics = ['Qoq Sales', 'Qoq Net Profit', 'Yoy Sales']
    for metric in growth_metrics:
        if metric in df.columns:
            score_name = f'{metric} Score'
            df[score_name] = df[metric].apply(growth_score)
            score_cols.append(score_name)

    # ROCE & ROE Scores
    def ratio_score(v):
        if v >= 25: return 5
        elif v >= 20: return 4
        elif v >= 15: return 3
        elif v >= 10: return 2
        elif v >= 5: return 1
        return 0

    if 'Return On Capital Employed' in df.columns:
        df['ROCE Score'] = df['Return On Capital Employed'].apply(ratio_score)
        score_cols.append('ROCE Score')
    if 'Return On Equity' in df.columns:
        df['ROE Score'] = df['Return On Equity'].apply(ratio_score)
        score_cols.append('ROE Score')

    # Calculate Totals
    df['Total Score'] = df[score_cols].sum(axis=1)
    
    # Ranking
    df = df.sort_values(by=['Total Score', 'Market Capitalization'], ascending=[False, False]).reset_index(drop=True)
    df['Rank'] = df.index + 1

    # Market Cap Categorization
    def get_cap_cat(mc):
        if mc < 2000: return 'Micro Cap'
        elif mc < 5000: return 'Small Cap'
        elif mc < 15000: return 'Mid Cap'
        return 'Large Cap'
    
    df['Market Cap Category'] = df['Market Capitalization'].apply(get_cap_cat)
    
    # Split DataFrames
    market_cap_cols = [
        'Rank', 'Name', 'Nse Code', 'Industry Group', 'Market Capitalization', 
        'Current Price', 'Total Score', 'ROCE Score', 'ROE Score', 
        'Return On Capital Employed', 'Return On Equity'
    ]
    # Filter columns that actually exist
    existing_cols = [col for col in market_cap_cols if col in df.columns]
    
    micro = df[df['Market Cap Category'] == 'Micro Cap'][existing_cols].copy()
    small = df[df['Market Cap Category'] == 'Small Cap'][existing_cols].copy()
    mid = df[df['Market Cap Category'] == 'Mid Cap'][existing_cols].copy()
    large = df[df['Market Cap Category'] == 'Large Cap'][existing_cols].copy()

    # Category Ranks (Individual ranks within each category)
    for d in [micro, small, mid, large]:
        if not d.empty:
            d['Rank'] = range(1, len(d) + 1)

    return df, micro, small, mid, large

def get_bulk_analytics(tickers, index_tickers, start_date=None, end_date=None, period="1y"):
    """
    Calculates Correlation and Beta for a batch of tickers against multiple benchmarks.
    Follows the 'Math Behind' logic from PS1.py: batch download + direct return correlation.
    """
    import yfinance as yf
    from scipy.stats import linregress
    
    try:
        all_tickers = list(set(tickers + index_tickers))
        print(f"[QUANT] Downloading data for {len(all_tickers)} tickers...")
        
        if start_date and end_date:
            data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        else:
            data = yf.download(all_tickers, period=period, progress=False)

        if data.empty:
            print(f"[QUANT] Error: yfinance returned no data for {all_tickers}")
            return {}

        # ROBUST PRICE EXTRACTION
        prices = pd.DataFrame(index=data.index)
        
        for t in all_tickers:
            try:
                # Check if ticker has data in the bulk download
                if ('Close', t) in data.columns:
                    ticker_data = data[('Close', t)]
                elif t in data.columns:
                    ticker_data = data[t]
                elif len(all_tickers) == 1 and 'Close' in data.columns:
                    ticker_data = data['Close']
                else:
                    ticker_data = pd.Series()

                # If ticker data is missing or empty, try individual download as fallback
                if ticker_data.dropna().empty:
                    print(f"[QUANT] Missing data for {t} in bulk download. Falling back to individual download...")
                    if start_date and end_date:
                        ind_data = yf.download(t, start=start_date, end=end_date, progress=False)
                    else:
                        ind_data = yf.download(t, period=period, progress=False)
                    
                    if not ind_data.empty:
                        if 'Close' in ind_data.columns:
                            ticker_data = ind_data['Close']
                        elif ('Close', t) in ind_data.columns:
                            ticker_data = ind_data[('Close', t)]
                
                if not ticker_data.empty:
                    prices[t] = ticker_data
            except Exception as e:
                print(f"[QUANT] Skipping {t} due to extraction error: {e}")

        if prices.empty:
            print("[QUANT] Error: Failed to extract any price columns.")
            return {}

        print(f"[QUANT] Extracted {len(prices.columns)} valid price columns.")
        returns = prices.pct_change()
        
        results = {}
        for t in tickers:
            results[t] = {}
            # Get clean returns for this specific ticker for stats
            # This handles partial data automatically (e.g. 3 months of returns if available)
            ticker_returns = returns[t].dropna() if t in returns.columns else pd.Series()
            
            asset_stats = {
                "count": len(ticker_returns),
                "sum": float(ticker_returns.sum()) if not ticker_returns.empty else 0,
                "mean": float(ticker_returns.mean()) if not ticker_returns.empty and not pd.isna(ticker_returns.mean()) else 0,
                "variance": float(ticker_returns.var()) if not ticker_returns.empty and not pd.isna(ticker_returns.var()) else 0,
                "std_dev": float(ticker_returns.std()) if not ticker_returns.empty and not pd.isna(ticker_returns.std()) else 0
            }

            if t not in returns.columns or ticker_returns.empty:
                print(f"[QUANT] Warning: No return data for {t}. Using zero stats.")
                for idx in index_tickers:
                    results[t][idx] = { "correlation": 0, "beta": 0, "count": 0, "stats": asset_stats }
                continue
                
            for idx in index_tickers:
                results[t][idx] = { "correlation": 0, "beta": 0, "count": 0, "stats": asset_stats }
                if idx not in returns.columns:
                    continue
                
                # Use Intersection of dates (Available overlap)
                pair_data = returns[[t, idx]].dropna()
                
                if len(pair_data) >= 3: # Lowered threshold to 3 points for very new stocks
                    try:
                        corr = pair_data[t].corr(pair_data[idx])
                        slope, intercept, r_val, p_val, std_err = linregress(pair_data[idx], pair_data[t])
                        
                        results[t][idx]["correlation"] = float(corr) if not pd.isna(corr) else 0
                        results[t][idx]["beta"] = float(slope) if not pd.isna(slope) else 0
                        results[t][idx]["count"] = int(len(pair_data))
                    except:
                        pass
        
        print(f"[QUANT] Processing complete for {len(tickers)} assets.")
        return results
    except Exception as e:
        print(f"[QUANT] Bulk Error: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_correlation_and_beta(asset_ticker, index_ticker, start_date=None, end_date=None, period="1y"):
    """
    Fallback for single-pair calculation.
    """
    res = get_bulk_analytics([asset_ticker], [index_ticker], start_date, end_date, period)
    return res.get(asset_ticker, {}).get(index_ticker, {"correlation": 0, "beta": 0, "count": 0})

def get_portfolio_stats(data, weights):
    """
    Calculates portfolio performance for a given set of weights.
    Returns (expected_return, volatility, sharpe_ratio).
    Includes manual math fallbacks if pypfopt is missing.
    """
    try:
        # Prepare returns
        returns = data.pct_change().dropna()
        if returns.empty: return None
        
        # 1. Calculate Expected Returns (Annualized)
        mu = returns.mean() * 252
        
        # 2. Calculate Covariance Matrix (Annualized)
        S = returns.cov() * 252
        
        # Prepare weight vector
        w_vector = np.array([weights.get(t, 0) for t in data.columns])
        
        # 3. Compute Performance
        exp_ret = np.dot(w_vector, mu)
        vol = np.sqrt(np.dot(w_vector.T, np.dot(S, w_vector)))
        sharpe = exp_ret / vol if vol > 0 else 0
        
        return {
            "expected_return": float(exp_ret),
            "volatility": float(vol),
            "sharpe_ratio": float(sharpe)
        }
    except Exception as e:
        print(f"Stats Error: {e}")
        return None

def monte_carlo_optimize(data, n_simulations=10000, objective="max_sharpe"):
    """
    Finds the best portfolio weights using Monte Carlo simulation.
    Bypasses need for C++ solvers.
    """
    try:
        returns = data.pct_change().dropna()
        if returns.empty: return None
        
        mu = returns.mean() * 252
        S = returns.cov() * 252
        num_assets = len(data.columns)
        
        results = np.zeros((3, n_simulations))
        weights_record = []
        
        for i in range(n_simulations):
            # Generate random weights
            w = np.random.random(num_assets)
            w /= np.sum(w) # Normalize to 1
            weights_record.append(w)
            
            # Calculate performance
            p_ret = np.dot(w, mu)
            p_vol = np.sqrt(np.dot(w.T, np.dot(S, w)))
            
            results[0,i] = p_ret
            results[1,i] = p_vol
            results[2,i] = p_ret / p_vol if p_vol > 0 else 0
            
        if objective == "max_sharpe":
            best_idx = np.argmax(results[2])
        elif objective == "min_volatility":
            best_idx = np.argmin(results[1])
        else: # random
            best_idx = np.random.randint(0, n_simulations)
            
        best_w = weights_record[best_idx]
        best_weights = {ticker: float(weight) for ticker, weight in zip(data.columns, best_w)}
        
        return {
            "weights": best_weights,
            "performance": {
                "expected_return": float(results[0, best_idx]),
                "volatility": float(results[1, best_idx]),
                "sharpe_ratio": float(results[2, best_idx])
            }
        }
    except Exception as e:
        print(f"MC Optimization Error: {e}")
        return None

def optimize_portfolio(tickers, start_date, end_date, weight_bounds=(0, 1), objective="max_sharpe", data=None):
    """
    Performs Markowitz Portfolio Optimization.
    Can accept pre-downloaded data for performance.
    """
    try:
        import yfinance as yf
        from pypfopt import EfficientFrontier, expected_returns, risk_models
        
        if data is None:
            raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            if 'Adj Close' in raw_data:
                data = raw_data['Adj Close']
            else:
                data = raw_data['Close']
            
        data = data.ffill().bfill().dropna(axis=1, how='all')
        
        if len(data.columns) < 2:
            return None
            
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        
        try:
            if objective == "max_sharpe":
                ef.max_sharpe()
            elif objective == "min_volatility":
                ef.min_volatility()
            
            weights = ef.clean_weights()
            perf = ef.portfolio_performance()
            return {
                "weights": dict(weights),
                "performance": {
                    "expected_return": float(perf[0]),
                    "volatility": float(perf[1]),
                    "sharpe_ratio": float(perf[2])
                }
            }
        except Exception as e:
            print(f"EF Error: {e}")
            return None
    except Exception as e:
        print(f"Optimization Error: {e}")
        return None
