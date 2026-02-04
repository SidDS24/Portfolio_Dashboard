from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import analytics_engine
from typing import List, Dict, Optional
from pydantic import BaseModel

import json

print("Backend starting...")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, you can replace "*" with your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RankingResponse(BaseModel):
    all: List[Dict]
    micro: List[Dict]
    small: List[Dict]
    mid: List[Dict]
    large: List[Dict]

@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Backend is running"}

class BatchCorrelationRequest(BaseModel):
    tickers: List[str]
    index_tickers: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    period: str = "1y"

def df_to_dict(df: pd.DataFrame) -> List[Dict]:
    """Converts a DataFrame to a list of dictionaries, filling NaNs with 0."""
    return df.fillna(0).to_dict(orient="records")

@app.post("/rankings", response_model=RankingResponse)
async def get_rankings(files: List[UploadFile] = File(...)):
    """
    Receives multiple CSV files, processes them using the analytics_engine,
    and returns ranked data split by market cap.
    """
    all_dfs = []
    for file in files:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        all_dfs.append(df)
    
    if not all_dfs:
        raise HTTPException(status_code=400, detail="No files uploaded")
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Run the REAL math logic
    all_ranks, micro, small, mid, large = analytics_engine.calculate_company_rankings(master_df)
    
    # Prepare JSON response
    return {
        "all": df_to_dict(all_ranks),
        "micro": df_to_dict(micro),
        "small": df_to_dict(small),
        "mid": df_to_dict(mid),
        "large": df_to_dict(large)
    }

@app.get("/mappings")
async def get_mappings():
    try:
        with open("mappings.json", "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correlation/batch")
async def calculate_batch_correlation(req: BatchCorrelationRequest):
    """
    Calculates Correlation and Beta for multiple tickers against multiple indices.
    Uses bulk processing for speed and reliability.
    """
    try:
        results_map = analytics_engine.get_bulk_analytics(
            tickers=req.tickers,
            index_tickers=req.index_tickers,
            start_date=req.start_date,
            end_date=req.end_date,
            period=req.period
        )
        
        # Format for frontend: List of {ticker: str, benchmarks: {index: stats}}
        final_results = []
        for ticker in req.tickers:
            benchmarks = results_map.get(ticker, {})
            final_results.append({
                "ticker": ticker,
                "benchmarks": benchmarks
            })
        return final_results
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/quant")
async def export_quant_excel(req: BatchCorrelationRequest):
    """
    Generates a professional Excel report for Quantitative Analysis.
    """
    try:
        results_map = analytics_engine.get_bulk_analytics(
            tickers=req.tickers,
            index_tickers=req.index_tickers,
            start_date=req.start_date,
            end_date=req.end_date,
            period=req.period
        )
        
        # Flatten data for Excel
        export_data = []
        for ticker in req.tickers:
            benchmarks = results_map.get(ticker, {})
            # Get common stats from the first available benchmark
            stats = None
            if benchmarks:
                first_bench = next(iter(benchmarks.values()))
                stats = first_bench.get('stats', {})
            
            row = {
                "Ticker": ticker,
                "Sum": stats.get('sum', 0) if stats else 0,
                "Count": stats.get('count', 0) if stats else 0,
                "Mean": stats.get('mean', 0) if stats else 0,
                "Variance": stats.get('variance', 0) if stats else 0,
                "Std Dev": stats.get('std_dev', 0) if stats else 0
            }
            
            # Add Benchmarks
            for idx_ticker in req.index_tickers:
                bench_data = benchmarks.get(idx_ticker, {})
                row[f"{idx_ticker} Corr"] = bench_data.get('correlation', 0)
                row[f"{idx_ticker} Beta"] = bench_data.get('beta', 0)
                
            export_data.append(row)
            
        df = pd.DataFrame(export_data)
        
        # Create Excel in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Quant Analysis')
            
            # Formatting (Bold headers)
            worksheet = writer.sheets['Quant Analysis']
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
                
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=Quant_Analysis_Report.xlsx"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class PortfolioConfig(BaseModel):
    name: str
    tickers: List[str]
    allocation: Optional[float] = None # Percentage of total capital (0-100)
    capital: Optional[float] = None    # Direct capital amount (₹)
    custom_weights: Optional[Dict[str, float]] = None

class PortfolioAnalysisRequest(BaseModel):
    total_capital: float
    portfolios: List[PortfolioConfig]
    start_date: str
    end_date: str

@app.post("/portfolio/analyze")
async def analyze_portfolios(req: PortfolioAnalysisRequest):
    """
    Analyzes multiple portfolios simultaneously.
    Provides performance comparison and investment execution plan.
    """
    import yfinance as yf
    import numpy as np
    
    try:
        # 1. Collect all unique tickers
        all_unique_tickers = set()
        for p in req.portfolios:
            all_unique_tickers.update(p.tickers)
        
        if not all_unique_tickers:
            raise HTTPException(status_code=400, detail="No tickers provided")

        # 2. Download data for all at once
        data = yf.download(list(all_unique_tickers), start=req.start_date, end=req.end_date, progress=False)
        if 'Adj Close' in data:
            data = data['Adj Close']
        else:
            data = data['Close']
        
        data = data.ffill().bfill()
        
        results = []
        
        # 3. Process each portfolio
        for p_config in req.portfolios:
            # Filter tickers that actually have data
            valid_tickers = [t for t in p_config.tickers if t in data.columns]
            
            # Determine portfolio capital
            p_capital = 0
            if p_config.capital is not None:
                p_capital = p_config.capital
            elif p_config.allocation is not None:
                p_capital = req.total_capital * (p_config.allocation / 100)
            
            p_results = {
                "name": p_config.name,
                "capital": p_capital,
                "strategies": {},
                "status": "success",
                "message": ""
            }

            if not valid_tickers:
                p_results["status"] = "error"
                p_results["message"] = f"No historical data found for tickers in the selected range."
                results.append(p_results)
                continue
                
            p_data = data[valid_tickers].dropna(axis=1, how='all')
            if p_data.empty:
                p_results["status"] = "error"
                p_results["message"] = "Data found but contains too many missing values (NaN)."
                results.append(p_results)
                continue

            if len(p_data.columns) < 2 and len(p_config.tickers) > 1:
                p_results["status"] = "error"
                p_results["message"] = f"Only {len(p_data.columns)} asset(s) had sufficient data. Need at least 2 for optimization."
                results.append(p_results)
                continue
            
            # Strategies to evaluate
            strategies = ["max_sharpe", "min_volatility", "equal_weight", "random"]
            if p_config.custom_weights:
                strategies.append("custom")
            
            for strat in strategies:
                if strat in ["max_sharpe", "min_volatility", "random"]:
                    # Try simulation first
                    s_res = analytics_engine.monte_carlo_optimize(p_data, n_simulations=10000, objective=strat)
                    
                    # Fallback to EF if library is somehow available
                    if not s_res:
                        s_res = analytics_engine.optimize_portfolio(valid_tickers, req.start_date, req.end_date, data=p_data, objective=strat)
                    
                    if s_res:
                        p_results["strategies"][strat] = s_res
                
                elif strat == "equal_weight":
                    weights = {t: 1.0/len(p_data.columns) for t in p_data.columns}
                    stats = analytics_engine.get_portfolio_stats(p_data, weights)
                    if stats:
                        p_results["strategies"][strat] = {"weights": weights, "performance": stats}
                        
                elif strat == "custom":
                    # Normalize custom weights to ensure they sum to 1
                    raw_w = {t: v for t, v in p_config.custom_weights.items() if t in p_data.columns}
                    if not raw_w: continue
                    total_w = sum(raw_w.values())
                    weights = {t: v/total_w for t, v in raw_w.items()}
                    stats = analytics_engine.get_portfolio_stats(p_data, weights)
                    if stats:
                        p_results["strategies"][strat] = {"weights": weights, "performance": stats}
            
            results.append(p_results)
            
        return {
            "total_capital": req.total_capital,
            "portfolios": results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/portfolio")
async def export_portfolio_excel(req: PortfolioAnalysisRequest):
    """
    Generates a professional Excel report matching the user's requested format:
    Horizontal comparison with Portfolio Type, Metrics, and Asset Weights.
    """
    try:
        # 1. Re-run the analysis logic
        analysis_res = await analyze_portfolios(req)
        portfolios = analysis_res.get("portfolios", [])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for p in portfolios:
                if p.get("status") == "error": continue
                
                p_name = p["name"]
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
                print(f"Generating Excel for portfolio: {p_name}")
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
                # 2. Collect all unique tickers in THIS portfolio to create columns
                all_tickers = []
                for strat_res in p["strategies"].values():
                    all_tickers.extend(strat_res["weights"].keys())
                all_tickers = sorted(list(set(all_tickers)))
                
                # 3. Build horizontal rows for each strategy
                rows = []
                for strat_id, strat_res in p["strategies"].items():
                    perf = strat_res["performance"]
                    weights = strat_res["weights"]
                    
                    # Core metadata and performance
                    row = {
                        "Portfolio Type": strat_id.replace('_', ' ').title(),
                        "Expected Return (%)": round(perf["expected_return"] * 100, 2),
                        "Std Deviation (%)": round(perf["volatility"] * 100, 2),
                        "Sharpe Ratio": round(perf["sharpe_ratio"], 3)
                    }
                    
                    # Add individual asset weights as columns
                    for ticker in all_tickers:
                        w = weights.get(ticker, 0)
                        row[ticker] = f"{round(w * 100, 2)}%"
                        
                    rows.append(row)
                
                if rows:
                    p_df = pd.DataFrame(rows)
                    sheet_name = p_name[:31] # Excel limit
                    print(f"Adding sheet: {sheet_name} with {len(rows)} strategies")
                    p_df.to_excel(writer, index=False, sheet_name=sheet_name)
                    
                    # Professional Formatting
                    worksheet = writer.sheets[sheet_name]
                    # Bold Headers
                    for cell in worksheet[1]:
                        cell.font = cell.font.copy(bold=True)
                    
                    # Adjust column widths automatically
                    for i, col in enumerate(p_df.columns):
                        max_len = max(p_df[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.column_dimensions[chr(65+i)].width = max_len
<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
            
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=Portfolio_Strategy_Report_{req.total_capital}.xlsx"}
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        )
=======
>>>>>>> Stashed changes
            
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=Portfolio_Strategy_Report_{req.total_capital}.xlsx"}
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # For online hosting, the port is usually provided via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
