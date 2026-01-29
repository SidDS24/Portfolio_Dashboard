




!pip install yfinance openpyxl PyPortfolioOpt xlsxwriter scikit-learn ipywidgets --quiet

import pandas as pd
import numpy as np
import os
import json
import time
import random
from datetime import datetime
from collections import defaultdict
from scipy.stats import linregress
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import files
import analytics_engine

# Upload multiple CSV files and combine them into a single Excel workbook
uploaded = files.upload()
excel_path = "Sectors.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    for filename in uploaded.keys():
        df = pd.read_csv(filename)
        sheet_name = os.path.splitext(filename)[0][:31]
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Optional: download the combined workbook for the user
files.download(excel_path)

# Use the combined workbook for the rest of the analysis
file_name = excel_path
xls = pd.ExcelFile(file_name)

# Step 2: Sector Ranking
sector_stats = [] 
sector_dfs = {}

for sheet in xls.sheet_names:
    if sheet.lower() in ['sheet1', 'summary', 'readme']:
        continue
    df = pd.read_excel(xls, sheet)
    sector_dfs[sheet] = df  # Save for later
    if 'Sales G' in df.columns and 'Sales Q' in df.columns:
        sales_g = pd.to_numeric(df['Sales G'], errors='coerce')
        sales_q = pd.to_numeric(df['Sales Q'], errors='coerce')
        avg_g = sales_g.mean(skipna=True)
        avg_q = sales_q.mean(skipna=True)
        total_avg = avg_g + avg_q
        sector_stats.append({
            'Sector': sheet,
            'Avg Sales G': avg_g,
            'Avg Sales Q': avg_q,
            'Total Avg': total_avg
        })

sector_df = pd.DataFrame(sector_stats)
sector_df['Total Avg'] = pd.to_numeric(sector_df['Total Avg'], errors='coerce')
sector_df = sector_df.dropna(subset=['Total Avg'])
sector_df['Rank'] = sector_df['Total Avg'].rank(ascending=False, method='min').astype(int)
sector_df = sector_df.sort_values('Rank')


# Logic moved to analytics_engine.py
def company_ranking(df):
    return analytics_engine.calculate_company_rankings(df)

def move_rank_to_last(df):
    cols = list(df.columns)
    if 'Rank' in cols:
        cols.remove('Rank')
        cols.append('Rank')
        return df[cols]
    return df
# Step 4: Write Output Excel File
output_filename = 'Final_Sector_and_Company_Rankings.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    sector_df.to_excel(writer, sheet_name='Sector_Ranking', index=False)
    for sector, df in sector_dfs.items():
        if sector.lower() in ['sheet1', 'summary', 'readme']:
            continue
        ranked_companies, micro_cap, small_cap, mid_cap, large_cap = company_ranking(df)
        ranked_companies.to_excel(writer, sheet_name=sector[:31], index=False)
        micro_cap.to_excel(writer, sheet_name=f'{sector[:25]}_MicroCap', index=False)
        small_cap.to_excel(writer, sheet_name=f'{sector[:25]}_SmallCap', index=False)
        mid_cap.to_excel(writer, sheet_name=f'{sector[:25]}_MidCap', index=False)
        large_cap.to_excel(writer, sheet_name=f'{sector[:25]}_LargeCap', index=False)


files.download(output_filename)




# --- Load Mappings from JSON ---
with open('mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)

broad_index_map = mappings['broad_index_map']
sector_index_map = mappings['sector_index_map']
sector_company_map = mappings['sector_company_map']
sector_to_index = mappings['sector_to_index']
bond_asset_options = mappings['bond_asset_options']


# --- Widgets ---
broad_index_selector = widgets.SelectMultiple(
    options=[(name, symbol) for symbol, name in broad_index_map.items()],
    description="Broad Indices:",
    rows=5
)
sector_index_selector = widgets.SelectMultiple(
    options=[(name, symbol) for symbol, name in sector_index_map.items()],
    description="Sector Indices:",
    rows=6
)
sector_selector = widgets.SelectMultiple(
    options=list(sector_company_map.keys()),
    description="Sectors:",
    rows=6
)
company_selector = widgets.SelectMultiple(
    options=[],
    description="Companies:",
    rows=10
)
start_date_picker = widgets.DatePicker(
    description='Start Date',
    value=datetime(2001, 1, 1)
)
end_date_picker = widgets.DatePicker(
    description='End Date',
    value=datetime.today()
)
output = widgets.Output()

# --- Update company selector based on sector selection ---
def update_companies(*args):
    selected_sectors = list(sector_selector.value)
    global selected_sectors_global
    selected_sectors_global = selected_sectors
    companies = {}
    for sector in selected_sectors:
        companies.update(sector_company_map.get(sector, {}))
    options = [("ALL", "__ALL__")]
    options += [(name, symbol) for symbol, name in companies.items()]
    company_selector.options = options

sector_selector.observe(update_companies, 'value')

# --- Main analysis function ---
def fetch_data(b):
    output.clear_output()
    with output:
        broad_indices = list(broad_index_selector.value)
        sector_indices = list(sector_index_selector.value)
        selected_sectors = list(sector_selector.value)
        selected_companies = list(company_selector.value)
        start_date = start_date_picker.value
        end_date = end_date_picker.value

        if not (broad_indices or sector_indices) or not selected_sectors or not selected_companies:
            print(" Please select at least one index, sector, and company.")
            return

        # Prepare all tickers for download
        all_companies = {}
        for sector in selected_sectors:
            all_companies.update(sector_company_map.get(sector, {}))
        if "__ALL__" in selected_companies:
            companies_by_sector = {sector: list(sector_company_map[sector].keys()) for sector in selected_sectors}
        else:
            companies_by_sector = {sector: [] for sector in selected_sectors}
            for symbol in selected_companies:
                for sector in selected_sectors:
                    if symbol in sector_company_map[sector]:
                        companies_by_sector[sector].append(symbol)
                        break

        # Collect all tickers to download
        all_tickers = set(broad_indices + sector_indices)
        for symlist in companies_by_sector.values():
            all_tickers.update(symlist)
        all_tickers = list(all_tickers)
        if not all_tickers:
            print(" No tickers to download.")
            return

        try:
            data = yf.download(
                tickers=all_tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
        except Exception as e:
            print(f" Error fetching data: {e}")
            return

        # Show a table for each sector
        sector_dfs = {}
        for sector in selected_sectors:
            summary_list = []
            relevant_sector_index = sector_to_index.get(sector)
            relevant_sector_index_selected = relevant_sector_index in sector_indices
            sector_index_name = sector_index_map.get(relevant_sector_index, relevant_sector_index)

            for company_symbol in companies_by_sector[sector]:
                company_name = sector_company_map[sector].get(company_symbol, company_symbol.replace('.NS', ''))
                if len(all_tickers) == 1:
                    df = data[['Close']].reset_index()
                else:
                    if company_symbol not in data:
                        print(f" No data for {company_name}")
                        continue
                    df = data[company_symbol][['Close']].reset_index()
                df['Daily Return'] = df['Close'].pct_change(fill_method=None)

                summary = {
                    'Company': company_name,
                    'Sum': df['Daily Return'].sum(),
                    'Count': df['Daily Return'].count(),
                    'Mean': df['Daily Return'].mean(),
                    'Variance': df['Daily Return'].var(),
                    'Std Dev': df['Daily Return'].std()
                }

                # Correlation and beta with all selected broad indices
                for idx in broad_indices:
                    idx_name = broad_index_map.get(idx, idx)
                    if idx not in data:
                        summary[f'Corr ({idx_name})'] = None
                        summary[f'Beta ({idx_name})'] = None
                        continue
                    idx_df = data[idx][['Close']].reset_index()
                    idx_df['Index Return'] = idx_df['Close'].pct_change(fill_method=None)
                    merged = pd.merge(df, idx_df[['Date', 'Index Return']], on='Date', how='inner')
                    merged.dropna(inplace=True)
                    if not merged.empty:
                        corr, beta = analytics_engine.get_correlation_and_beta(data, company_symbol, idx, broad_index_map)
                        summary[f'Corr ({idx_name})'] = corr
                        summary[f'Beta ({idx_name})'] = beta
                    else:
                        summary[f'Corr ({idx_name})'] = None
                        summary[f'Beta ({idx_name})'] = None

                # Correlation and beta with relevant sector index (if selected and mapped)
                if relevant_sector_index and relevant_sector_index_selected:
                    if relevant_sector_index not in data:
                        summary[f'Corr ({sector_index_name})'] = None
                        summary[f'Beta ({sector_index_name})'] = None
                    else:
                        idx_df = data[relevant_sector_index][['Close']].reset_index()
                        idx_df['Index Return'] = idx_df['Close'].pct_change(fill_method=None)
                        merged = pd.merge(df, idx_df[['Date', 'Index Return']], on='Date', how='inner')
                        merged.dropna(inplace=True)
                        if not merged.empty:
                            corr = merged['Daily Return'].corr(merged['Index Return'])
                            beta = linregress(merged['Index Return'], merged['Daily Return']).slope
                            summary[f'Corr ({sector_index_name})'] = corr
                            summary[f'Beta ({sector_index_name})'] = beta
                        else:
                            summary[f'Corr ({sector_index_name})'] = None
                            summary[f'Beta ({sector_index_name})'] = None

                summary_list.append(summary)

            if summary_list:
                summary_df = pd.DataFrame(summary_list)
                sector_dfs[sector] = summary_df
                print(f"\n=== Sector: {sector} ===")
                summary_df = pd.DataFrame(summary_list)
                display(summary_df)
            else:
                print(f"\n=== Sector: {sector} ===")
                print(" No valid data to display for this sector.")

         # --- Export to Excel ---
        if sector_dfs:
            excel_filename = "sector_analysis_results.xlsx"
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                for sector, df in sector_dfs.items():
                    # Sheet names can't be longer than 31 chars
                    sheet_name = sector[:31]
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            try:
                from google.colab import files
                files.download(excel_filename)
                print(f" Excel file '{excel_filename}' created and downloaded.")
            except ImportError:
                print(f" Excel file '{excel_filename}' created in your working directory.")

# --- Button ---
fetch_button = widgets.Button(description="Run Analysis")
fetch_button.on_click(fetch_data)

# --- Display Widgets ---
display(broad_index_selector, sector_index_selector, sector_selector, company_selector,
        start_date_picker, end_date_picker, fetch_button, output)




# ---- Ticker and Asset Setup ----


# Assume sector_company_map and selected_sectors_global are pre-populated
company_options = []
if 'selected_sectors_global' in globals() and isinstance(selected_sectors_global, (list, tuple, set)):
    for sector in selected_sectors_global:
        if sector in sector_company_map:
            company_options.extend([
                (f"{name} ({symbol})", symbol) for symbol, name in sector_company_map[sector].items()
            ])

# ---- Helper Functions ----

def get_latest_start_date(tickers):
    min_dates = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="max")
            if not hist.empty:
                min_dates.append(hist.index[0])
        except Exception as e:
            print(f"Error fetching {t}: {e}")
    if min_dates:
        return max(min_dates).strftime("2024-09-01")
    else:
        return "2025-04-01"

end_date = datetime.today()
end_date_str = end_date.strftime('%Y-%m-%d')

# ---- Dynamic UI ----

num_portfolios = widgets.IntText(value=2, description="No. of Portfolios:")
total_capital = widgets.FloatText(value=10000000, description="Total Capital (₹):")
portfolio_boxes = widgets.VBox()

def build_portfolio_widgets(N, company_options, bond_options):
    portfolio_widgets = []
    for i in range(N):
        company_sel = widgets.SelectMultiple(
            options=company_options,
            description=f'Companies P{i+1}',
            layout=widgets.Layout(width='70%', height='120px')
        )
        bond_sel = widgets.SelectMultiple(
            options=[(v, k) for k, v in bond_options.items()],
            description=f'Bonds P{i+1}',
            layout=widgets.Layout(width='70%', height='80px')
        )
        alloc_slider = widgets.FloatSlider(
            value=round(100.0/N, 2), min=0, max=100, step=1,
            description=f'Alloc P{i+1} (%)',
            layout=widgets.Layout(width='60%')
        )
        weight_box = widgets.Text(
            description="Custom Weights:",
            placeholder="e.g. 10 ,33,6,32,7,11",
            layout=widgets.Layout(width='60%')
        )
        box = widgets.VBox([company_sel, bond_sel, alloc_slider, weight_box])
        portfolio_widgets.append(box)
    return portfolio_widgets

def update_portfolio_boxes(change=None):
    N = num_portfolios.value
    portfolio_boxes.children = build_portfolio_widgets(N, company_options, bond_asset_options)

num_portfolios.observe(update_portfolio_boxes, names='value')
update_portfolio_boxes()

output = widgets.Output()

# ---- Portfolio Analysis ----

def run_analysis(_):
    with output:
        clear_output()
        N = num_portfolios.value
        capital = total_capital.value
        all_portfolios = []
        allocs = []
        custom_weights_input = []

        for box in portfolio_boxes.children:
            company_sel, bond_sel, alloc_slider, weight_box = box.children
            tickers = list(company_sel.value) + list(bond_sel.value)
            all_portfolios.append(tickers)
            allocs.append(alloc_slider.value)
            custom_weights_input.append(weight_box.value.strip())

        # Validation
        if abs(sum(allocs) - 100) > 1e-2:
            print("⚠️ Total Allocation must equal 100%.")
            return
        if any(len(tickers) < 2 for tickers in all_portfolios):
            print("⚠️ Each portfolio must have at least 2 assets selected.")
            return

        writer = pd.ExcelWriter("portfolio_results.xlsx", engine='openpyxl')
        summary = []
        combined_weights = defaultdict(float)
        combined_returns, combined_stddevs, combined_sharpes = [], [], []

        for idx, group in enumerate(all_portfolios):
            print(f"Analyzing Portfolio {idx+1}: {group}")
            try:
                start_date_dyn = get_latest_start_date(group)
                data = yf.download(group, start=start_date_dyn, end=end_date_str)['Close']
                data = data.ffill().bfill().dropna(axis=1, thresh=int(len(data) * 0.7))
                if len(data.columns) < 2:
                    print(f"⚠️ Portfolio {idx+1}: Insufficient data.")
                    continue

                mu = expected_returns.mean_historical_return(data)
                S = risk_models.sample_cov(data)
                group = list(data.columns)

                def safe_pf(tickers, obj="max_sharpe"):
                    result = analytics_engine.optimize_portfolio(tickers, start_date_dyn, end_date_str, objective=obj)
                    if result:
                        return result["weights"], (result["performance"]["expected_return"], result["performance"]["volatility"], result["performance"]["sharpe_ratio"])
                    else:
                        return {t: 0 for t in group}, (np.nan, np.nan, np.nan)

                # --- Custom user weights ---
                user_input = custom_weights_input[idx]
                user_weights = {}
                if user_input:
                    try:
                        vals = [float(x.strip()) for x in user_input.split(',')]
                        if len(vals) == len(group):
                            normed = np.array(vals) / np.sum(vals)
                            user_weights = dict(zip(group, normed))
                            r_user = np.dot(normed, mu)
                            s_user = np.sqrt(np.dot(normed.T, S.dot(normed)))
                            sh_user = r_user / s_user if s_user > 0 else np.nan
                        else:
                            print(f"⚠️ Portfolio {idx+1}: Weight count mismatch. Ignoring manual weights.")
                    except:
                        print(f"⚠️ Portfolio {idx+1}: Invalid manual weights format.")
                else:
                    user_weights = {t: 1/len(group) for t in group}
                    r_user = np.dot(list(user_weights.values()), mu)
                    s_user = np.sqrt(np.dot(list(user_weights.values()), S.dot(list(user_weights.values()))))
                    sh_user = r_user / s_user

                # --- Other portfolio types ---
                ew_weights = {t: 1/len(group) for t in group}
                ew_ret = np.dot(list(ew_weights.values()), mu)
                ew_std = np.sqrt(np.dot(list(ew_weights.values()), S.dot(list(ew_weights.values()))))
                ew_sharpe = ew_ret / ew_std

                max_sharpe, (ms_ret, ms_std, ms_sharpe) = safe_pf(group, "max_sharpe")
                min_var, (mv_ret, mv_std, mv_sharpe) = safe_pf(group, "min_volatility")
                rand_w = np.random.dirichlet(np.ones(len(group)))
                rand_ret = np.dot(rand_w, mu)
                rand_std = np.sqrt(np.dot(rand_w.T, S.dot(rand_w)))
                rand_sharpe = rand_ret / rand_std
                rand_weights = dict(zip(group, rand_w))

                # Combine results
                all_rows = []
                for typ, w, r, s, sh in [
                    ("Your Weights", user_weights, r_user, s_user, sh_user),
                    ("Equal Weight", ew_weights, ew_ret, ew_std, ew_sharpe),
                    ("Max Sharpe", max_sharpe, ms_ret, ms_std, ms_sharpe),
                    ("Min Variance", min_var, mv_ret, mv_std, mv_sharpe),
                    ("Random", rand_weights, rand_ret, rand_std, rand_sharpe)
                ]:
                    row = {
                        "Portfolio Type": typ,
                        "Expected Return (%)": round(r*100, 2) if pd.notna(r) else "N/A",
                        "Std Deviation (%)": round(s*100, 2) if pd.notna(s) else "N/A",
                        "Sharpe Ratio": round(sh, 3) if pd.notna(sh) else "N/A"
                    }
                    row.update({t: f"{w.get(t,0)*100:.2f}%" for t in group})
                    all_rows.append(row)

                pd.DataFrame(all_rows).to_excel(writer, sheet_name=f"Portfolio {idx+1}"[:31], index=False)

            except Exception as e:
                print(f"⚠️ Error analyzing portfolio {idx+1}: {e}")

        writer.close()
        print("✅ Analysis complete: portfolio_results.xlsx")
        if os.path.exists("portfolio_results.xlsx"):
            files.download("portfolio_results.xlsx")

btn = widgets.Button(description="Run Portfolio Analysis", button_style='success')
btn.on_click(run_analysis)

# ---- Display the Dynamic UI ----

display(widgets.VBox([
    num_portfolios,
    total_capital,
    portfolio_boxes,
    btn,
    output
]))















