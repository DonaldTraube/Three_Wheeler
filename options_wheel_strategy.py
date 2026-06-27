#!/usr/bin/env python3
"""
Options Wheel Strategy Backtester with ROI Estimation
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
PARAMS = {
    'data_path': 'Stonks - Grok (2).csv',   # Change only if filename differs
    'tickers': ['SOXL', 'TNA', 'FAS', 'UPRO', 'NAIL'],
    'sma_months': 10,
    'trend_threshold': 1.10,
    'momentum_weeks_lookback': 14,
    'put_strike_discount': 0.99,
    'cc_strike_premium': 1.05,
    'csp_premium_pct': 0.015,   # Estimated 1.5% of strike for CSP
    'cc_premium_pct': 0.009,    # Estimated 0.9% of strike for CC
    'output_dir': './plots',
    'start_date': None,
}

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.set_index('Date').sort_index()
    for col in PARAMS['tickers']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(0, np.nan)
    if PARAMS['start_date']:
        df = df[df.index >= PARAMS['start_date']]
    return df

def calculate_monthly_trend_filter(df, ticker):
    monthly = df[[ticker]].resample('M').last().rename(columns={ticker: 'm_close'})
    monthly['sma_10m'] = monthly['m_close'].rolling(
        window=PARAMS['sma_months'], min_periods=5
    ).mean()
    monthly['prior_m_close'] = monthly['m_close'].shift(1)
    monthly['sma_10m_lag'] = monthly['sma_10m'].shift(1)
    monthly['trend_ok'] = (monthly['prior_m_close'] > (monthly['sma_10m_lag'] * PARAMS['trend_threshold'])).astype(bool)
    monthly['trend_ok'] = monthly['trend_ok'].ffill()
    monthly['prior_m_close_ff'] = monthly['prior_m_close'].ffill()
    monthly['sma_10m_ff'] = monthly['sma_10m_lag'].ffill()
    df_out = df.join(monthly[['trend_ok', 'prior_m_close_ff', 'sma_10m_ff']], how='left')
    df_out['trend_ok'] = df_out['trend_ok'].ffill().fillna(False)
    df_out['prior_month_close'] = df_out['prior_m_close_ff']
    df_out['sma_10m'] = df_out['sma_10m_ff']
    return df_out.drop(columns=['prior_m_close_ff', 'sma_10m_ff'], errors='ignore')

def calculate_weekly_momentum(df, ticker):
    weekly = df[[ticker]].resample('W-FRI').last().rename(columns={ticker: 'w_close'})
    weekly['mom_raw'] = weekly['w_close'].shift(1) - weekly['w_close'].shift(PARAMS['momentum_weeks_lookback'])
    weekly['mom_ok'] = (weekly['mom_raw'] > 0).astype(bool)
    weekly['mom_ok_lagged'] = weekly['mom_ok'].shift(1)
    daily_mom = weekly[['mom_ok_lagged', 'mom_raw']].reindex(df.index, method='ffill')
    daily_mom = daily_mom.rename(columns={'mom_ok_lagged': 'mom_ok', 'mom_raw': 'momentum'})
    return daily_mom

def prepare_ticker_data(df, ticker):
    d = df[[ticker]].copy()
    d = calculate_monthly_trend_filter(d, ticker)
    mom_df = calculate_weekly_momentum(df, ticker)
    d = d.join(mom_df, how='left')
    d['csp_signal'] = (d['trend_ok'] & d['mom_ok']).astype(bool)
    d['sma_200d'] = d[ticker].rolling(window=200, min_periods=50).mean()
    weekly_agg = df[[ticker]].resample('W-FRI').agg(['min', 'max', 'last'])
    weekly_agg.columns = ['_'.join(col).strip() for col in weekly_agg.columns.values]
    weekly_agg = weekly_agg.rename(columns={
        f'{ticker}_min': 'w_low',
        f'{ticker}_max': 'w_high',
        f'{ticker}_last': 'w_close'
    })
    d = d.join(weekly_agg[['w_low', 'w_high', 'w_close']], how='left')
    d[['w_low', 'w_high', 'w_close']] = d[['w_low', 'w_high', 'w_close']].ffill()
    return d

def create_signal_chart(ticker_df, ticker, save_html=True):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.65, 0.17, 0.18],
                        subplot_titles=(f"{ticker} - Price & CSP Signal Zones", "Trend Filter", "Momentum"))
    fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df[ticker], mode='lines', name='Close', line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df['sma_200d'], mode='lines', name='200d SMA', line=dict(color='orange', dash='dash')), row=1, col=1)
    valid_sma = ticker_df['sma_10m'].notna()
    fig.add_trace(go.Scatter(x=ticker_df.index[valid_sma], y=ticker_df.loc[valid_sma, 'sma_10m'], mode='lines', name='10M SMA', line=dict(color='red', dash='dot')), row=1, col=1)
    signal_mask = ticker_df['csp_signal']
    if signal_mask.any():
        fig.add_trace(go.Scatter(x=ticker_df.index[signal_mask], y=ticker_df.loc[signal_mask, ticker], mode='markers', name='CSP Signal', marker=dict(color='limegreen', size=8, symbol='triangle-up')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticker_df.index, y=ticker_df['trend_ok'].astype(int), mode='lines', name='Trend OK', line=dict(color='#2ca02c'), fill='tozeroy'), row=2, col=1)
    mom_plot = ticker_df['mom_ok'].fillna(False).astype(int)
    fig.add_trace(go.Scatter(x=ticker_df.index, y=mom_plot, mode='lines', name='Momentum OK', line=dict(color='#d62728'), fill='tozeroy'), row=3, col=1)
    fig.update_layout(title=f"{ticker} Wheel Signals", height=900, template='plotly_white')
    if save_html:
        ensure_dir(PARAMS['output_dir'])
        fig.write_html(os.path.join(PARAMS['output_dir'], f"{ticker}_wheel_signals.html"))
    return fig

def simulate_wheel(ticker_df, ticker):
    weekly = ticker_df[['w_low', 'w_high', 'w_close', 'prior_month_close', 'csp_signal']].dropna(subset=['w_close'])
    weekly = weekly.resample('W-FRI').last()
    trades = []
    holding = False
    cost_basis = None
    csp_sold = cc_sold = assignments = cc_called = weeks_in_cc = 0
    for week_end, row in weekly.iterrows():
        signal = bool(row['csp_signal'])
        prior_m = row['prior_month_close']
        if pd.isna(prior_m):
            continue
        if not holding:
            if signal:
                strike = np.floor(prior_m * PARAMS['put_strike_discount'])
                capital = strike * 100
                premium = strike * PARAMS['csp_premium_pct'] * 100
                roi = (premium / capital) * 100
                csp_sold += 1
                assigned = (row['w_low'] < strike) or (row['w_close'] < strike)
                trades.append({'week_end': week_end, 'action': 'SELL_CSP', 'strike': strike, 'premium': premium, 'roi_pct': roi, 'assigned': assigned})
                if assigned:
                    assignments += 1
                    holding = True
                    cost_basis = strike
        else:
            weeks_in_cc += 1
            cc_strike = np.floor(cost_basis * PARAMS['cc_strike_premium'])
            premium = cc_strike * PARAMS['cc_premium_pct'] * 100
            roi = (premium / (cost_basis * 100)) * 100
            cc_sold += 1
            called = (row['w_high'] > cc_strike) or (row['w_close'] > cc_strike)
            trades.append({'week_end': week_end, 'action': 'SELL_CC', 'strike': cc_strike, 'premium': premium, 'roi_pct': roi, 'called_away': called})
            if called:
                cc_called += 1
                holding = False
                cost_basis = None
    trade_log = pd.DataFrame(trades)
    stats = {
        'ticker': ticker,
        'csp_sold': csp_sold,
        'assignments': assignments,
        'assignment_rate': (assignments / csp_sold * 100) if csp_sold > 0 else 0,
        'cc_sold': cc_sold,
        'cc_called_away': cc_called,
        'cc_call_rate': (cc_called / cc_sold * 100) if cc_sold > 0 else 0,
        'weeks_in_cc_mode': weeks_in_cc,
        'final_holding': holding
    }
    return stats, trade_log

def main():
    ensure_dir(PARAMS['output_dir'])
    print("Loading data...")
    df = load_and_clean_data(PARAMS['data_path'])
    print("Running analysis...")
    all_stats = []
    for ticker in PARAMS['tickers']:
        if ticker not in df.columns:
            continue
        print(f"Processing {ticker}...")
        tdf = prepare_ticker_data(df, ticker)
        stats, trade_log = simulate_wheel(tdf, ticker)
        all_stats.append(stats)
        trade_log.to_csv(os.path.join(PARAMS['output_dir'], f"{ticker}_trade_log.csv"), index=False)
        print(f"  CSP sold: {stats['csp_sold']} | Assignment rate: {stats['assignment_rate']:.1f}%")
        print(f"  CC sold: {stats['cc_sold']} | Called away rate: {stats['cc_call_rate']:.1f}%")
    summary = pd.DataFrame(all_stats)
    print("\nSUMMARY:")
    print(summary)
    summary.to_csv(os.path.join(PARAMS['output_dir'], 'wheel_summary.csv'), index=False)
    print(f"\nDone! Charts and logs saved in: {PARAMS['output_dir']}")

if __name__ == "__main__":
    main()