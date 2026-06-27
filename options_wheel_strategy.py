#!/usr/bin/env python3
"""
Options Wheel Strategy Backtester & Visualizer for Triple Leveraged ETFs
========================================================================

This script demonstrates the user's options wheel strategy on historical data
for SOXL, TNA, FAS, UPRO, NAIL.

Strategy Rules (as specified):
- SELL WEEKLY CASH SECURED PUT (CSP) only if BOTH:
  1. Prior month close > 10% above the 10-month SMA (monthly closes)
  2. Weekly momentum > 0: (prior week close - close 13 weeks prior to that) > 0
     i.e., weekly_close[t-1] - weekly_close[t-14] > 0   (using completed weeks)
- Position size: 1 contract (100 shares)
- CSP strike: floor(prior_month_close * 0.99)   # realistic options chain
- Entry filter (premium): >= 1% ROI on capital at risk (strike * 100)
  NOTE: Real option premiums/IV not available in price-only data.
        The script uses the structural signals (trend + momentum) for illustration.
        You can later add a premium estimator (e.g. via historical vol + BS approx).
- If assigned on CSP: Buy 100 shares at the put strike.
- Then sell WEEKLY COVERED CALLS (CC) at strike = assigned_price * 1.05
  (regardless of premium ROI)
- If CC assigned (called away): Shares sold, return to CSP mode when signals allow.
- If CC expires: Repeat CC next week until called away.
- Goal: Visualize signals + simulate wheel mechanics + report assignment stats.

Outputs:
- Interactive Plotly HTML charts per ticker (price + signals + phases)
- Summary statistics (CSP sold, assignment rate, CC sold, called away rate, etc.)
- CSV logs of trades per ticker (optional)

Run:
    python options_wheel_strategy.py --tickers SOXL TNA --save-plots

Dependencies: pandas, numpy, plotly (matplotlib optional fallback)
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
PARAMS = {
    'data_path': 'Stonks - Grok (2).csv',
    'tickers': ['SOXL', 'TNA', 'FAS', 'UPRO', 'NAIL'],
    'sma_months': 10,
    'trend_threshold': 1.10,      # prior month close must be this x SMA
    'momentum_weeks_lookback': 14, # prior week vs 13 weeks earlier (t-1 vs t-14)
    'put_strike_discount': 0.99,  # 99% of prior month close, then floor
    'cc_strike_premium': 1.05,    # 5% OTM for covered calls
    'min_premium_roi': 0.01,      # 1% ROI filter (placeholder - see note above)
    'output_dir': './plots',
    'start_date': None,           # e.g. '2015-01-01' or None for all
}

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_and_clean_data(path):
    """Load CSV, strip column spaces, parse dates, handle early 0s as NaN."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.set_index('Date').sort_index()
    
    for col in PARAMS['tickers']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Early 0s mean ticker didn't exist yet
        df[col] = df[col].replace(0, np.nan)
    
    if PARAMS['start_date']:
        df = df[df.index >= PARAMS['start_date']]
    
    return df

def calculate_monthly_trend_filter(df, ticker):
    """
    Calculate the monthly trend filter.
    - prior_month_close: last close of the previous calendar month
    - 10-month SMA of monthly closes (up to prior month)
    - trend_ok = prior_month_close > 1.10 * sma_10m
    Forward-filled to daily for easy merging.
    """
    monthly = df[[ticker]].resample('M').last().rename(columns={ticker: 'm_close'})
    
    monthly['sma_10m'] = monthly['m_close'].rolling(
        window=PARAMS['sma_months'], min_periods=max(3, PARAMS['sma_months']//2)
    ).mean()
    
    monthly['prior_m_close'] = monthly['m_close'].shift(1)
    monthly['sma_10m_lag'] = monthly['sma_10m'].shift(1)  # SMA known at end of prior month
    
    monthly['trend_ok'] = (
        monthly['prior_m_close'] > (monthly['sma_10m_lag'] * PARAMS['trend_threshold'])
    ).astype(bool)
    
    # Forward fill the regime to all days in the following month
    monthly['trend_ok'] = monthly['trend_ok'].ffill()
    monthly['prior_m_close_ff'] = monthly['prior_m_close'].ffill()
    monthly['sma_10m_ff'] = monthly['sma_10m_lag'].ffill()
    
    # Join back to daily df
    df_out = df.join(
        monthly[['trend_ok', 'prior_m_close_ff', 'sma_10m_ff']], 
        how='left'
    )
    df_out['trend_ok'] = df_out['trend_ok'].ffill().fillna(False)
    df_out['prior_month_close'] = df_out['prior_m_close_ff']
    df_out['sma_10m'] = df_out['sma_10m_ff']
    
    return df_out.drop(columns=['prior_m_close_ff', 'sma_10m_ff'], errors='ignore')

def calculate_weekly_momentum(df, ticker):
    """
    Weekly momentum on Friday closes.
    momentum = weekly_close (most recent completed week) - weekly_close 13 weeks earlier
    mom_ok = momentum > 0
    We lag it by 1 week so decision uses only past completed weeks.
    """
    weekly = df[[ticker]].resample('W-FRI').last().rename(columns={ticker: 'w_close'})
    
    # prior week close minus the close from 13 weeks before the prior week
    # i.e. w_close.shift(1) - w_close.shift(14)
    weekly['mom_raw'] = weekly['w_close'].shift(1) - weekly['w_close'].shift(PARAMS['momentum_weeks_lookback'])
    weekly['mom_ok'] = (weekly['mom_raw'] > 0).astype(bool)
    
    # Lag the mom_ok so we don't use the just-completed week's data for the decision
    # (conservative). Comment out .shift(1) if you want to include current week close.
    weekly['mom_ok_lagged'] = weekly['mom_ok'].shift(1)
    
    # Reindex to daily and ffill (the weekly signal applies for the following week)
    daily_mom = weekly[['mom_ok_lagged', 'mom_raw']].reindex(df.index, method='ffill')
    daily_mom = daily_mom.rename(columns={'mom_ok_lagged': 'mom_ok', 'mom_raw': 'momentum'})
    
    return daily_mom

def prepare_ticker_data(df, ticker):
    """Full pipeline for one ticker: trend + momentum + CSP signal."""
    d = df[[ticker]].copy()
    d = calculate_monthly_trend_filter(d, ticker)
    mom_df = calculate_weekly_momentum(df, ticker)
    d = d.join(mom_df, how='left')
    
    d['csp_signal'] = (d['trend_ok'] & d['mom_ok']).astype(bool)
    
    # Also compute a 200-day SMA for plotting (common daily proxy for 10M)
    d['sma_200d'] = d[ticker].rolling(window=200, min_periods=50).mean()
    
    # Weekly low/high/close for assignment simulation
    weekly_agg = df[[ticker]].resample('W-FRI').agg(['min', 'max', 'last'])
    weekly_agg.columns = ['_'.join(col).strip() for col in weekly_agg.columns.values]
    wcol = f'{ticker}_min'
    # Rename for convenience
    weekly_agg = weekly_agg.rename(columns={
        f'{ticker}_min': 'w_low',
        f'{ticker}_max': 'w_high',
        f'{ticker}_last': 'w_close'
    })
    d = d.join(weekly_agg[['w_low', 'w_high', 'w_close']], how='left')
    d[['w_low', 'w_high', 'w_close']] = d[['w_low', 'w_high', 'w_close']].ffill()
    
    return d

def create_signal_chart(ticker_df, ticker, save_html=True):
    """Interactive Plotly chart: price, SMAs, CSP signal markers, trend/mom status."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.65, 0.17, 0.18],
        subplot_titles=(f"{ticker} - Price & CSP Signal Zones", 
                        "Trend Filter (Monthly)", "Momentum (Weekly)")
    )
    
    # Main price trace
    fig.add_trace(
        go.Scatter(
            x=ticker_df.index,
            y=ticker_df[ticker],
            mode='lines',
            name='Close',
            line=dict(color='#1f77b4', width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 200-day SMA
    fig.add_trace(
        go.Scatter(
            x=ticker_df.index,
            y=ticker_df['sma_200d'],
            mode='lines',
            name='200d SMA (proxy)',
            line=dict(color='orange', width=1, dash='dash'),
            hovertemplate='200d SMA: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Monthly SMA (10m) - only where defined
    valid_sma = ticker_df['sma_10m'].notna()
    fig.add_trace(
        go.Scatter(
            x=ticker_df.index[valid_sma],
            y=ticker_df.loc[valid_sma, 'sma_10m'],
            mode='lines',
            name='10-Month SMA (monthly)',
            line=dict(color='red', width=1.5, dash='dot'),
            hovertemplate='10M SMA: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # CSP Signal markers (where both conditions true)
    signal_mask = ticker_df['csp_signal']
    if signal_mask.any():
        fig.add_trace(
            go.Scatter(
                x=ticker_df.index[signal_mask],
                y=ticker_df.loc[signal_mask, ticker],
                mode='markers',
                name='CSP SELL Signal',
                marker=dict(color='limegreen', size=8, symbol='triangle-up', line=dict(width=1, color='darkgreen')),
                hovertemplate='%{x|%Y-%m-%d}<br>CSP Signal ON<br>Close: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Shade background where trend_ok (light blue)
    trend_changes = ticker_df['trend_ok'].astype(int).diff().fillna(0)
    starts = ticker_df.index[trend_changes == 1]
    ends = ticker_df.index[trend_changes == -1]
    if len(starts) > 0 and (len(ends) == 0 or ends[-1] < starts[-1]):
        ends = list(ends) + [ticker_df.index[-1]]
    
    for s, e in zip(starts, ends):
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(0, 100, 255, 0.08)",
            layer="below",
            line_width=0,
            row=1, col=1
        )
    
    # Row 2: Trend filter status
    fig.add_trace(
        go.Scatter(
            x=ticker_df.index,
            y=ticker_df['trend_ok'].astype(int),
            mode='lines',
            name='Trend OK (Monthly)',
            line=dict(color='#2ca02c', width=2),
            fill='tozeroy',
            fillcolor='rgba(44, 160, 44, 0.3)',
            hovertemplate='Trend OK: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Row 3: Momentum status
    mom_plot = ticker_df['mom_ok'].fillna(False).astype(int)
    fig.add_trace(
        go.Scatter(
            x=ticker_df.index,
            y=mom_plot,
            mode='lines',
            name='Momentum OK (Weekly)',
            line=dict(color='#d62728', width=2),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.3)',
            hovertemplate='Mom OK: %{y}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"Options Wheel Strategy Signals - {ticker}<br>"
              f"<sup>Green triangles = CSP sell opportunity (Trend + Momentum OK). Blue shading = Trend regime active.</sup>",
        height=900,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=80),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trend (0/1)", row=2, col=1, tickvals=[0,1])
    fig.update_yaxes(title_text="Momentum (0/1)", row=3, col=1, tickvals=[0,1])
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    if save_html:
        ensure_dir(PARAMS['output_dir'])
        out_path = os.path.join(PARAMS['output_dir'], f"{ticker}_wheel_signals.html")
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"  Saved interactive chart: {out_path}")
    
    return fig

def simulate_wheel(ticker_df, ticker):
    """
    Event-driven wheel simulator.
    Iterates weekly. When CSP signal present and not holding shares:
      - "Sells" CSP at strike = floor(prior_month_close * 0.99)
      - Checks if assigned this week using w_low or w_close < strike
    If assigned -> switch to CC mode, sell CCs at +5% until called away (w_high or w_close > cc_strike)
    Tracks stats only (no real $ P/L because premiums unknown).
    Returns: stats dict + trade log DataFrame
    """
    # Get weekly rows only (cleaner loop)
    weekly = ticker_df[['w_low', 'w_high', 'w_close', 'prior_month_close', 'csp_signal']].dropna(subset=['w_close'])
    weekly = weekly.resample('W-FRI').last()  # already weekly indexed mostly
    
    trades = []
    holding_shares = False
    shares_cost_basis = None   # the put strike we were assigned at
    cc_strike = None
    csp_sold_count = 0
    assignments = 0
    cc_sold_count = 0
    cc_called_away = 0
    weeks_in_cc = 0
    
    current_prior_month_close = None
    
    for week_end, row in weekly.iterrows():
        signal = bool(row['csp_signal'])
        w_low = row['w_low']
        w_high = row['w_high']
        w_close = row['w_close']
        prior_m = row['prior_month_close']
        
        if pd.isna(prior_m) or pd.isna(w_close):
            continue
        
        current_prior_month_close = prior_m
        
        if not holding_shares:
            # CSP mode
            if signal:
                # Sell CSP
                strike = np.floor(prior_m * PARAMS['put_strike_discount'])
                csp_sold_count += 1
                
                # Simulate assignment: conservative - if low of week < strike OR close < strike
                assigned = (w_low < strike) or (w_close < strike)
                
                trade = {
                    'week_end': week_end,
                    'action': 'SELL_CSP',
                    'strike': strike,
                    'prior_m_close': prior_m,
                    'assigned': assigned,
                    'w_low': w_low,
                    'w_close': w_close
                }
                trades.append(trade)
                
                if assigned:
                    assignments += 1
                    holding_shares = True
                    shares_cost_basis = strike
                    cc_strike = np.floor(shares_cost_basis * PARAMS['cc_strike_premium'])
                    # print(f"  {week_end.date()}: CSP ASSIGNED @ ${strike:.0f} -> now CC mode @ ${cc_strike:.0f}")
        else:
            # CC mode - sell covered call this week
            weeks_in_cc += 1
            cc_strike = np.floor(shares_cost_basis * PARAMS['cc_strike_premium'])  # recalc in case, but fixed
            
            cc_sold_count += 1
            
            # Called away if high or close exceeds CC strike
            called = (w_high > cc_strike) or (w_close > cc_strike)
            
            trade = {
                'week_end': week_end,
                'action': 'SELL_CC',
                'strike': cc_strike,
                'cost_basis': shares_cost_basis,
                'called_away': called,
                'w_high': w_high,
                'w_close': w_close
            }
            trades.append(trade)
            
            if called:
                cc_called_away += 1
                holding_shares = False
                shares_cost_basis = None
                cc_strike = None
                # print(f"  {week_end.date()}: CC CALLED AWAY @ ${cc_strike:.0f} -> back to CSP mode")
    
    trade_log = pd.DataFrame(trades)
    
    stats = {
        'ticker': ticker,
        'csp_sold': csp_sold_count,
        'assignments': assignments,
        'assignment_rate': (assignments / csp_sold_count * 100) if csp_sold_count > 0 else 0,
        'cc_sold': cc_sold_count,
        'cc_called_away': cc_called_away,
        'cc_call_rate': (cc_called_away / cc_sold_count * 100) if cc_sold_count > 0 else 0,
        'weeks_in_cc_mode': weeks_in_cc,
        'final_state_holding': holding_shares
    }
    
    return stats, trade_log

def create_phase_chart(ticker_df, trade_log, ticker, stats, save_html=True):
    """Chart showing price + background colored by wheel phase (CSP eligible vs in CC)."""
    fig = go.Figure()
    
    # Price
    fig.add_trace(go.Scatter(
        x=ticker_df.index, y=ticker_df[ticker],
        mode='lines', name='Close', line=dict(color='#1f77b4', width=1.2),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>'
    ))
    
    # Color background for phases using trade_log
    # Simple approach: reconstruct holding periods from trade_log
    in_cc_periods = []
    if not trade_log.empty:
        cc_trades = trade_log[trade_log['action'] == 'SELL_CC']
        csp_trades = trade_log[trade_log['action'] == 'SELL_CSP']
        
        # Find periods between assignment (CSP assigned=True) and next called away
        assigned_weeks = csp_trades[csp_trades['assigned'] == True]['week_end'].tolist() if not csp_trades.empty else []
        called_weeks = cc_trades[cc_trades['called_away'] == True]['week_end'].tolist() if not cc_trades.empty else []
        
        # Pair them (simplified - assumes proper alternation)
        for i, assign_date in enumerate(assigned_weeks):
            if i < len(called_weeks):
                end_date = called_weeks[i]
            else:
                end_date = ticker_df.index[-1]
            in_cc_periods.append((assign_date, end_date))
    
    for start, end in in_cc_periods:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255, 165, 0, 0.15)",  # orange for CC mode
            layer="below", line_width=0
        )
    
    # CSP signals as markers
    signals = ticker_df[ticker_df['csp_signal']]
    if not signals.empty:
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals[ticker],
            mode='markers', name='CSP Signal',
            marker=dict(color='limegreen', size=7, symbol='triangle-up'),
            hovertemplate='CSP Signal<br>%{x|%Y-%m-%d}<extra></extra>'
        ))
    
    title = (f"{ticker} Wheel Simulation - Phase View<br>"
             f"<sup>CSP sold: {stats['csp_sold']} | Assigned: {stats['assignments']} ({stats['assignment_rate']:.1f}%) | "
             f"CC sold: {stats['cc_sold']} | Called away: {stats['cc_called_away']} ({stats['cc_call_rate']:.1f}%) | "
             f"Weeks in CC: {stats['weeks_in_cc_mode']}</sup>")
    
    fig.update_layout(
        title=title,
        height=600,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02),
        annotations=[dict(
            text="Orange shading = periods holding shares (CC mode). Green triangles = weeks CSP signal was active.",
            x=0.5, y=-0.12, xref='paper', yref='paper', showarrow=False, font=dict(size=10)
        )]
    )
    fig.update_yaxes(title="Price ($)")
    fig.update_xaxes(title="Date")
    
    if save_html:
        ensure_dir(PARAMS['output_dir'])
        out_path = os.path.join(PARAMS['output_dir'], f"{ticker}_wheel_phases.html")
        fig.write_html(out_path, include_plotlyjs='cdn')
        print(f"  Saved phase chart: {out_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Options Wheel Strategy Visualizer & Simulator")
    parser.add_argument('--tickers', nargs='+', default=PARAMS['tickers'],
                        help='Tickers to process (default: all)')
    parser.add_argument('--no-plots', action='store_true', help='Skip saving HTML plots')
    parser.add_argument('--output-dir', default=PARAMS['output_dir'], help='Where to save plots')
    args = parser.parse_args()
    
    PARAMS['output_dir'] = args.output_dir
    ensure_dir(PARAMS['output_dir'])
    
    print("=" * 70)
    print("OPTIONS WHEEL STRATEGY - BACKTEST & VISUALIZATION")
    print("=" * 70)
    print(f"Data: {PARAMS['data_path']}")
    print(f"Tickers: {args.tickers}")
    print(f"Trend: Prior month close > {PARAMS['trend_threshold']*100:.0f}% of 10M SMA")
    print(f"Momentum: Weekly close (t-1) - close (t-14) > 0")
    print(f"CSP Strike: floor(prior_m_close * {PARAMS['put_strike_discount']})")
    print(f"CC Strike: floor(assigned * {PARAMS['cc_strike_premium']})")
    print(f"Note: Premium >=1% ROI filter is structural only (no IV data).")
    print("-" * 70)
    
    df = load_and_clean_data(PARAMS['data_path'])
    
    all_stats = []
    
    for ticker in args.tickers:
        if ticker not in df.columns:
            print(f"Skipping unknown ticker: {ticker}")
            continue
        
        print(f"\n>>> Processing {ticker} ...")
        
        tdf = prepare_ticker_data(df, ticker)
        
        # 1. Signal visualization chart
        if not args.no_plots:
            create_signal_chart(tdf, ticker, save_html=True)
        
        # 2. Wheel simulation
        stats, trade_log = simulate_wheel(tdf, ticker)
        all_stats.append(stats)
        
        print(f"    CSP sold: {stats['csp_sold']}")
        print(f"    Assignments: {stats['assignments']}  (rate: {stats['assignment_rate']:.1f}%)")
        print(f"    CC sold: {stats['cc_sold']}")
        print(f"    CC called away: {stats['cc_called_away']}  (rate: {stats['cc_call_rate']:.1f}%)")
        print(f"    Weeks spent in CC mode: {stats['weeks_in_cc_mode']}")
        
        if not args.no_plots and not trade_log.empty:
            create_phase_chart(tdf, trade_log, ticker, stats, save_html=True)
        
        # Optional: save trade log
        if not trade_log.empty:
            log_path = os.path.join(PARAMS['output_dir'], f"{ticker}_trade_log.csv")
            trade_log.to_csv(log_path, index=False)
            print(f"    Trade log saved: {log_path}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY - ALL TICKERS")
    print("=" * 70)
    summary_df = pd.DataFrame(all_stats)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(PARAMS['output_dir'], 'wheel_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\nDone! Open the .html files in a browser for interactive charts.")
    print("You can extend this script with:")
    print("  - Real premium estimation (historical vol + approx Black-Scholes or delta)")
    print("  - Portfolio-level capital allocation & P/L (with assumed avg premium %)")
    print("  - Optimization of parameters (thresholds, strike %)")
    print("  - Integration with yfinance or polygon for live options data")

if __name__ == "__main__":
    main()