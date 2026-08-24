import sys
import os
sys.path.append(os.path.abspath('.'))

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import direct pour la sensitivity analysis (évite 300 appels API)
try:
    from src.options_pricer import BlackScholesPricer
    LOCAL_PRICER = True
except ImportError:
    LOCAL_PRICER = False

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="ChipDiffusion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ ChipDiffusion")
st.sidebar.caption("Vol Arb · Semiconductor Supply Chain · NVDA")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📐 BSM Pricer", "📈 Volatility Surface",
     "🤖 LSTM+HMM Signal", "📊 Backtest", "🎯 VRP Monitor"]
)

def call_api(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ API unavailable : run: `uvicorn api.main:app --reload --port 8000`"
    except requests.exceptions.HTTPError as e:
        return None, f"❌ API error {e.response.status_code}: {e.response.json().get('detail', str(e))}"
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"

def local_price(S0, K, T, r, sigma, option_type):
    """Use local pricer if available : avoids API round-trip for bulk computation."""
    if not LOCAL_PRICER:
        return None
    p = BlackScholesPricer(S0, K, T, r, sigma)
    price = p.call_price() if option_type == 'call' else p.put_price()
    return {
        'price': round(price, 4),
        'delta': round(p.delta(option_type), 4),
        'gamma': round(p.gamma(), 6),
        'vega': round(p.vega(), 4),
        'theta': round(p.theta(option_type), 4),
    }

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("⚡ ChipDiffusion")
    st.markdown("**Volatility Arbitrage Engine · Semiconductor Supply Chain · NVDA**")
    st.divider()

    health, err = call_api("/health")
    if err:
        st.error(err)
        st.stop()
    st.success("✅ API operational")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("API Endpoints", "7")
    with col2:
        st.metric("Pricing Models", "BSM + Heston")
    with col3:
        st.metric("Signal", "LSTM + HMM")
    with col4:
        st.metric("Underlying", "NVDA")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Project Context")
        st.markdown("""
        ChipDiffusion is the continuous-time successor to **ChipChainReaction**
        (discrete-time HMM + LSTM trading system on the semiconductor supply chain).

        The name carries a double meaning: *chip* for the semiconductor supply chain,
        and *diffusion* as a nod to Brownian motion, first encountered
        studying particle diffusion driven by thermal agitation in physics,
        and later rediscovered as the mathematical engine of option pricing.

        The economic motivation comes from *Chip War* (Chris Miller, 2022),
        which traces the geopolitical and industrial architecture of the global
        semiconductor supply chain:
        **Commodities → Foundries (TSM, ASML) → Designers (NVDA, AMD) → Integrators (AAPL, MSFT)**

        ChipDiffusion tests whether these structural dependencies generate
        a measurable and tradeable **Variance Risk Premium** on NVDA.
        """)

    with col_b:
        st.subheader("Mathematical Core")
        st.markdown("""
        The PnL of a delta-hedged short straddle decomposes as:

        $$dPnL \\approx \\frac{1}{2} \\Gamma S^2 (\\sigma_{imp}^2 - \\sigma_{real}^2) \\, dt$$

        If **IV > RV**, each instant $dt$ generates a profit proportional
        to the option's Gamma and the squared volatility spread,
        independently of the direction of the underlying.

        The strategy sells this spread systematically,
        filtered by an LSTM+HMM signal to avoid unfavourable regimes.
        """)

    st.divider()
    st.subheader("Key Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Win Rate", "71.4%", help="Historical backtest on real NVDA prices, 2023–2026")
    with c2:
        st.metric("Sharpe (approx)", "0.95")
    with c3:
        st.metric("vs Baseline", "+27% PnL/trade", delta="21 filtered vs 31 unfiltered")
    with c4:
        st.metric("LSTM Edge", "p=0.006", help="10-seed robustness test, 9/10 seeds positive")
    with c5:
        st.metric("Heston Feller ✓", "0.72 > 0.25", help="2κθ > ξ² - variance stays non-negative")

    st.divider()
    st.subheader("Architecture")
    st.markdown("""
    ```
    Data (yfinance)
        │
        ├── Notebook 01 ── Monte Carlo GBM (engine validation)
        ├── Notebook 02 ── BSM Pricer + Greeks + IV (Newton-Raphson + Brentq)
        ├── Notebook 03 ── NVDA Volatility Surface (empirical skew)
        ├── Notebook 04 ── Vol Arb theory (Straddle + simulated Delta Hedging)
        ├── Notebook 05 ── ML Engine (HMM Walk-Forward + CNN1D/LSTM)
        │                  → Export: data/processed/signal_vol_arb.csv
        ├── Notebook 06 ── Historical backtest (real NVDA prices, 2023–2026)
        ├── Notebook 07 ── Heston Monte Carlo (beyond BSM)
        │
        ├── src/            Reusable Python modules
        ├── api/            FastAPI REST (7 endpoints, 4 routers)
        └── streamlit_app.py  This dashboard
    ```
    """)

# ── BSM PRICER ────────────────────────────────────────────────────────────────
elif page == "📐 BSM Pricer":
    st.title("📐 Black-Scholes Pricer")
    st.caption("Analytical European option pricing + Greeks")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Parameters")
        S0 = st.number_input("Underlying price S₀ ($)", value=275.0, min_value=0.01, step=1.0)
        K  = st.number_input("Strike K ($)", value=280.0, min_value=0.01, step=1.0)
        T  = st.slider("Maturity T (years)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
        sigma = st.slider("Volatility σ (%)", min_value=1, max_value=150, value=35) / 100
        r  = st.slider("Risk-free rate r (%)", min_value=0.0, max_value=10.0, value=4.5) / 100
        option_type = st.radio("Option type", ["call", "put"], horizontal=True)

        moneyness_label = "ATM" if abs(S0 - K) < 5 else (
            "ITM" if (option_type == "call" and S0 > K) or
                     (option_type == "put"  and S0 < K) else "OTM"
        )
        st.caption(f"Moneyness: {moneyness_label} | T = {T*252:.0f} trading days")

    with col2:
        st.subheader("Results")
        # Single pricing: use API (one call is fine)
        data, err = call_api("/pricer/", {
            "S0": S0, "K": K, "T": T, "r": r,
            "sigma": sigma, "option_type": option_type
        })
        if err:
            st.error(err)
        else:
            price = data['price']
            intrinsic = max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)

            cp1, cp2 = st.columns(2)
            with cp1:
                st.metric(f"{option_type.upper()} Price", f"${price:.4f}")
            with cp2:
                st.metric("Intrinsic Value", f"${intrinsic:.4f}",
                          delta=f"Time value: ${price - intrinsic:.4f}")

            st.divider()
            st.subheader("Greeks")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("Δ Delta", f"{data['delta']:.4f}",
                          help="Option price change per $1 move in underlying")
            with g2:
                st.metric("Γ Gamma", f"{data['gamma']:.6f}",
                          help="Delta change per $1 move in underlying")
            with g3:
                st.metric("ν Vega", f"${data['vega']:.4f}",
                          help="P&L change if σ increases by 1%")
            with g4:
                st.metric("Θ Theta", f"${data['theta']:.4f}/day",
                          help="Value lost per calendar day (time decay)")

    st.divider()
    st.subheader("Sensitivity Analysis")

    if not LOCAL_PRICER:
        st.warning(
            "Local pricer not found (`src/options_pricer.py`). "
            "Sensitivity charts are disabled to avoid 300+ API calls."
        )
    else:
        S_range = np.linspace(S0 * 0.70, S0 * 1.30, 100)
        sigma_range = np.linspace(0.05, 1.0, 100)

        tab1, tab2, tab3 = st.tabs(["Price vs S₀", "Price vs σ", "Greeks vs S₀"])

        with tab1:
            prices_s = [local_price(s, K, T, r, sigma, option_type)['price']
                        for s in S_range]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=S_range, y=prices_s, mode='lines',
                                      line=dict(color='#7c3aed', width=2)))
            fig.add_vline(x=S0, line_dash="dash", line_color="gray",
                          annotation_text=f"S₀={S0:.0f}")
            fig.add_vline(x=K, line_dash="dot", line_color="red",
                          annotation_text=f"K={K:.0f}")
            fig.update_layout(
                title=f"{option_type.upper()} price vs underlying price",
                xaxis_title="Price ($)", yaxis_title="Option price ($)",
                height=350, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            prices_sig = [local_price(S0, K, T, r, s, option_type)['price']
                          for s in sigma_range]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=sigma_range * 100, y=prices_sig, mode='lines',
                                       line=dict(color='#06b6d4', width=2)))
            fig2.add_vline(x=sigma * 100, line_dash="dash", line_color="gray",
                           annotation_text=f"σ={sigma*100:.0f}%")
            fig2.update_layout(
                title=f"{option_type.upper()} price vs volatility",
                xaxis_title="Volatility (%)", yaxis_title="Option price ($)",
                height=350, template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            results_s = [local_price(s, K, T, r, sigma, option_type) for s in S_range]
            deltas = [d['delta'] for d in results_s]
            gammas = [d['gamma'] for d in results_s]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=S_range, y=deltas, name='Delta',
                                       line=dict(color='#7c3aed')))
            fig3.add_trace(go.Scatter(x=S_range, y=[g * 100 for g in gammas],
                                       name='Gamma ×100', line=dict(color='#06b6d4')))
            fig3.add_vline(x=S0, line_dash="dash", line_color="gray")
            fig3.add_vline(x=K, line_dash="dot", line_color="red")
            fig3.update_layout(
                title="Delta and Gamma vs underlying price",
                xaxis_title="Price ($)", height=350, template="plotly_dark"
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("🔍 Implied Volatility Calculator")
    col_iv1, col_iv2 = st.columns([1, 1])
    with col_iv1:
        market_price = st.number_input("Observed market price ($)", value=15.0, min_value=0.01)
        if st.button("Calculate IV", type="primary"):
            iv_data, iv_err = call_api("/pricer/implied_vol", {
                "market_price": market_price, "S0": S0, "K": K,
                "T": T, "r": r, "option_type": option_type
            })
            if iv_err:
                st.error(iv_err)
            else:
                with col_iv2:
                    st.metric("Implied Volatility",
                              f"{iv_data['implied_volatility_pct']:.2f}%")
                    st.caption(
                        f"To justify a market price of ${market_price:.2f} on this "
                        f"{option_type}, the market is pricing in a volatility of "
                        f"{iv_data['implied_volatility_pct']:.2f}%."
                    )

# ── VOLATILITY SURFACE ────────────────────────────────────────────────────────
elif page == "📈 Volatility Surface":
    st.title("📈 Implied Volatility Surface")
    st.caption("Volatility smile and skew : live yfinance data")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="NVDA").upper()
    with col2:
        n_exp = st.slider("Number of expirations", 1, 5, 3)
    with col3:
        r_vol = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.5) / 100

    if st.button("📡 Fetch volatility surface", type="primary"):
        with st.spinner(f"Fetching {ticker} options chain..."):
            data, err = call_api("/vol_surface/", {
                "ticker": ticker, "r": r_vol,
                "n_expirations": n_exp,
                "moneyness_min": 0.80, "moneyness_max": 1.20
            })

        if err:
            st.error(err)
        else:
            st.success(f"✅ {data['n_contracts']} liquid contracts | {ticker} @ ${data['S0']}")

            df_vol = pd.DataFrame(data['data'])
            colors = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

            fig = go.Figure()
            for i, t_days in enumerate(sorted(df_vol['T_days'].unique())):
                subset = df_vol[df_vol['T_days'] == t_days].sort_values('moneyness')
                fig.add_trace(go.Scatter(
                    x=subset['moneyness'], y=subset['implied_volatility_pct'],
                    mode='lines+markers', name=f"T = {t_days}d",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5)
                ))
            fig.add_vline(x=1.0, line_dash="dash", line_color="white",
                          annotation_text="ATM")
            fig.update_layout(
                title=f"Implied Volatility Smile : {ticker}",
                xaxis_title="Moneyness (K/S₀)", yaxis_title="Implied Volatility (%)",
                template="plotly_dark", height=450
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Raw data")
            st.dataframe(
                df_vol[['K', 'T_days', 'moneyness', 'mid_price',
                         'implied_volatility_pct', 'yahoo_iv_pct']].round(2),
                use_container_width=True
            )

# ── SIGNAL ────────────────────────────────────────────────────────────────────
elif page == "🤖 LSTM+HMM Signal":
    st.title("🤖 LSTM+HMM Signal")
    st.caption("Volatility oracle : last known signal state")
    st.divider()

    data, err = call_api("/signal/")
    if err:
        st.error(err)
    else:
        action = data['action']
        label  = "SELL VOL" if action == "VENDRE_VOL" else "NEUTRAL"
        color  = "🟢" if action == "VENDRE_VOL" else "🟡"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", f"{color} {label}")
        with col2:
            st.metric("Date", data['last_date'])
        with col3:
            st.metric("HMM Regime", data['hmm_regime'])

        if action == "VENDRE_VOL":
            st.success(
                "**FAVOURABLE** : Predicted future RV below current RV, non-Bear regime. "
                "Conditions suggest selling volatility (Short Straddle)."
            )
        else:
            st.info(
                "**NEUTRAL** : LSTM signal or HMM regime unfavourable. "
                "No position recommended."
            )
        st.warning(data['warning'])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Current RV (21d)", f"{data['rv_current_21d_pct']:.1f}%")
        with col_b:
            st.metric("Predicted ΔRV", f"{data['delta_rv_predicted']:+.1f}%")
        with col_c:
            st.metric("Predicted Future RV", f"{data['rv_future_predicted_pct']:.1f}%",
                      delta=f"{data['delta_rv_predicted']:+.1f}%")

        st.divider()
        st.subheader("Signal History (last 60 days)")
        hist_data, hist_err = call_api("/signal/history", {"last_n": 60})
        if not hist_err:
            df_hist = pd.DataFrame(hist_data['history'])
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['rv_current_pct'],
                mode='lines', name='Current RV (%)',
                line=dict(color='#7c3aed', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['delta_rv_predicted_pct'],
                mode='lines', name='Predicted ΔRV (%)',
                line=dict(color='#06b6d4', width=1.5)
            ))
            for d in df_hist[df_hist['signal_combined'] == 1]['date']:
                fig.add_vrect(x0=d, x1=d + pd.Timedelta(days=1),
                              fillcolor="green", opacity=0.1, line_width=0)
            fig.update_layout(
                title="LSTM+HMM Signal : 60-day history (green = signal active)",
                xaxis_title="Date", yaxis_title="%",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Signal active on {hist_data['pct_signal_active']:.1f}% of days "
                f"over the last {hist_data['n_days']} days."
            )

# ── BACKTEST ──────────────────────────────────────────────────────────────────
elif page == "📊 Backtest":
    st.title("📊 Backtest : Short Straddle Delta-Hedged")
    st.caption("Historical results on NVDA · 2023–2026 · LSTM+HMM Signal")
    st.divider()

    data, err = call_api("/backtest/summary")
    if err:
        st.error(err)
    else:
        perf     = data['performance']
        baseline = data['vs_baseline']

        st.subheader("Performance : LSTM+HMM Signal")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Trades", perf['n_trades'])
        with c2:
            st.metric("Net PnL", f"+${perf['pnl_total_net_usd']}")
        with c3:
            st.metric("Win Rate", f"{perf['win_rate_pct']}%")
        with c4:
            pf = perf['profit_factor']
            st.metric("Profit Factor", pf,
                      delta="⚠ below 1.0" if pf < 1 else "✓ above 1.0")
        with c5:
            st.metric("Sharpe (approx)", perf['sharpe_approx'])

        st.divider()
        st.subheader("vs Systematic Baseline")
        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            st.dataframe(pd.DataFrame({
                "Metric": ["Trades", "Net PnL ($)", "PnL/Trade ($)", "Win Rate (%)"],
                "LSTM+HMM": [perf['n_trades'], perf['pnl_total_net_usd'],
                              perf['pnl_avg_per_trade_usd'], perf['win_rate_pct']],
                "Baseline": [baseline['baseline_n_trades'], baseline['baseline_pnl_total_usd'],
                              baseline['baseline_pnl_avg_per_trade_usd'],
                              baseline['baseline_win_rate_pct']]
            }), use_container_width=True, hide_index=True)
        with col_cmp2:
            st.metric("PnL/Trade improvement",
                      f"+{baseline['improvement_pnl_per_trade_pct']:.1f}%")
            st.metric("Trade count",
                      f"{perf['n_trades']} vs {baseline['baseline_n_trades']}",
                      delta=f"-{baseline['baseline_n_trades'] - perf['n_trades']} filtered")

        st.divider()
        st.subheader("Individual Trades")
        trades_data, trades_err = call_api("/backtest/trades")
        if trades_err:
            st.warning(trades_err)
            st.caption(
                "To activate: `df_trades.to_csv('../data/processed/backtest_trades.csv')` "
                "at the end of Notebook 06."
            )
        else:
            df_t = pd.DataFrame(trades_data['trades'])
            df_t['cumulative_pnl'] = df_t['pnl_net_usd'].cumsum()
            df_t['entry_date']     = pd.to_datetime(df_t['entry_date'])

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['cumulative_pnl'],
                mode='lines+markers', name='Cumulative PnL',
                line=dict(color='#7c3aed', width=2),
                marker=dict(
                    color=['green' if p > 0 else 'red' for p in df_t['pnl_net_usd']],
                    size=8
                )
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pnl.update_layout(
                title="Cumulative PnL : Individual Trades",
                xaxis_title="Entry Date", yaxis_title="PnL ($)",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            fig_vrp = go.Figure()
            fig_vrp.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['iv_used_pct'],
                name='Estimated IV (%)', line=dict(color='orange', dash='dash')
            ))
            fig_vrp.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['rv_realized_pct'],
                name='Realised RV (%)', line=dict(color='white')
            ))
            fig_vrp.update_layout(
                title="Estimated IV vs Realised RV per Trade",
                xaxis_title="Date", yaxis_title="Volatility (%)",
                template="plotly_dark", height=300
            )
            st.plotly_chart(fig_vrp, use_container_width=True)

            st.dataframe(
                df_t[['entry_date', 'exit_date', 'S0', 'K',
                       'iv_used_pct', 'rv_realized_pct', 'vrp_pct',
                       'pnl_net_usd', 'result']].rename(columns={
                    'entry_date': 'Entry', 'exit_date': 'Exit',
                    'iv_used_pct': 'IV (%)', 'rv_realized_pct': 'RV (%)',
                    'vrp_pct': 'VRP (%)', 'pnl_net_usd': 'PnL ($)',
                    'result': 'Result'
                }),
                use_container_width=True, hide_index=True
            )

        st.divider()
        st.subheader("⚠ Known Limitations")
        for lim in data['known_limitations']:
            st.markdown(f"- {lim}")

# ── VRP MONITOR ───────────────────────────────────────────────────────────────
elif page == "🎯 VRP Monitor":
    st.title("🎯 Live VRP Monitor")
    st.caption(
        "**Variance Risk Premium = Implied Volatility − Realised Volatility.** "
        "A positive VRP means the market pays more for vol than it realises : "
        "the structural edge of the Short Straddle strategy."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker_vrp  = st.text_input("Ticker", value="NVDA", key="vrp_ticker").upper()
    with col2:
        r_vrp       = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.5, key="vrp_r") / 100
    with col3:
        target_days = st.slider("Target expiration (days)", 7, 90, 30, key="vrp_days")

    if st.button("📡 Compute VRP", type="primary"):
        with st.spinner(f"Fetching {ticker_vrp} options and computing VRP..."):
            surf_data,   surf_err   = call_api("/vol_surface/", {
                "ticker": ticker_vrp, "r": r_vrp,
                "n_expirations": 3,
                "moneyness_min": 0.95, "moneyness_max": 1.05
            })
            signal_data, signal_err = call_api("/signal/")

        if surf_err:
            st.error(surf_err)
        elif signal_err:
            st.error(signal_err)
        else:
            df_surf = pd.DataFrame(surf_data['data'])
            if df_surf.empty:
                st.error("No ATM contracts found in [0.95, 1.05] moneyness window.")
                st.stop()

            df_surf['dist_atm']    = abs(df_surf['moneyness'] - 1.0)
            df_surf['dist_target'] = abs(df_surf['T_days'] - target_days)
            best_row = df_surf.sort_values(['dist_target', 'dist_atm']).iloc[0]

            current_iv = best_row['implied_volatility_pct']
            current_rv = signal_data.get('rv_current_21d_pct', 0.0)

            # VRP via calculate_vrp equivalent (IV - RV)
            vrp = current_iv - current_rv

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Implied Vol (ATM)",
                          f"{current_iv:.1f}%",
                          help=f"Moneyness {best_row['moneyness']:.3f}, T={best_row['T_days']}d")
            with col_b:
                st.metric("Realised Vol (21d)", f"{current_rv:.1f}%",
                          help="From LSTM+HMM signal export (last known state)")
            with col_c:
                st.metric("VRP = IV − RV", f"{vrp:+.1f}%",
                          delta="Favourable ✓" if vrp > 0 else "Unfavourable ✗",
                          delta_color="normal" if vrp > 0 else "inverse")

            st.divider()
            if vrp > 0:
                st.success(
                    f"**Positive VRP (+{vrp:.1f}%)** : The market prices in {current_iv:.1f}% vol "
                    f"but only {current_rv:.1f}% has realised over the past 21 days. "
                    f"Conditions are structurally favourable for a Short Straddle."
                )
            else:
                st.error(
                    f"**Negative VRP ({vrp:.1f}%)** : Realised vol ({current_rv:.1f}%) exceeds "
                    f"implied vol ({current_iv:.1f}%). Selling volatility here would result in a loss."
                )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=vrp,
                delta={'reference': 0,
                       'increasing': {'color': "green"},
                       'decreasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [-30, 30]},
                    'bar': {'color': "green" if vrp > 0 else "red"},
                    'steps': [
                        {'range': [-30, 0], 'color': '#2d1b1b'},
                        {'range': [0,  30], 'color': '#1b2d1b'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75, 'value': 0
                    }
                },
                title={'text': f"VRP : {ticker_vrp} "
                               f"(ATM IV, T≈{best_row['T_days']}d)"}
            ))
            fig_gauge.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()
            st.subheader("ATM contracts used")
            st.dataframe(
                df_surf.sort_values('dist_atm').head(5)[
                    ['K', 'T_days', 'moneyness', 'mid_price',
                     'implied_volatility_pct', 'yahoo_iv_pct']
                ].round(2),
                use_container_width=True
            )
            st.caption(
                "IV is computed from live yfinance data. "
                "RV (21d) is the last value exported by Notebook 05 : "
                "re-run that notebook to refresh the estimate. "
                "VRP computed via `BlackScholesPricer.calculate_vrp(iv, rv)` "
                "from `src/options_pricer.py`."
            )