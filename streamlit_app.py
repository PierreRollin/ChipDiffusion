import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="ChipDiffusion",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS minimal
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 3px solid #7c3aed;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/chip.png", width=60)      # oui c'est un jeton et pas une puce électronique, mais c'est bien le but de l'icône
st.sidebar.title("ChipDiffusion")
st.sidebar.caption("Vol Arb · Supply Chain Semis · NVDA")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📐 Pricer BSM", "📈 Surface de Volatilité", "🤖 Signal LSTM+HMM", "📊 Backtest"]
)

def call_api(endpoint: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ API non disponible — lance `uvicorn api.main:app --reload --port 8000`"
    except requests.exceptions.HTTPError as e:
        return None, f"❌ Erreur API {e.response.status_code} : {e.response.json().get('detail', str(e))}"
    except Exception as e:
        return None, f"❌ Erreur inattendue : {str(e)}"

# ── PAGE : DASHBOARD ─────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("⚡ ChipDiffusion")
    st.markdown("**Volatility Arbitrage · Semi-Conductor Supply Chain · NVDA**")
    st.divider()

    health, err = call_api("/health")
    if err:
        st.error(err)
        st.stop()

    st.success("✅ API opérationnelle")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Routes disponibles", "6")
    with col2:
        st.metric("Modèle de pricing", "BSM + Heston")
    with col3:
        st.metric("Signal", "LSTM + HMM")
    with col4:
        st.metric("Actif principal", "NVDA")

    st.divider()
    st.subheader("Architecture du projet")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Modules**
        - `Notebook 01-02` — Monte Carlo GBM + BSM
        - `Notebook 03` — Surface de Volatilité Implicite
        - `Notebook 04` — HMM Régimes de Marché
        - `Notebook 05` — LSTM Oracle de Volatilité
        - `Notebook 06` — Backtest Delta Hedging Historique
        - `Notebook 07` — Modèle de Heston
        """)
    with col_b:
        st.markdown("""
        **Thèse économique**

        Les chocs sur les matières premières (USO, PICK) et les fonderies
        (TSM, ASML, SMIC) se répercutent avec un décalage temporel sur
        les designers (NVDA, AMD) et les intégrateurs (AAPL, MSFT).

        La Variance Risk Premium (IV − RV) sur NVDA est structurellement
        positive, créant une opportunité de vente de volatilité filtrée
        par signal LSTM+HMM.
        """)

# ── PAGE : PRICER BSM ────────────────────────────────────────────────────
elif page == "📐 Pricer BSM":
    st.title("📐 Pricer Black-Scholes")
    st.caption("Pricing analytique d'options européennes + Greeks")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Paramètres")
        S0 = st.number_input("Prix du sous-jacent S₀ ($)", value=275.0, min_value=0.01, step=1.0)
        K = st.number_input("Strike K ($)", value=280.0, min_value=0.01, step=1.0)
        T = st.slider("Maturité T (années)", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
        sigma = st.slider("Volatilité σ (%)", min_value=1, max_value=150, value=35) / 100
        r = st.slider("Taux sans risque r (%)", min_value=0.0, max_value=10.0, value=4.5) / 100
        option_type = st.radio("Type d'option", ["call", "put"], horizontal=True)

        st.caption(f"Moneyness : {'ATM' if abs(S0-K) < 5 else 'ITM' if (option_type=='call' and S0>K) or (option_type=='put' and S0<K) else 'OTM'} | T = {T*252:.0f} jours ouvrés")

    with col2:
        st.subheader("Résultats")
        data, err = call_api("/pricer/", {"S0": S0, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type})

        if err:
            st.error(err)
        else:
            price = data['price']
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric(f"Prix du {option_type.upper()}", f"{price:.4f} $")
            with col_p2:
                intrinsic = max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
                st.metric("Valeur intrinsèque", f"{intrinsic:.4f} $",
                          delta=f"Time value: {price - intrinsic:.4f} $")

            st.divider()
            st.subheader("Greeks")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("Δ Delta", f"{data['delta']:.4f}",
                          help="Sensibilité au prix : variation du prix de l'option pour +1$ sur le sous-jacent")
            with g2:
                st.metric("Γ Gamma", f"{data['gamma']:.6f}",
                          help="Accélération du Delta : variation du Delta pour +1$ sur le sous-jacent")
            with g3:
                st.metric("ν Vega", f"{data['vega']:.4f} $",
                          help="Sensibilité à la vol : gain si σ augmente de 1%")
            with g4:
                st.metric("Θ Theta", f"{data['theta']:.4f} $/j",
                          help="Décroissance temporelle : perte de valeur par jour")

    st.divider()
    st.subheader("Sensibilités")

    tab1, tab2, tab3 = st.tabs(["Prix vs S₀", "Prix vs σ", "Greeks vs S₀"])

    with tab1:
        S_range = np.linspace(S0 * 0.7, S0 * 1.3, 100)
        prices_range = []
        for s in S_range:
            from src.options_pricer import BlackScholesPricer
            p = BlackScholesPricer(s, K, T, r, sigma)
            prices_range.append(p.call_price() if option_type == 'call' else p.put_price())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=prices_range, mode='lines',
                                  line=dict(color='#7c3aed', width=2)))
        fig.add_vline(x=S0, line_dash="dash", line_color="gray",
                      annotation_text=f"S₀={S0}")
        fig.add_vline(x=K, line_dash="dot", line_color="red",
                      annotation_text=f"K={K}")
        fig.update_layout(title=f"Prix du {option_type} vs Prix du sous-jacent",
                          xaxis_title="Prix ($)", yaxis_title="Prix de l'option ($)",
                          height=350, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        sigma_range = np.linspace(0.05, 1.0, 100)
        prices_sigma = []
        for s in sigma_range:
            from src.options_pricer import BlackScholesPricer
            p = BlackScholesPricer(S0, K, T, r, s)
            prices_sigma.append(p.call_price() if option_type == 'call' else p.put_price())

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sigma_range * 100, y=prices_sigma, mode='lines',
                                   line=dict(color='#06b6d4', width=2)))
        fig2.add_vline(x=sigma * 100, line_dash="dash", line_color="gray",
                       annotation_text=f"σ={sigma*100:.0f}%")
        fig2.update_layout(title=f"Prix du {option_type} vs Volatilité",
                           xaxis_title="Volatilité (%)", yaxis_title="Prix ($)",
                           height=350, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        from src.options_pricer import BlackScholesPricer
        deltas, gammas, vegas, thetas = [], [], [], []
        for s in S_range:
            p = BlackScholesPricer(s, K, T, r, sigma)
            deltas.append(p.delta(option_type))
            gammas.append(p.gamma())
            vegas.append(p.vega())
            thetas.append(p.theta(option_type))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=S_range, y=deltas, name='Delta', line=dict(color='#7c3aed')))
        fig3.add_trace(go.Scatter(x=S_range, y=[g * 100 for g in gammas], name='Gamma ×100', line=dict(color='#06b6d4')))
        fig3.add_vline(x=S0, line_dash="dash", line_color="gray")
        fig3.add_vline(x=K, line_dash="dot", line_color="red")
        fig3.update_layout(title="Delta et Gamma vs Prix du sous-jacent",
                           xaxis_title="Prix ($)", height=350, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.subheader("🔍 Volatilité Implicite")
    col_iv1, col_iv2 = st.columns([1, 1])
    with col_iv1:
        market_price = st.number_input("Prix de marché observé ($)", value=15.0, min_value=0.01)
        if st.button("Calculer IV", type="primary"):
            iv_data, iv_err = call_api("/pricer/implied_vol", {
                "market_price": market_price, "S0": S0, "K": K,
                "T": T, "r": r, "option_type": option_type
            })
            if iv_err:
                st.error(iv_err)
            else:
                with col_iv2:
                    st.metric("Volatilité Implicite",
                              f"{iv_data['implied_volatility_pct']:.2f}%")
                    st.caption(f"Pour justifier un prix de {market_price}$ sur ce {option_type}, le marché anticipe une vol de {iv_data['implied_volatility_pct']:.2f}%")

# ── PAGE : VOL SURFACE ───────────────────────────────────────────────────
elif page == "📈 Surface de Volatilité":
    st.title("📈 Surface de Volatilité Implicite")
    st.caption("Smile et skew de volatilité — données yfinance temps réel")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker", value="NVDA").upper()
    with col2:
        n_exp = st.slider("Nombre d'échéances", 1, 5, 3)
    with col3:
        r_vol = st.slider("Taux sans risque (%)", 0.0, 10.0, 4.5) / 100

    if st.button("📡 Récupérer la surface", type="primary"):
        with st.spinner(f"Récupération des options {ticker}..."):
            data, err = call_api("/vol_surface/", {
                "ticker": ticker, "r": r_vol,
                "n_expirations": n_exp,
                "moneyness_min": 0.80, "moneyness_max": 1.20
            })

        if err:
            st.error(err)
        else:
            st.success(f"✅ {data['n_contracts']} contrats | {ticker} @ {data['S0']}$")

            df_vol = pd.DataFrame(data['data'])

            # Smile par échéance
            fig = go.Figure()
            colors = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

            for i, t_days in enumerate(sorted(df_vol['T_days'].unique())):
                subset = df_vol[df_vol['T_days'] == t_days].sort_values('moneyness')
                fig.add_trace(go.Scatter(
                    x=subset['moneyness'],
                    y=subset['implied_volatility_pct'],
                    mode='lines+markers',
                    name=f"T = {t_days}j",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5)
                ))

            fig.add_vline(x=1.0, line_dash="dash", line_color="white",
                          annotation_text="ATM")
            fig.update_layout(
                title=f"Smile de Volatilité Implicite — {ticker}",
                xaxis_title="Moneyness (K/S₀)",
                yaxis_title="Volatilité Implicite (%)",
                template="plotly_dark", height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tableau
            st.subheader("Données brutes")
            st.dataframe(
                df_vol[['K', 'T_days', 'moneyness', 'mid_price',
                         'implied_volatility_pct', 'yahoo_iv_pct']].round(2),
                use_container_width=True
            )

# ── PAGE : SIGNAL ────────────────────────────────────────────────────────
elif page == "🤖 Signal LSTM+HMM":
    st.title("🤖 Signal LSTM+HMM")
    st.caption("Oracle de volatilité — Dernier état connu du signal de vol arbitrage")
    st.divider()

    data, err = call_api("/signal/")
    if err:
        st.error(err)
    else:
        col1, col2, col3 = st.columns(3)

        action = data['action']
        color = "🟢" if action == "VENDRE_VOL" else "🟡"

        with col1:
            st.metric("Signal", f"{color} {action}")
        with col2:
            st.metric("Date", data['last_date'])
        with col3:
            st.metric("Régime HMM", data['hmm_regime'])

        st.info(data['interpretation'])
        st.warning(data['warning'])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("RV actuelle (21j)", f"{data['rv_current_21d_pct']:.1f}%")
        with col_b:
            st.metric("Δ RV prédit", f"{data['delta_rv_predicted']:+.1f}%")
        with col_c:
            st.metric("RV future prédite", f"{data['rv_future_predicted_pct']:.1f}%",
                      delta=f"{data['delta_rv_predicted']:+.1f}%")

        st.divider()
        st.subheader("Historique du signal (30 derniers jours)")

        hist_data, hist_err = call_api("/signal/history", {"last_n": 60})
        if not hist_err:
            df_hist = pd.DataFrame(hist_data['history'])
            df_hist['date'] = pd.to_datetime(df_hist['date'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist['date'], y=df_hist['rv_current_pct'],
                mode='lines', name='RV actuelle (%)',
                line=dict(color='#7c3aed', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=df_hist['date'],
                y=df_hist['delta_rv_predicted_pct'],
                mode='lines', name='Δ RV prédit (%)',
                line=dict(color='#06b6d4', width=1.5)
            ))

            # Zones signal actif
            signal_dates = df_hist[df_hist['signal_combined'] == 1]['date']
            for d in signal_dates:
                fig.add_vrect(x0=d, x1=d + pd.Timedelta(days=1),
                              fillcolor="green", opacity=0.1, line_width=0)

            fig.update_layout(
                title="Signal LSTM+HMM — Historique 60 jours",
                xaxis_title="Date", yaxis_title="%",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Signal actif sur {hist_data['pct_signal_active']:.1f}% des jours")

# ── PAGE : BACKTEST ──────────────────────────────────────────────────────
elif page == "📊 Backtest":
    st.title("📊 Backtest — Short Straddle Delta-Hedgé")
    st.caption("Résultats historiques sur NVDA · 2023-2026 · Signal LSTM+HMM")
    st.divider()

    data, err = call_api("/backtest/summary")
    if err:
        st.error(err)
    else:
        perf = data['performance']
        baseline = data['vs_baseline']

        st.subheader("Performance — Signal LSTM+HMM")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Trades", perf['n_trades'])
        with c2:
            st.metric("PnL Total", f"+{perf['pnl_total_net_usd']}$")
        with c3:
            st.metric("Win Rate", f"{perf['win_rate_pct']}%")
        with c4:
            st.metric("Profit Factor", perf['profit_factor'],
                      delta="< 1.0 ⚠" if perf['profit_factor'] < 1 else "✓")
        with c5:
            st.metric("Sharpe", perf['sharpe_approx'])

        st.divider()
        st.subheader("vs Baseline Systématique")

        col_cmp1, col_cmp2 = st.columns(2)
        with col_cmp1:
            compare_df = pd.DataFrame({
                "Métrique": ["Trades", "PnL Total ($)", "PnL/Trade ($)", "Win Rate (%)"],
                "LSTM+HMM": [
                    perf['n_trades'],
                    perf['pnl_total_net_usd'],
                    perf['pnl_avg_per_trade_usd'],
                    perf['win_rate_pct']
                ],
                "Baseline": [
                    baseline['baseline_n_trades'],
                    baseline['baseline_pnl_total_usd'],
                    baseline['baseline_pnl_avg_per_trade_usd'],
                    baseline['baseline_win_rate_pct']
                ]
            })
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        with col_cmp2:
            st.metric("Amélioration PnL/Trade",
                      f"+{baseline['improvement_pnl_per_trade_pct']:.1f}%",
                      help="Le signal LSTM+HMM améliore le PnL moyen par trade de 27%")
            st.metric("Réduction nombre de trades",
                      f"{perf['n_trades']} vs {baseline['baseline_n_trades']}",
                      delta=f"-{baseline['baseline_n_trades']-perf['n_trades']} trades évités")

        st.divider()
        st.subheader("Trades individuels")

        trades_data, trades_err = call_api("/backtest/trades")
        if trades_err:
            st.warning(trades_err)
            st.caption("Pour activer : `df_trades.to_csv('../data/processed/backtest_trades.csv')` en fin de notebook 06")
        else:
            df_t = pd.DataFrame(trades_data['trades'])
            df_t['color'] = df_t['result'].map({'WIN': '🟢', 'LOSS': '🔴'})

            # PnL cumulé
            df_t['cumulative_pnl'] = df_t['pnl_net_usd'].cumsum()
            df_t['entry_date'] = pd.to_datetime(df_t['entry_date'])

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['cumulative_pnl'],
                mode='lines+markers',
                line=dict(color='#7c3aed', width=2),
                marker=dict(
                    color=['green' if p > 0 else 'red' for p in df_t['pnl_net_usd']],
                    size=8
                ),
                name='PnL Cumulé'
            ))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_pnl.update_layout(
                title="PnL Cumulé — Trades individuels",
                xaxis_title="Date d'entrée",
                yaxis_title="PnL ($)",
                template="plotly_dark", height=350
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

            # IV vs RV
            fig_vrp = go.Figure()
            fig_vrp.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['iv_used_pct'],
                name='IV estimée (%)', line=dict(color='orange', dash='dash')
            ))
            fig_vrp.add_trace(go.Scatter(
                x=df_t['entry_date'], y=df_t['rv_realized_pct'],
                name='RV réalisée (%)', line=dict(color='white')
            ))
            fig_vrp.update_layout(
                title="IV Estimée vs RV Réalisée par Trade",
                xaxis_title="Date", yaxis_title="Volatilité (%)",
                template="plotly_dark", height=300
            )
            st.plotly_chart(fig_vrp, use_container_width=True)

            st.dataframe(
                df_t[['entry_date', 'exit_date', 'S0', 'K',
                       'iv_used_pct', 'rv_realized_pct', 'vrp_pct',
                       'pnl_net_usd', 'result']].rename(columns={
                    'entry_date': 'Entrée', 'exit_date': 'Sortie',
                    'iv_used_pct': 'IV (%)', 'rv_realized_pct': 'RV (%)',
                    'vrp_pct': 'VRP (%)', 'pnl_net_usd': 'PnL ($)',
                    'result': 'Résultat'
                }),
                use_container_width=True, hide_index=True
            )

        st.divider()
        st.subheader("⚠ Limites connues")
        for lim in data['known_limitations']:
            st.markdown(f"- {lim}")