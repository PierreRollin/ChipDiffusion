# ChipDiffusion : Volatility Arbitrage Engine

> End-to-end quantitative finance system modelling **Variance Risk Premium (IV - RV)** on NVDA within the semiconductor supply chain.

---

## Key Results

| Module | Result | Detail |
|---|---|---|
| BSM Pricer | MAE vs Yahoo Finance: 5.0% | Structural gap: BSM vs binomial with dividends |
| Volatility Surface | Full NVDA smile reconstructed | Moneyness [0.80–1.20], Newton-Raphson inversion |
| LSTM VRP Engine | Edge confirmed: gain +0.38% | p=0.006, 9/10 seeds positive (n=10 robustness test) |
| Historical Backtest | Win Rate 71.4%, Sharpe 0.95 | 21 trades on real NVDA prices, 2023–2026 |
| vs Systematic Baseline | +27% PnL/trade | 21 filtered trades vs 31 unfiltered |
| Heston Monte Carlo | Kurtosis > 3, Q5% −$9 vs BSM | ρ=−0.7, Feller condition verified ✓ |

---

## Economic Thesis

Shocks in the semiconductor supply chain propagate with a time lag from raw materials (commodities) to foundries (TSM, ASML, SMIC) to chip designers (NVDA, AMD) and finally to integrators (AAPL, MSFT). This structural cascade creates **predictable windows of elevated implied volatility** on NVDA - windows where IV systematically exceeds realised volatility (RV), generating a positive Variance Risk Premium.

The strategy: **sell a delta-hedged ATM straddle** on NVDA during periods where the VRP signal (LSTM + HMM) indicates a favourable regime, then delta-hedge daily to isolate the volatility exposure.

The mathematical justification:

$$dPnL \approx \frac{1}{2} \Gamma S^2 (\sigma_{imp}^2 - \sigma_{real}^2) \, dt$$

If IV > RV, each instant dt generates a profit proportional to the option's Gamma and the squared volatility spread - independently of the direction of the underlying price.

---

## Architecture

```
Data (yfinance)
    │
    ├── Notebook 01 ── Monte Carlo GBM (engine validation)
    ├── Notebook 02 ── BSM Pricer + Greeks + IV (Newton-Raphson + Brentq)
    ├── Notebook 03 ── NVDA Volatility Surface (empirical skew)
    │
    ├── Notebook 04 ── Vol Arb theory (Straddle + simulated Delta Hedging)
    ├── Notebook 05 ── ML Engine (HMM Walk-Forward + CNN1D/LSTM)
    │                  → Signal export: data/processed/signal_vol_arb.csv
    ├── Notebook 06 ── Historical backtest (real NVDA prices 2023–2026)
    ├── Notebook 07 ── Heston Monte Carlo (beyond BSM)
    │
    ├── src/
    │   ├── options_pricer.py   BSM + Greeks + IV + VRP
    │   ├── backtester.py       Discrete Delta Hedging + stop-loss
    │   └── stochastic.py       Monte Carlo GBM
    │
    ├── api/                    FastAPI REST API (7 endpoints, 4 routers)
    └── streamlit_app.py        Interactive dashboard
```

---

## Notebooks

| # | Title | Key concepts |
|---|---|---|
| 01 | Monte Carlo GBM | Stochastic simulation, convergence validation |
| 02 | BSM Pricer | Analytical pricing, Greeks, Newton-Raphson + Brentq IV |
| 03 | Volatility Surface | Implied vol smile, moneyness filter, skew analysis |
| 04 | Vol Arb Theory | Straddle, delta hedging, PnL decomposition |
| 05 | ML Engine | HMM Walk-Forward, CNN1D + LSTM, 10-seed robustness test |
| 06 | Historical Backtest | Real NVDA prices, stop-loss, baseline comparison |
| 07 | Heston Model | Stochastic vol, Feller condition, empirical skew reproduction |

---

## API - Endpoints

Launch:
```bash
uvicorn api.main:app --reload --port 8000
# Documentation: http://localhost:8000/docs
```

| Router | Endpoint | Description |
|---|---|---|
| Pricer | `GET /pricer/` | BSM price + Greeks (Delta, Gamma, Vega, Theta) |
| Pricer | `GET /pricer/implied_vol` | Implied volatility from observed market price |
| Vol Surface | `GET /vol_surface/` | Live NVDA options chain → IV smile |
| Signal | `GET /signal/` | Last LSTM+HMM signal state |
| Signal | `GET /signal/history` | Signal history (last N days) |
| Backtest | `GET /backtest/summary` | Aggregated backtest metrics |
| Backtest | `GET /backtest/trades` | Individual trade log |

---

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
# http://localhost:8501
```

Five pages:
- **Dashboard** - Project overview and API health
- **BSM Pricer** - Interactive option pricing + Greeks + sensitivity charts
- **Volatility Surface** - Live IV smile for any ticker
- **LSTM+HMM Signal** - Last signal state + 60-day history
- **Backtest** - PnL curves, trade log, IV vs RV comparison, **VRP live monitor**

---

## Installation

```bash
git clone https://github.com/PierreRollin/ChipDiffusion
cd chipdiffusion
pip install -r requirements.txt
```

**Requirements:**
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.12.0
tensorflow>=2.15.0
hmmlearn>=0.3.0
streamlit>=1.35.0
plotly>=5.22.0
scikit-learn>=1.4.0
requests>=2.31.0
```

Notebooks additionally require:
```
matplotlib>=3.8.0
statsmodels>=0.14.0
```

---

## Known Limitations

- **IV proxy**: no historical NVDA options data available via yfinance. IV is approximated as `RV_current + 5%` in the backtest - the primary limitation of Notebook 06.
- **Profit Factor < 1.0**: losses on extreme shock events (Jan 2025: DeepSeek/tech correction, RV 85% vs IV 49%) exceed average gains. Documented and analysed in Notebook 06.
- **LSTM edge is weak**: gain of +0.38% over naive baseline, confirmed statistically (p=0.006) but economically marginal. The model is used as a directional filter, not a price oracle.
- **Heston not calibrated**: parameters (κ, θ, ξ, ρ) are illustrative. Production use would require numerical calibration on observed option prices.
- **Static signal**: the LSTM+HMM signal requires re-running Notebook 05 to update. It is not a real-time system.

---

## Project Context

ChipDiffusion is the continuous-time successor to ChipChainReaction (discrete-time HMM + LSTM trading system on the semiconductor supply chain).

The name carries a double meaning: chip for the semiconductor supply chain, and diffusion as a nod to Brownian motion, first encountered studying particle diffusion driven by thermal agitation in physics, and later rediscovered as the mathematical engine of option pricing.

The economic motivation comes from Chip War (Chris Miller, 2022), which traces the geopolitical and industrial architecture of the global semiconductor supply chain: Commodities → Foundries (TSM, ASML) → Designers (NVDA, AMD) → Integrators (AAPL, MSFT)

ChipDiffusion tests whether these structural dependencies generate a measurable and tradeable Variance Risk Premium on NVDA.

---

*Stack: Python · TensorFlow/Keras · hmmlearn · Scipy · FastAPI · Streamlit · Plotly · yfinance*