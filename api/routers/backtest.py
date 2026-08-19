from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import os

router = APIRouter(prefix="/backtest", tags=["Backtest Summary"])

SIGNAL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'processed', 'signal_vol_arb.csv'
)


@router.get("/summary", summary="Résumé des métriques du backtest historique (notebook 06)")
def backtest_summary():
    """
    Retourne les métriques pré-calculées du backtest de vol arbitrage.
    Ces métriques sont issues du notebook 06 — elles ne sont pas recalculées
    en temps réel pour des raisons de performance.
    """
    # Métriques figées issues du notebook 06 (version sans filtre Earnings)
    summary = {
        "strategy": "Short Straddle Delta-Hedged — Signal LSTM+HMM",
        "underlying": "NVDA",
        "period": {
            "start": "2023-10-31",
            "end": "2026-05-26"
        },
        "signal_filter": "LSTM (delta RV < 0) + HMM (régime non-Bear)",
        "performance": {
            "n_trades": 21,
            "pnl_total_net_usd": 32.02,
            "pnl_avg_per_trade_usd": 1.52,
            "win_rate_pct": 71.4,
            "profit_factor": 0.84,
            "max_drawdown_usd": -15.69,
            "sharpe_approx": 0.95,
            "vrp_avg_pct": 7.6,
            "trades_vrp_positive": "16/21"
        },
        "vs_baseline": {
            "baseline_strategy": "Short Straddle systématique (sans signal)",
            "baseline_n_trades": 31,
            "baseline_pnl_total_usd": 37.17,
            "baseline_pnl_avg_per_trade_usd": 1.20,
            "baseline_win_rate_pct": 67.7,
            "improvement_pnl_per_trade_pct": 26.7
        },
        "known_limitations": [
            "IV proxy = RV_actuelle + 5% fixe (pas d'IV historique réelle)",
            "Profit Factor < 1.0 (pertes extrêmes > gains moyens)",
            "Trade 2025-01-07 : choc DeepSeek/correction tech, RV réalisée 85% vs IV vendue 49%",
            "Coûts de transaction simplifiés (pas de bid/ask options, pas de margin)",
            "n=21 trades insuffisant pour validation statistique robuste"
        ],
        "data_source": "Notebook 06 — dernière exécution requise pour mise à jour"
    }

    return summary


@router.get("/trades", summary="Liste des trades individuels du backtest")
def backtest_trades():
    """
    Retourne les trades individuels si le fichier de résultats est disponible.
    Sinon retourne les métriques agrégées uniquement.
    """
    trades_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'processed', 'backtest_trades.csv'
    )

    if not os.path.exists(trades_path):
        raise HTTPException(
            status_code=503,
            detail="Fichier backtest_trades.csv introuvable. Exporter df_trades depuis le notebook 06 : df_trades.to_csv('../data/processed/backtest_trades.csv')"
        )

    df = pd.read_csv(trades_path, index_col=0, parse_dates=True)
    df = df.replace({np.nan: None})

    records = []
    for date, row in df.iterrows():
        records.append({
            "entry_date": str(date.date()),
            "exit_date": str(row.get('Exit_Date', '')),
            "S0": round(float(row['S0']), 2),
            "K": float(row['K']),
            "iv_used_pct": round(float(row['IV_used']) * 100, 1),
            "rv_realized_pct": round(float(row['RV_realized']) * 100, 1),
            "vrp_pct": round(float(row['VRP_realized']) * 100, 1),
            "pnl_net_usd": round(float(row['Final_PnL_net']), 2),
            "result": "WIN" if float(row['Final_PnL_net']) > 0 else "LOSS"
        })

    total_pnl = sum(r['pnl_net_usd'] for r in records)
    win_rate = sum(1 for r in records if r['result'] == 'WIN') / len(records) * 100

    return {
        "n_trades": len(records),
        "total_pnl_net_usd": round(total_pnl, 2),
        "win_rate_pct": round(win_rate, 1),
        "trades": records
    }