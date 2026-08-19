from fastapi import APIRouter, HTTPException
import pandas as pd
import os

router = APIRouter(prefix="/signal", tags=["Signal LSTM+HMM"])

SIGNAL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'processed', 'signal_vol_arb.csv'
)


@router.get("/", summary="Retourne le dernier état du signal de vol arbitrage (LSTM+HMM)")
def get_signal():
    if not os.path.exists(SIGNAL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Fichier signal introuvable. Relancer le notebook 05 pour générer signal_vol_arb.csv"
        )

    df = pd.read_csv(SIGNAL_PATH, index_col=0, parse_dates=True)

    if df.empty:
        raise HTTPException(status_code=503, detail="Signal vide")

    last = df.iloc[-1]
    last_date = df.index[-1]

    # Interprétation du signal
    signal_active = int(last.get('Signal_Combined', 0))
    delta_rv = float(last.get('Delta_RV_Predicted', 0))
    rv_current = float(last.get('RV_Current_21d', 0))

    if signal_active == 1:
        interpretation = "FAVORABLE — RV future prédite inférieure à RV actuelle, régime non-Bear. Condition favorable pour vendre la volatilité."
        action = "VENDRE_VOL"
    else:
        interpretation = "NEUTRE — Signal LSTM ou régime HMM défavorable. Pas de position recommandée."
        action = "NEUTRE"

    # Récupération du régime HMM
    regime_cols = [c for c in df.columns if 'Regime' in c]
    regime_info = {col: int(last[col]) for col in regime_cols if col in last}
    regime_actif = [k for k, v in regime_info.items() if v == 1]

    return {
        "last_date": str(last_date.date()),
        "signal_combined": signal_active,
        "action": action,
        "interpretation": interpretation,
        "delta_rv_predicted": round(delta_rv * 100, 2),
        "rv_current_21d_pct": round(rv_current * 100, 2),
        "rv_future_predicted_pct": round((rv_current + delta_rv) * 100, 2),
        "hmm_regime": regime_actif[0] if regime_actif else "inconnu",
        "regimes_detail": regime_info,
        "warning": "Signal basé sur données historiques jusqu'au dernier run du notebook 05. Non temps-réel."
    }


@router.get("/history", summary="Retourne l'historique complet du signal")
def get_signal_history(last_n: int = 30):
    if not os.path.exists(SIGNAL_PATH):
        raise HTTPException(status_code=503, detail="Fichier signal introuvable")

    df = pd.read_csv(SIGNAL_PATH, index_col=0, parse_dates=True)
    df_last = df.tail(last_n)

    records = []
    for date, row in df_last.iterrows():
        records.append({
            "date": str(date.date()),
            "signal_combined": int(row.get('Signal_Combined', 0)),
            "delta_rv_predicted_pct": round(float(row.get('Delta_RV_Predicted', 0)) * 100, 2),
            "rv_current_pct": round(float(row.get('RV_Current_21d', 0)) * 100, 2),
        })

    return {
        "n_days": len(records),
        "pct_signal_active": round(sum(r['signal_combined'] for r in records) / len(records) * 100, 1),
        "history": records
    }