from fastapi import APIRouter, HTTPException, Query
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from src.data_loader import get_option_chain_data
from src.options_pricer import BlackScholesPricer

router = APIRouter(prefix="/vol_surface", tags=["Surface de Volatilité"])


@router.get("/", summary="Calcule le smile de volatilité implicite pour un ticker donné")
def vol_surface(
    ticker: str = Query("NVDA", description="Ticker Yahoo Finance", example="NVDA"),
    r: float = Query(0.045, description="Taux sans risque"),
    moneyness_min: float = Query(0.80, description="Borne inférieure de moneyness"),
    moneyness_max: float = Query(1.20, description="Borne supérieure de moneyness"),
    n_expirations: int = Query(3, description="Nombre d'échéances à afficher", ge=1, le=6),
):
    try:
        df_options = get_option_chain_data(ticker, min_days_to_expiration=5)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur récupération options {ticker} : {str(e)}")

    S0 = df_options['S0'].iloc[0]

    # Filtre moneyness et type call uniquement
    df_calls = df_options[
        (df_options['option_type'] == 'call') &
        (df_options['K'] >= S0 * moneyness_min) &
        (df_options['K'] <= S0 * moneyness_max) &
        (df_options['volume'] >= 10)
    ].copy()

    if df_calls.empty:
        raise HTTPException(status_code=404, detail="Aucun contrat liquide trouvé dans cette fenêtre de moneyness")

    # Sélection des N premières échéances
    echeances = sorted(df_calls['T'].unique())[:n_expirations]
    df_calls = df_calls[df_calls['T'].isin(echeances)]

    # Calcul IV
    results = []
    for _, row in df_calls.iterrows():
        iv = BlackScholesPricer.implied_volatility(
            market_price=row['mid_price'],
            S0=row['S0'], K=row['K'], T=row['T'],
            r=r, option_type='call'
        )
        if iv and not np.isnan(iv) and 0.01 < iv < 2.0:
            results.append({
                "ticker": ticker,
                "S0": round(S0, 2),
                "K": row['K'],
                "T_years": round(row['T'], 4),
                "T_days": round(row['T'] * 365),
                "moneyness": round(row['K'] / S0, 4),
                "mid_price": round(row['mid_price'], 4),
                "implied_volatility": round(iv, 4),
                "implied_volatility_pct": round(iv * 100, 2),
                "yahoo_iv_pct": round(row['yahoo_iv'] * 100, 2) if not pd.isna(row['yahoo_iv']) else None,
            })

    if not results:
        raise HTTPException(status_code=422, detail="Aucune IV calculable sur cette fenêtre")

    return {
        "ticker": ticker,
        "S0": round(S0, 2),
        "n_contracts": len(results),
        "expirations_days": [round(e * 365) for e in echeances],
        "data": results
    }