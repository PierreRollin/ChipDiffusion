from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.options_pricer import BlackScholesPricer

router = APIRouter(prefix="/pricer", tags=["Pricer BSM"])


class PricerResponse(BaseModel):
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: str
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    implied_volatility: float | None = None


@router.get("/", response_model=PricerResponse, summary="Price une option européenne (BSM) et retourne les Greeks")
def price_option(
    S0: float = Query(..., description="Prix actuel du sous-jacent ($)", examples=275.0),
    K: float = Query(..., description="Strike de l'option ($)", examples=280.0),
    T: float = Query(..., description="Maturité en années (ex: 0.5 = 6 mois)", examples=0.5),
    r: float = Query(0.045, description="Taux sans risque annualisé", examples=0.045),
    sigma: float = Query(..., description="Volatilité implicite (ex: 0.35 = 35%)", examples=0.35),
    option_type: Literal['call', 'put'] = Query('call', description="Type d'option"),
):
    if S0 <= 0 or K <= 0:
        raise HTTPException(status_code=422, detail="S0 et K doivent être strictement positifs")
    if T <= 0:
        raise HTTPException(status_code=422, detail="T doit être strictement positif")
    if sigma <= 0:
        raise HTTPException(status_code=422, detail="sigma doit être strictement positif")

    pricer = BlackScholesPricer(S0, K, T, r, sigma)

    price = pricer.call_price() if option_type == 'call' else pricer.put_price()
    delta = pricer.delta(option_type)
    gamma = pricer.gamma()
    vega = pricer.vega()
    theta = pricer.theta(option_type)

    return PricerResponse(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        option_type=option_type,
        price=round(price, 4),
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta=round(theta, 4),
    )


@router.get("/implied_vol", summary="Calcule la volatilité implicite à partir d'un prix de marché")
def implied_vol(
    market_price: float = Query(..., description="Prix observé sur le marché ($)", examples=15.0),
    S0: float = Query(..., examples=275.0),
    K: float = Query(..., examples=280.0),
    T: float = Query(..., examples=0.5),
    r: float = Query(0.045),
    option_type: Literal['call', 'put'] = Query('call'),
):
    iv = BlackScholesPricer.implied_volatility(
        market_price=market_price,
        S0=S0, K=K, T=T, r=r,
        option_type=option_type
    )

    if iv is None or (hasattr(iv, '__float__') and __import__('math').isnan(float(iv))):
        raise HTTPException(
            status_code=422,
            detail="Impossible de calculer l'IV : prix incohérent avec les paramètres (arbitrage ou hors bornes)"
        )

    return {
        "market_price": market_price,
        "S0": S0, "K": K, "T": T, "r": r,
        "option_type": option_type,
        "implied_volatility": round(float(iv), 4),
        "implied_volatility_pct": round(float(iv) * 100, 2)
    }