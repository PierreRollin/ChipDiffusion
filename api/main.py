from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import pricer, vol_surface, signal, backtest

app = FastAPI(
    title="ChipDiffusion API",
    description="""
    API de pricing d'options et de vol arbitrage sur la supply chain semi-conducteurs.
    
    ## Modules disponibles
    
    - **/pricer** — Black-Scholes : prix, Greeks, volatilité implicite
    - **/vol_surface** — Smile de volatilité implicite (données yfinance temps réel)
    - **/signal** — Signal LSTM+HMM de vol arbitrage (dernier état connu)
    - **/backtest** — Métriques du backtest historique (notebook 06)
    
    ## Stack technique
    
    Black-Scholes (analytique) | HMM Walk-Forward | LSTM CNN1D | Delta Hedging discret
    
    ## Limites
    
    Les données options proviennent de Yahoo Finance (hors heures de marché, 
    pas d'IV historique). Le signal LSTM+HMM est statique — il nécessite 
    un re-run du notebook 05 pour être mis à jour.
    """,
    version="1.0.0",
    contact={"name": "ChipDiffusion Project"}
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pricer.router)
app.include_router(vol_surface.router)
app.include_router(signal.router)
app.include_router(backtest.router)


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "project": "ChipDiffusion",
        "version": "1.0.0",
        "routes": ["/pricer", "/vol_surface", "/signal", "/backtest/summary", "/backtest/trades", "/docs"]
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}