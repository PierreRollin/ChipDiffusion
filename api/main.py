from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import pricer, vol_surface, signal, backtest

app = FastAPI(
    title="ChipDiffusion API",
    description="""
    Options pricing and volatility arbitrage API 
    for the semiconductor supply chain.

    ## Available Modules

    - **/pricer** — Black-Scholes: price, Greeks, implied volatility
    - **/vol_surface** — Implied volatility smile (live yfinance data)
    - **/signal** — LSTM+HMM vol arbitrage signal (last known state)
    - **/backtest** — Historical backtest metrics (notebook 06)

    ## Technical Stack

    Black-Scholes (analytical) | HMM Walk-Forward | 
    LSTM CNN1D | Discrete Delta Hedging

    ## Known Limitations

    Options data from Yahoo Finance (off-market hours, 
    no historical IV). LSTM+HMM signal is static — 
    requires re-running notebook 05 to update.
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
        "routes": {
            "pricer": ["/pricer/", "/pricer/implied_vol"],
            "vol_surface": ["/vol_surface/"],
            "signal": ["/signal/", "/signal/history"],
            "backtest": ["/backtest/summary", "/backtest/trades"],
            "health": ["/", "/health"],
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}