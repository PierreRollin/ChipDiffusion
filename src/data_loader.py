import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_market_data(tickers, start_date, end_date, output_path):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']    # prix de clôture ajustés
    data.to_csv(output_path)
    return data

def process_market_data(raw_csv_path, processed_prices_path, processed_returns_path):
    data = pd.read_csv(raw_csv_path, index_col=0, parse_dates=True)
    
    # On supprime les jours où la bourse US est fermée (week-ends, jours fériés US)
    data_us_aligned = data.dropna(subset=['NVDA'])

    # on récupère les jours où l'on fill les prix
    mask_missing = data_us_aligned.isna()
    
    # On propage le dernier prix connu afin de synchroniser
    # les actifs étrangers sur le calendrier de référence US (notamment ici avec SMIC : bourse HK)
    prices_filled = data_us_aligned.ffill()
    
    # On supprime les IPOs récentes (les lignes du début avec des NaN persistants)
    prices_filled = prices_filled.dropna()
    
    # On supprime les 2 derniers jours pour éviter les décalages de fuseau (Clôture HK vs Ouverture US)
    prices_filled = prices_filled.iloc[:-2]
    
    # SAUVEGARDE DES PRIX
    prices_filled.to_csv(processed_prices_path)
    

    # CORRECTION QUANTITATIVE DES RENDEMENTS
    # On calcule les rendements log sur les prix synchronisés
    returns = np.log(prices_filled / prices_filled.shift(1))
    
    # On neutralise les rendements associés aux observations
    # reconstruites via ffill().
    mask_missing = mask_missing.loc[prices_filled.index]
    returns[mask_missing] = np.nan
    
    # SAUVEGARDE DES RENDEMENTS POUR LE MOTEUR MONTE CARLO
    returns.to_csv(processed_returns_path)

    return prices_filled, returns




commodities_and_macro = ['USO', 'CGW', 'PICK', '^VIX', '^TNX']  # 'USO' (Pétrole), 'CGW' (Eau), 'PICK' (Métaux/Silicium), '^VIX' (Peur), '^TNX' (Taux d'intérêt)
foundries_and_equipment = ['ASML', 'TSM', '0981.HK'] # 'TSM' (TSMC), '0981.HK' (SMIC / Proxy Huawei Chine)
fabless_designers = ['NVDA', 'AMD', 'INTC', 'AVGO'] # 'AVGO' (Broadcom)
integrators_and_ai = ['AAPL', 'MSFT', 'GOOGL', 'TSLA'] # 'MSFT' (Proxy OpenAI), 'GOOGL' (Proxy Anthropic)
inversely_correlated_tickers = ['GLD', 'TLT']  # 'GLD' (Or), 'TLT' (Obligations à long terme)

all_tickers = commodities_and_macro + foundries_and_equipment + fabless_designers + integrators_and_ai + inversely_correlated_tickers

print("Nombres total de tickers : ", len(all_tickers))
print ("Tickers : ", all_tickers)

raw_csv = RAW_DATA_DIR / "chip_chain_raw.csv"
processed_prices = PROCESSED_DATA_DIR / "chip_chain_prices.csv"
processed_returns = PROCESSED_DATA_DIR / "chip_chain_returns.csv"

download_market_data(all_tickers, "2010-01-01", "2026-12-31", raw_csv)
process_market_data(raw_csv, processed_prices, processed_returns)