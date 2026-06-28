import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

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



def get_option_chain_data(ticker_symbol: str, min_days_to_expiration: int = 3) -> pd.DataFrame:
    """
    Télécharge et nettoie la chaîne d'options complète pour un ticker donné.
    Filtre les contrats illiquides et ceux trop proches de l'expiration.
    """
    print(f"Récupération de la chaîne d'options pour {ticker_symbol}...")
    stock = yf.Ticker(ticker_symbol)
    
    # 1. Récupération du prix actuel du sous-jacent (S0)
    try:
        # history() est plus fiable que fast_info pour avoir le dernier prix
        S0 = stock.history(period="1d")['Close'].iloc[-1] 
    except Exception as e:
        raise ValueError(f"Impossible de récupérer le prix de {ticker_symbol} : {e}")

    # 2. Récupération des dates d'expiration
    expirations = stock.options
    if not expirations:
        raise ValueError(f"Aucune option trouvée pour {ticker_symbol}")

    options_data = []
    today = pd.Timestamp(datetime.today().date())

    # On ignore les avertissements de pandas lors des concaténations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 3. Boucle sur chaque échéance
        for date_str in expirations:
            exp_date = pd.Timestamp(date_str)
            days_to_exp = (exp_date - today).days
            
            # FILTRE 1 : On ignore les options qui expirent dans moins de X jours 
            # (Pour éviter l'explosion du Gamma et la division par zéro du Vega)
            if days_to_exp < min_days_to_expiration:
                continue
                
            T = days_to_exp / 365.0 # Temps en années
            
            # Téléchargement des Calls et Puts
            opt_chain = stock.option_chain(date_str)
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()
            
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'
            
            # Assemblage
            chain = pd.concat([calls, puts], ignore_index=True)
            chain['T'] = T
            chain['S0'] = S0
            chain['expiration_date'] = exp_date
            
            options_data.append(chain)

    if not options_data:
        raise ValueError("Toutes les options ont été filtrées (trop proches de l'expiration).")

    # 4. Création du DataFrame global
    full_chain = pd.concat(options_data, ignore_index=True)
    
    # 5. LES FILTRES QUANTITATIFS (Nettoyage de la donnée de marché)
    # Règle d'or : On s'assure que le Bid et le Ask existent.
    full_chain = full_chain.dropna(subset=['bid', 'ask'])
    
    # On supprime les options sans aucun acheteur ou vendeur (Spread inexistant)
    full_chain = full_chain[((full_chain['bid'] > 0) & (full_chain['ask'] > 0)) | (full_chain['lastPrice'] > 0)]
    
    # On évite les spreads monstrueux (Bid/Ask aberrant) qui faussent le Mid-Price
    # Ex: Bid à 1$, Ask à 50$ -> Le spread est plus grand que le Bid lui-même. On filtre.
    full_chain = full_chain[(full_chain['ask'] - full_chain['bid']) / full_chain['bid'] < 2.0]
    
    # 6. Le Prix du Marché : On utilise le Mid-Price (Moyenne Bid/Ask)
    # C'est la vraie valeur consensuelle, car le 'lastPrice' peut dater de plusieurs heures/jours
    full_chain['mid_price'] = np.where(
        (full_chain['bid'] > 0) & (full_chain['ask'] > 0),
        (full_chain['bid'] + full_chain['ask']) / 2.0,
        full_chain['lastPrice']
    )
    
    # 7. Sélection des colonnes utiles pour notre Pricer
    columns_to_keep = ['contractSymbol', 'option_type', 'expiration_date', 'strike', 'T', 'S0', 'bid', 'ask', 'mid_price', 'volume', 'impliedVolatility']
    clean_chain = full_chain[columns_to_keep].rename(columns={'strike': 'K', 'impliedVolatility': 'yahoo_iv'})
    
    print(f"{len(clean_chain)} contrats d'options liquides extraits avec succès.")
    return clean_chain