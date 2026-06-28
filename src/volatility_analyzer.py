import pandas as pd
import numpy as np
from src.data_loader import get_option_chain_data
from src.options_pricer import BlackScholesPricer

class VolatilityAnalyzer:
    
    @staticmethod
    def calculate_realized_volatility(returns_series: pd.Series, window: int = 21) -> pd.Series:
        """
        Calcule la Volatilité Réalisée (RV) glissante annualisée.
        :param returns_series: Série pandas des rendements log quotidiens.
        :param window: Fenêtre glissante (21 jours = 1 mois de bourse).
        """
        # Écart-type glissant annualisé
        return returns_series.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def extract_atm_iv(ticker: str, r: float = 0.045, target_days: int = 30) -> float:
        """
        Récupère la chaîne d'options, filtre les options At-The-Money (ATM)
        proches de 30 jours, et retourne l'IV moyenne du marché.
        C'est notre indicateur de "Peur Actuelle".
        """
        df_options = get_option_chain_data(ticker, min_days_to_expiration=5)
        
        # 1. Isoler la maturité la plus proche de target_days
        df_options['days_to_target'] = abs((df_options['T'] * 365) - target_days)
        best_T = df_options.loc[df_options['days_to_target'].idxmin(), 'T']
        
        df_target = df_options[df_options['T'] == best_T].copy()
        
        # 2. Filtrage Strict ATM (Moneyness K/S0 entre 0.95 et 1.05)
        df_atm = df_target[(df_target['K'] / df_target['S0'] >= 0.95) & (df_target['K'] / df_target['S0'] <= 1.05)]
        
        if df_atm.empty:
            return np.nan
            
        # 3. Calcul de l'IV
        ivs = []
        for _, row in df_atm.iterrows():
            iv = BlackScholesPricer.implied_volatility(
                row['mid_price'], row['S0'], row['K'], row['T'], r, row['option_type']
            )
            ivs.append(iv)
            
        return np.nanmean(ivs)

    @staticmethod
    def calculate_vrp(current_iv: float, current_rv: float) -> float:
        """
        Calcule la Volatility Risk Premium (Prime de Risque de Volatilité).
        """
        return current_iv - current_rv
    
    @staticmethod
    def get_smile_data(ticker: str, r: float = 0.045, target_days: int = 30, option_type: str = 'call') -> pd.DataFrame:
        """
        Extrait les données nécessaires pour tracer le Volatility Smile.
        Retourne un DataFrame avec les colonnes : [Moneyness, IV].
        """
        df_options = get_option_chain_data(ticker, min_days_to_expiration=5)
        
        # 1. Sélectionner l'échéance la plus proche
        df_options['days_to_target'] = abs((df_options['T'] * 365) - target_days)
        best_T = df_options.loc[df_options['days_to_target'].idxmin(), 'T']
        
        # 2. Filtrer par type (Call ou Put) et échéance
        df_smile = df_options[(df_options['T'] == best_T) & (df_options['option_type'] == option_type)].copy()
        
        # 3. Restreindre la Moneyness (ex: entre 0.80 et 1.20) pour éviter les crashs de Newton-Raphson
        df_smile = df_smile[(df_smile['K'] / df_smile['S0'] >= 0.80) & (df_smile['K'] / df_smile['S0'] <= 1.20)]
        
        # 4. Calcul de l'IV
        ivs = []
        for _, row in df_smile.iterrows():
            iv = BlackScholesPricer.implied_volatility(
                row['mid_price'], row['S0'], row['K'], row['T'], r, row['option_type']
            )
            ivs.append(iv)
            
        df_smile['IV'] = ivs
        df_smile['Moneyness'] = df_smile['K'] / df_smile['S0']
        
        # Nettoyage des NaN et tri
        df_smile = df_smile.dropna(subset=['IV']).sort_values(by='Moneyness')
        
        return df_smile[['Moneyness', 'IV', 'K', 'mid_price', 'T']]