import unittest
import pandas as pd
import numpy as np
from src.volatility_analyzer import VolatilityAnalyzer

class TestVolatilityAnalyzer(unittest.TestCase):
    
    def test_calculate_realized_volatility(self):
        """Vérifie le calcul de la Volatilité Réalisée (RV)"""
        
        # 1. On crée 100 jours de rendements constants à 1% (0.01)
        # L'écart-type d'une constante est 0. Donc la RV doit être 0.
        ret_constants = pd.Series(np.full(100, 0.01))
        rv_constants = VolatilityAnalyzer.calculate_realized_volatility(ret_constants, window=21)
        
        # Le dernier jour doit valoir exactement 0
        self.assertAlmostEqual(rv_constants.iloc[-1], 0.0, places=5)
        
        # 2. On crée une série avec une variance mathématique connue
        # Si on alterne +1% et -1%, l'écart-type est approximativement 0.01
        ret_alternes = pd.Series([0.01, -0.01] * 50)
        rv_alternes = VolatilityAnalyzer.calculate_realized_volatility(ret_alternes, window=21)
        
        # La valeur théorique doit être proche de 0.01 * sqrt(252) ≈ 0.1587 (soit 15.87%)
        valeur_theorique = 0.01 * np.sqrt(252)
        
        # On teste que la valeur calculée est proche de la théorie (avec une tolérance car c'est un échantillon)
        self.assertAlmostEqual(rv_alternes.iloc[-1], valeur_theorique, places=2)

if __name__ == '__main__':
    unittest.main()