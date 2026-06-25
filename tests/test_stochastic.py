import unittest
import numpy as np
from src.stochastic import MonteCarloGBM

class TestMonteCarloGBM(unittest.TestCase):
    
    def test_expected_value(self):
        """Vérifie que la moyenne de la simulation respecte E[S_T] = S0 * exp(mu * T)"""
        # Paramètres de test
        S0, mu, sigma, T, N, M = 100.0, 0.05, 0.20, 1.0, 252, 50000
        
        gbm = MonteCarloGBM(S0, mu, sigma, T, N, M, seed=42)
        paths = gbm.simulate()
        
        valeur_empirique = np.mean(paths[-1, :])
        valeur_theorique = S0 * np.exp(mu * T)
        
        erreur_relative = abs(valeur_empirique - valeur_theorique) / valeur_theorique
        
        # Le test passe si l'erreur est inférieure à 0.5%
        self.assertLess(erreur_relative, 0.005, f"Erreur de pricing trop élevée: {erreur_relative}")

if __name__ == '__main__':
    unittest.main()