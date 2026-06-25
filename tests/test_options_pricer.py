import unittest
import numpy as np
from src.options_pricer import BlackScholesPricer

class TestBlackScholesPricer(unittest.TestCase):
    
    def setUp(self):
        # Paramètres de la Séance 4 (Notebook 02_black_scholes_maths.ipynb)
        self.S0 = 180.0
        self.K = 185.0
        self.T = 0.5
        self.r = 0.04
        self.sigma = 0.25
        self.pricer = BlackScholesPricer(self.S0, self.K, self.T, self.r, self.sigma)

    def test_call_price(self):
        """Vérifie le calcul analytique du Call"""
        call = self.pricer.call_price()
        self.assertAlmostEqual(call, 12.068, places=3, msg="Prix du Call incorrect")    # avant à 12.067 mais modif pour question d'arrondi

    def test_put_price(self):
        """Vérifie le calcul analytique du Put"""
        put = self.pricer.put_price()
        self.assertAlmostEqual(put, 13.404, places=3, msg="Prix du Put incorrect")

    def test_call_put_parity(self):
        """Vérifie la parité Call-Put : Call - Put = S0 - K * exp(-rT)"""
        call = self.pricer.call_price()
        put = self.pricer.put_price()
        
        gauche = call - put
        droite = self.S0 - self.K * np.exp(-self.r * self.T)
        
        self.assertAlmostEqual(gauche, droite, places=5, msg="Violation de la Parité Call-Put")

    def test_expiration_zero(self):
        """Vérifie qu'il n'y a pas de division par zéro à T=0 et que l'on renvoie la valeur intrinsèque"""
        pricer_exp = BlackScholesPricer(S0=100, K=90, T=0.0, r=0.04, sigma=0.25)
        # Call ITM : doit valoir 10
        self.assertEqual(pricer_exp.call_price(), 10.0)
        # Put OTM : doit valoir 0
        self.assertEqual(pricer_exp.put_price(), 0.0)

    def test_implied_volatility(self):
        """Vérifie le Round-Trip : Volatilité -> Prix -> Volatilité Implicite"""
        # 1. On part d'une volatilité connue (ex: 20%)
        vol_initiale = 0.20
        pricer_test = BlackScholesPricer(S0=100, K=100, T=1.0, r=0.05, sigma=vol_initiale)
        
        # 2. On calcule le prix du marché théorique
        prix_marche = pricer_test.call_price()
        
        # 3. On inverse la machine pour retrouver la volatilité
        vol_retrouvee = BlackScholesPricer.implied_volatility(
            market_price=prix_marche, 
            S0=100, K=100, T=1.0, r=0.05, option_type='call'
        )
        
        # L'algorithme doit retrouver 0.20 !
        self.assertAlmostEqual(vol_initiale, vol_retrouvee, places=4, msg="L'algorithme de Newton-Raphson a divergé")

if __name__ == '__main__':
    unittest.main()