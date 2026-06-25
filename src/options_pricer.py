import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    """
    Évaluateur d'Options Européennes via le modèle de Black-Scholes-Merton.
    """
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Gestion du piège de l'expiration (T <= 0)
        if self.T <= 0:
            self.d1 = 0.0
            self.d2 = 0.0
        else:
            self.d1 = self._calculate_d1()
            self.d2 = self._calculate_d2()

    def _calculate_d1(self) -> float:
        return (np.log(self.S0 / self.K) + (self.r + (self.sigma**2) / 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _calculate_d2(self) -> float:
        return self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self) -> float:
        """Calcule le prix d'un Call Européen"""
        if self.T <= 0:
            return max(self.S0 - self.K, 0)
        return self.S0 * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def put_price(self) -> float:
        """Calcule le prix d'un Put Européen via la Parité Call-Put"""
        if self.T <= 0:
            return max(self.K - self.S0, 0)
        # Utilisation de ton excellente intuition mathématique !
        return self.call_price() - self.S0 + self.K * np.exp(-self.r * self.T)

    def delta(self, option_type: str = 'call') -> float:
        """Calcule le Delta de l'option"""
        if self.T <= 0:
            return 1.0 if (option_type == 'call' and self.S0 > self.K) else (-1.0 if (option_type == 'put' and self.S0 < self.K) else 0.0)
        if option_type == 'call':
            return norm.cdf(self.d1)
        elif option_type == 'put':
            return norm.cdf(self.d1) - 1.0
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")

    def gamma(self) -> float:
        """Calcule le Gamma (identique pour Call et Put)."""
        if self.T <= 0:
            return 0.0
        return norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Calcule le Vega (identique pour Call et Put). Retourne la valeur pour 1% de changement de volatilité."""
        if self.T <= 0:
            return 0.0
        # On divise par 100 pour que le résultat représente un changement de 1 point (1%) de volatilité
        return self.S0 * norm.pdf(self.d1) * np.sqrt(self.T) / 100.0

    def theta(self, option_type: str = 'call') -> float:
        """Calcule le Theta (décroissance quotidienne). Retourne la perte de valeur par jour."""
        if self.T <= 0:
            return 0.0
        
        term1 = - (self.S0 * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            theta_annual = term1 - term2
        elif option_type == 'put':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            theta_annual = term1 + term2
        else:
            raise ValueError("option_type doit être 'call' ou 'put'")
            
        # On divise par 365 pour donner la perte de valeur par jour
        return theta_annual / 365.0
    

    @classmethod
    def implied_volatility(cls, market_price: float, S0: float, K: float, T: float, r: float, option_type: str = 'call', tol: float = 1e-5, max_iter: int = 100) -> float:
        """
        Calcule la volatilité implicite via l'algorithme de Newton-Raphson.
        Utilise @classmethod car on instancie des pricers temporaires pour converger sans modifier l'instance actuelle.
        """
        # Si l'option est expirée, la volatilité implicite n'a plus de sens
        if T <= 0:
            return 0.0

        # Point de départ (Initial guess) : 20% de volatilité
        sigma = 0.20 
        
        for i in range(max_iter):
            # 1. On crée un pricer temporaire avec le sigma actuel
            temp_pricer = cls(S0, K, T, r, sigma)
            
            # 2. On calcule le prix théorique avec ce sigma
            if option_type == 'call':
                price_calculated = temp_pricer.call_price()
            else:
                price_calculated = temp_pricer.put_price()
                
            # 3. L'erreur par rapport au marché
            diff = price_calculated - market_price
            
            # Si l'erreur est plus petite que la tolérance, on a trouvé !
            if abs(diff) < tol:
                return sigma
                
            # 4. On récupère le Vega (la dérivée)
            # Attention: dans notre classe, vega() renvoie le changement pour 1% (divisé par 100)
            # Pour Newton-Raphson (maths pures), il nous faut le vrai Vega brut (la dérivée partielle exacte)
            # Donc on le remultiplie par 100
            vega_exact = temp_pricer.vega() * 100.0
            
            # Sécurité pour éviter la division par zéro si Vega est nul (option très OTM/ITM)
            if vega_exact < 1e-8:
                # Si le gradient est nul, on retourne NaN car l'algo ne peut plus converger
                return np.nan 
                
            # 5. La mise à jour de Newton-Raphson
            sigma = sigma - (diff / vega_exact)
            
            # Sécurité : la volatilité ne peut pas être négative
            if sigma <= 0.0:
                sigma = 0.001
                
        # Si on dépasse max_iter sans converger
        return np.nan