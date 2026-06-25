import numpy as np

class MonteCarloGBM:
    """
    Générateur de trajectoires Monte Carlo utilisant le Mouvement Brownien Géométrique (GBM).
    Schéma de discrétisation : Euler-Maruyama.
    """
    def __init__(self, S0: float, mu: float, sigma: float, T: float, N: int, M: int, seed: int = None):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N     # Nombre de pas de temps (ex: 252)
        self.M = M     # Nombre de trajectoires (ex: 10000)
        self.dt = T / N
        self.seed = seed

    def simulate(self) -> np.ndarray:
        """
        Génère les trajectoires.
        Retourne une matrice NumPy de dimension (N, M).
        La ligne 0 n'est pas S0, c'est S0 * évolution du Jour 1.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Génération vectorisée des chocs aléatoires normaux standard
        Z = np.random.normal(loc=0.0, scale=1.0, size=(self.N, self.M))
        
        # Discrétisation d'Euler-Maruyama vectorisée
        step_factors = 1 + self.mu * self.dt + self.sigma * np.sqrt(self.dt) * Z
        
        # Calcul des prix via produit cumulé sur l'axe du temps (axis=0)
        trajectories = self.S0 * np.cumprod(step_factors, axis=0)
        
        return trajectories