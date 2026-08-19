import numpy as np
from src.stochastic import MonteCarloGBM
from src.options_pricer import BlackScholesPricer


class DeltaHedgingBacktester:
    """
    Simule l'Achat ou la Vente d'un Straddle et sa couverture en Delta quotidienne
    sur un chemin de prix simulé (Monte Carlo GBM).
    """
    def __init__(self, S0: float, K: float, T: float, r: float,
                 iv_market: float, rv_actual: float, N_days: int,
                 position: str = 'short'):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.iv = iv_market
        self.rv = rv_actual
        self.N_days = N_days
        self.dt = T / N_days
        self.position = position.lower()

    def run_simulation(self) -> dict:
        gbm = MonteCarloGBM(self.S0, mu=self.r, sigma=self.rv,
                            T=self.T, N=self.N_days, M=1)
        stock_path = gbm.simulate().flatten()

        pricer_init = BlackScholesPricer(self.S0, self.K, self.T, self.r, self.iv)
        premium = pricer_init.call_price() + pricer_init.put_price()

        if self.position == 'short':
            cash_account = premium
            direction_multiplier = -1.0
        elif self.position == 'long':
            cash_account = -premium
            direction_multiplier = 1.0
        else:
            raise ValueError("La position doit être 'short' ou 'long'")

        shares_held = 0.0

        for t in range(self.N_days):
            current_S = stock_path[t]
            time_to_maturity = max(self.T - (t * self.dt), 0.0001)

            pricer_t = BlackScholesPricer(current_S, self.K,
                                          time_to_maturity, self.r, self.iv)
            straddle_delta = pricer_t.delta('call') + pricer_t.delta('put')
            target_shares = -(straddle_delta * direction_multiplier)

            shares_to_trade = target_shares - shares_held
            cash_account -= shares_to_trade * current_S
            shares_held = target_shares

        final_S = stock_path[-1]
        cash_account += shares_held * final_S

        payoff_call = max(final_S - self.K, 0)
        payoff_put = max(self.K - final_S, 0)
        payout_total = payoff_call + payoff_put

        if self.position == 'short':
            cash_account -= payout_total
        else:
            cash_account += payout_total

        return {
            "Premium_Initial": premium if self.position == 'short' else -premium,
            "Payout_Final": -payout_total if self.position == 'short' else payout_total,
            "Final_PnL": cash_account
        }


class HistoricalDeltaHedgingBacktester:
    """
    Simule la Vente d'un Straddle et sa couverture en Delta sur un chemin de prix HISTORIQUE RÉEL.
    """
    def __init__(self, historical_prices: np.ndarray, K: float, T: float, r: float, 
                 iv_market: float, position: str = 'short'):
        self.prices = historical_prices
        self.S0 = historical_prices[0]
        self.K = K
        self.T = T
        self.r = r
        self.iv = iv_market  
        self.N_days = len(historical_prices) - 1
        self.dt = T / self.N_days
        self.position = position.lower()

    def run_backtest(self) -> dict:
        # Vente/Achat initial du Straddle pricé avec l'IV du marché
        pricer_init = BlackScholesPricer(self.S0, self.K, self.T, self.r, self.iv)
        premium = pricer_init.call_price() + pricer_init.put_price()
        
        if self.position == 'short':
            cash_account = premium
            direction_multiplier = -1.0 
        elif self.position == 'long':
            cash_account = -premium
            direction_multiplier = 1.0  
        else:
            raise ValueError("Position doit être 'short' ou 'long'")

        shares_held = 0.0

        # Boucle de Delta-Hedging sur les vrais prix historiques
        for t in range(self.N_days):
            current_S = self.prices[t]
            time_to_maturity = self.T - (t * self.dt)
            if time_to_maturity <= 0: time_to_maturity = 0.0001 

            pricer_t = BlackScholesPricer(current_S, self.K, time_to_maturity, self.r, self.iv)
            straddle_delta = pricer_t.delta('call') + pricer_t.delta('put')
            
            target_shares = - (straddle_delta * direction_multiplier)
            shares_to_trade = target_shares - shares_held
            cash_account -= shares_to_trade * current_S
            shares_held = target_shares

        # Dénouement à l'expiration
        final_S = self.prices[-1]
        cash_account += shares_held * final_S
        
        payout_total = max(final_S - self.K, 0) + max(self.K - final_S, 0)
        
        if self.position == 'short':
            cash_account -= payout_total
        else:
            cash_account += payout_total
            
        return {
            "Premium_Initial": premium if self.position == 'short' else -premium,
            "Payout_Final": -payout_total if self.position == 'short' else payout_total,
            "Final_PnL": cash_account
        }