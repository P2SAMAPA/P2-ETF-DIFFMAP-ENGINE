# portfolio.py

import numpy as np
from datetime import timedelta
from config import ALL_ETFS

TRANSACTION_COST = 0.0012  # 12 bps
TSL_THRESHOLD = -0.12
Z_EXIT = 0.75

class PortfolioState:
    def __init__(self):
        self.current_etf = None
        self.in_cash = False
        self.last_returns = []
    
    def update_returns(self, ret):
        self.last_returns.append(ret)
        if len(self.last_returns) > 2:
            self.last_returns.pop(0)

    def check_tsl(self):
        if len(self.last_returns) < 2:
            return False
        cum_ret = np.prod([1+r for r in self.last_returns]) - 1
        return cum_ret <= TSL_THRESHOLD

    def compute_zscore(self, samples):
        mu = samples.mean()
        std = samples.std() + 1e-6
        return mu / std

    def decide(self, predictions, samples_dict, tbill_daily):

        # TSL check
        if self.check_tsl():
            self.in_cash = True
            self.current_etf = "CASH"
            return "CASH", tbill_daily

        # If in CASH → wait for re-entry
        if self.in_cash:
            best_etf = max(predictions, key=predictions.get)
            z = self.compute_zscore(samples_dict[best_etf])

            if z >= Z_EXIT:
                self.in_cash = False
            else:
                return "CASH", tbill_daily

        # Normal selection
        best_etf = max(predictions, key=predictions.get)

        # Transaction cost penalty
        if self.current_etf and self.current_etf != best_etf:
            predictions[best_etf] -= TRANSACTION_COST

        self.current_etf = best_etf
        return best_etf, predictions[best_etf]
