import numpy as np
import cvxpy as cp

class Consumer:
    """
    A class that model a consumer in the Arrow-Debreu Economy
    """

    def __init__(self, budget, valuation, rho):
        self.budget = budget # consumer's budget
        self.valuation = valuation # consumer's valuation of goods
        self.rho = rho # parameter for the CES utility function

    def get_budget(self):
        """
        Return endowment of consumer
        """
        return self.budget
        

    def get_demand(self, prices):
        """
        Function that calculates the demand of the consumer for goods
        at gives prices
        """
        # Solution to the optimal budget constrained demand set of CES utilities
        # can be found Lagrangian methods analytically.
        c = (self.rho/ self.rho - 1)
        num = self.budget *((np.power(self.valuation, 1 - c) * np.power(prices, c - 1)))
        denom = np.sum((np.power(self.valuation, 1 - c) * np.power(prices, c )))
        demand = num/denom
        
        assert demand.shape[0] == self.valuation.shape[0]
        return demand
